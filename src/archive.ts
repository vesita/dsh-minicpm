/**
 * Minimal, dependency-free `.tar.gz` extractor.
 *
 * The engine fetch downloads a llama.cpp release archive, and pulling in an npm
 * tarball library (or shelling out to `tar`) for one operation is not worth the
 * supply-chain or portability cost. This reader covers exactly the subset those
 * archives use — regular files, directories, symbolic and hard links, plus the
 * GNU long-name and PAX extensions — and refuses anything that would escape the
 * destination directory.
 *
 * @module dsh-minicpm/archive
 */
import fs from 'node:fs'
import path from 'node:path'
import zlib from 'node:zlib'

/** One parsed tar entry header. */
interface TarEntry {
  name: string
  size: number
  type: string
  linkName: string
  mode: number
}

/** Number of bytes in one tar block. */
const BLOCK = 512

/**
 * Extract a gzipped tar archive into a directory.
 *
 * @param archive - the complete `.tar.gz` bytes.
 * @param destDir - directory to extract into; created when absent.
 * @returns the absolute paths written, in archive order.
 * @throws {Error} when an entry would escape `destDir`.
 */
export function extractTarGz(archive: Buffer, destDir: string): string[] {
  const tar = zlib.gunzipSync(archive)
  return extractTar(tar, destDir)
}

/**
 * Extract an uncompressed tar image into a directory.
 *
 * @param tar - the complete tar bytes.
 * @param destDir - directory to extract into; created when absent.
 * @returns the absolute paths written, in archive order.
 */
export function extractTar(tar: Buffer, destDir: string): string[] {
  fs.mkdirSync(destDir, { recursive: true })
  const root = fs.realpathSync(destDir)
  const written: string[] = []

  let offset = 0
  let pendingLongName: string | null = null
  let pendingPax: Record<string, string> = {}
  /** Hard links may point at an entry that already landed. */
  const seen = new Map<string, string>()

  while (offset + BLOCK <= tar.length) {
    const header = tar.subarray(offset, offset + BLOCK)
    offset += BLOCK

    // Two consecutive zero blocks end the archive.
    if (isZeroBlock(header)) break
    if (!verifyChecksum(header)) throw new Error(`tar 头部校验失败 @${offset - BLOCK}`)

    const entry = readHeader(header)
    const payload = tar.subarray(offset, offset + align(entry.size))
    offset += align(entry.size)

    // GNU and PAX both defer naming to a preceding metadata entry.
    if (entry.type === 'L') {
      pendingLongName = payload.subarray(0, entry.size).toString('utf8').replace(/\0+$/, '')
      continue
    }
    if (entry.type === 'x' || entry.type === 'g') {
      pendingPax = { ...pendingPax, ...parsePax(payload.subarray(0, entry.size).toString('utf8')) }
      continue
    }

    const name = resolveName(entry.name, pendingLongName, pendingPax.path)
    const linkName = pendingPax.linkpath || entry.linkName
    pendingLongName = null
    pendingPax = {}

    const target = safeJoin(root, name)
    if (target === null) throw new Error(`tar 条目越出目标目录：${name}`)
    written.push(target)

    switch (entry.type) {
      case '5': {
        fs.mkdirSync(target, { recursive: true })
        break
      }
      case '2': {
        fs.mkdirSync(path.dirname(target), { recursive: true })
        fs.rmSync(target, { force: true })
        fs.symlinkSync(linkName, target)
        break
      }
      case '1': {
        // Hard links inside a release archive always point at an earlier entry.
        const source = safeJoin(root, linkName)
        fs.mkdirSync(path.dirname(target), { recursive: true })
        fs.rmSync(target, { force: true })
        if (source !== null && fs.existsSync(source)) {
          fs.linkSync(source, target)
        } else {
          const existing = seen.get(linkName)
          if (existing !== undefined && fs.existsSync(existing)) fs.linkSync(existing, target)
        }
        break
      }
      case '0':
      case '7':
      case '\0':
      case '': {
        fs.mkdirSync(path.dirname(target), { recursive: true })
        fs.writeFileSync(target, payload.subarray(0, entry.size), { mode: entry.mode & 0o777 })
        seen.set(name, target)
        break
      }
      default: {
        // Character/block devices, FIFOs and anything else a release archive
        // should never contain are skipped rather than created.
        written.pop()
        break
      }
    }
  }

  return written
}

/** Read one 512-byte header into structured form. */
function readHeader(block: Buffer): TarEntry {
  return {
    name: readString(block, 0, 100),
    mode: readOctal(block, 100, 8),
    size: readOctal(block, 124, 12),
    type: String.fromCharCode(block[156] || 0),
    linkName: readString(block, 157, 100),
    // ustar splits long paths across `prefix` and `name`.
    ...(readString(block, 257, 6) === 'ustar' && readString(block, 345, 155)
      ? { name: `${readString(block, 345, 155)}/${readString(block, 0, 100)}` }
      : {})
  }
}

/** A NUL-terminated string field. */
function readString(block: Buffer, start: number, length: number): string {
  const slice = block.subarray(start, start + length)
  const end = slice.indexOf(0)
  return slice.subarray(0, end === -1 ? slice.length : end).toString('utf8').trim()
}

/** A NUL-or-space-terminated octal numeric field. */
function readOctal(block: Buffer, start: number, length: number): number {
  const raw = readString(block, start, length).replace(/\0/g, '').trim()
  if (raw === '') return 0
  // GNU base-256 encoding for values that do not fit octal.
  if ((block[start] & 0x80) !== 0) {
    let value = 0
    for (let index = 0; index < length; index += 1) value = value * 256 + block[start + index]
    return value
  }
  const parsed = parseInt(raw, 8)
  return Number.isFinite(parsed) ? parsed : 0
}

/** Whether a block is entirely zero, which terminates the archive. */
function isZeroBlock(block: Buffer): boolean {
  for (let index = 0; index < block.length; index += 1) if (block[index] !== 0) return false
  return true
}

/** Verify the header checksum, which is what distinguishes a real header. */
function verifyChecksum(block: Buffer): boolean {
  const stored = readOctal(block, 148, 8)
  if (stored === 0) return false
  let sum = 0
  for (let index = 0; index < BLOCK; index += 1) {
    // The checksum field itself counts as spaces.
    sum += index >= 148 && index < 156 ? 32 : block[index]
  }
  return sum === stored
}

/** Parse PAX extended-header records into a key/value map. */
function parsePax(text: string): Record<string, string> {
  const out: Record<string, string> = {}
  let offset = 0
  while (offset < text.length) {
    const space = text.indexOf(' ', offset)
    if (space === -1) break
    const length = parseInt(text.slice(offset, space), 10)
    if (!Number.isFinite(length) || length <= 0) break
    const record = text.slice(space + 1, offset + length).replace(/\n$/, '')
    const equals = record.indexOf('=')
    if (equals > 0) out[record.slice(0, equals)] = record.slice(equals + 1)
    offset += length
  }
  return out
}

/** Choose the effective entry name across the plain, GNU and PAX forms. */
function resolveName(name: string, longName: string | null, paxPath: string | undefined): string {
  return paxPath || longName || name
}

/**
 * Resolve an archive path under a root, refusing escapes.
 *
 * @returns the absolute target, or null when the entry leaves the root.
 */
function safeJoin(root: string, name: string): string | null {
  const cleaned = name.replace(/^\.\//, '').replace(/^\/+/, '')
  if (cleaned === '' || cleaned.includes('\0')) return null
  const target = path.resolve(root, cleaned)
  if (target !== root && !target.startsWith(root + path.sep)) return null
  return target
}

/** Round a size up to the next block boundary. */
function align(size: number): number {
  return Math.ceil(size / BLOCK) * BLOCK
}
