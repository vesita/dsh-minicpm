/**
 * Weight and runtime acquisition, with observable progress.
 *
 * Both downloads are long (the 4-bit weight is ~1.6 GB) and both are things a
 * user starts from the settings card and then watches. So they run as *jobs*:
 * a registry holds their live progress, the browser polls it, and a job can be
 * cancelled without leaving a half-written file behind.
 *
 * @module dsh-minicpm/fetch
 */
import fs from 'node:fs'
import path from 'node:path'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { extractTarGz } from './archive.js'
import { engineDir, modelsDir } from './paths.js'
import { HF_REPO, remoteFileOf } from './models.js'
import type { ModelSpec } from './models.js'

const run = promisify(execFile)

/** What a job is acquiring. */
export type JobKind = 'model' | 'engine'

/** Lifecycle of one acquisition. */
export type JobStatus = 'running' | 'done' | 'error' | 'cancelled'

/** One observable acquisition. */
export interface Job {
  /** Stable id the UI polls with. */
  id: string
  kind: JobKind
  /** Human-readable name of what is being fetched. */
  label: string
  status: JobStatus
  /** Bytes written so far. */
  doneBytes: number
  /** Expected total, when the server disclosed one. */
  totalBytes: number | null
  /** Epoch ms the job started. */
  startedAt: number
  /** Epoch ms the job settled, null while running. */
  finishedAt: number | null
  /** Failure text, empty unless `status` is `error`. */
  error: string | null
  /** Current phase, for a progress line under the bar. */
  detail: string | null
  /** Bytes per second averaged over the job. */
  bytesPerSecond: number | null
  /** Absolute path the job wrote, once known. */
  resultPath: string | null
}

/** A live job the caller can watch and cancel. */
export interface JobHandle {
  /** Snapshot of the job's current state. */
  readonly id: string
  /** Resolves when the job settles, successfully or not. */
  readonly completion: Promise<string>
  /** Ask the job to stop; the partial file is removed. */
  cancel(): void
}

/** In-memory registry of running and recently finished jobs. */
export class JobRegistry {
  readonly #jobs = new Map<string, Job>()
  readonly #controllers = new Map<string, AbortController>()
  /** Counter used to build stable ids. */
  #sequence = 0

  /** Every job, newest first. */
  list(): Job[] {
    return [...this.#jobs.values()].map(job => ({ ...job })).sort((a, b) => b.startedAt - a.startedAt)
  }

  /** One job by id, or null. */
  get(id: string): Job | null {
    const job = this.#jobs.get(id)
    return job === undefined ? null : { ...job }
  }

  /** The most recent job of one kind, or null. */
  latest(kind: JobKind): Job | null {
    return this.list().find(job => job.kind === kind) ?? null
  }

  /** Whether any job of one kind is still running. */
  busy(kind: JobKind): boolean {
    return this.list().some(job => job.kind === kind && job.status === 'running')
  }

  /**
   * Ask one running job to stop.
   *
   * @param id - job to cancel.
   * @returns true when a running job was signalled, false when the id is
   *   unknown or the job had already settled.
   */
  cancel(id: string): boolean {
    const controller = this.#controllers.get(id)
    if (controller === undefined) return false
    controller.abort()
    return true
  }

  /**
   * Start one tracked acquisition.
   *
   * @param kind - what is being acquired.
   * @param label - display name.
   * @param work - receives the job's progress reporter and abort signal, and
   *   returns the absolute path it produced.
   * @returns a handle the caller can await or cancel.
   */
  start(
    kind: JobKind,
    label: string,
    work: (report: (done: number, total: number | null, detail?: string) => void, signal: AbortSignal) => Promise<string>
  ): JobHandle {
    this.#sequence += 1
    const id = `${kind}-${Date.now().toString(36)}-${this.#sequence}`
    const job: Job = {
      id,
      kind,
      label,
      status: 'running',
      doneBytes: 0,
      totalBytes: null,
      startedAt: Date.now(),
      finishedAt: null,
      error: null,
      detail: null,
      bytesPerSecond: null,
      resultPath: null
    }
    this.#jobs.set(id, job)

    const controller = new AbortController()
    this.#controllers.set(id, controller)

    const report = (done: number, total: number | null, detail?: string): void => {
      job.doneBytes = done
      job.totalBytes = total
      if (detail !== undefined) job.detail = detail
      const elapsed = (Date.now() - job.startedAt) / 1000
      job.bytesPerSecond = elapsed > 0.5 ? Math.round(done / elapsed) : null
    }

    const completion = (async () => {
      try {
        const result = await work(report, controller.signal)
        job.status = 'done'
        job.resultPath = result
        job.detail = null
        return result
      } catch (error) {
        // A cancel is a settled outcome, not a failure the UI should redden.
        if (controller.signal.aborted) {
          job.status = 'cancelled'
          job.error = null
        } else {
          job.status = 'error'
          job.error = error instanceof Error ? error.message : String(error)
        }
        throw error
      } finally {
        job.finishedAt = Date.now()
        this.#controllers.delete(id)
        this.#prune()
      }
    })()

    // Keep the registry from rejecting when nobody awaits the handle.
    completion.catch(() => {})

    return {
      id,
      completion,
      cancel: () => {
        if (job.status !== 'running') return
        controller.abort()
      }
    }
  }

  /** Keep only the most recent few finished jobs. */
  #prune(): void {
    const finished = this.list().filter(job => job.status !== 'running')
    for (const job of finished.slice(8)) this.#jobs.delete(job.id)
  }

  /** Abort every running job, for plugin teardown. */
  dispose(): void {
    for (const controller of this.#controllers.values()) controller.abort()
    this.#controllers.clear()
  }
}

/** Options shared by both downloads. */
export interface FetchOptions {
  /** Base URL for Hugging Face; a mirror keeps large pulls off the public CDN. */
  hfEndpoint?: string
  /** Pinned llama.cpp build tag, e.g. `b10951`; latest is discovered when unset. */
  engineBuild?: string
  /** How many release assets to inspect when discovering the newest build. */
  releaseScanDepth?: number
}

/**
 * Download a model's weights into the plugin's models directory.
 *
 * @param spec - the catalogue entry to fetch.
 * @param options - mirror and engine preferences.
 * @param report - progress reporter.
 * @param signal - cancellation.
 * @returns the absolute path of the written weight file.
 */
export async function fetchModel(
  spec: ModelSpec,
  options: FetchOptions,
  report: (done: number, total: number | null, detail?: string) => void,
  signal: AbortSignal
): Promise<string> {
  const endpoint = (options.hfEndpoint || process.env.HF_ENDPOINT || 'https://huggingface.co').replace(/\/+$/, '')
  const file = remoteFileOf(spec)
  const url = `${endpoint}/${HF_REPO}/resolve/main/${file}`
  const destination = path.isAbsolute(spec.file) ? spec.file : path.join(modelsDir(), spec.file)
  fs.mkdirSync(path.dirname(destination), { recursive: true })
  report(0, spec.sizeBytes ?? null, `下载 ${file}`)
  await downloadToFile(url, destination, report, signal)
  return destination
}

/**
 * Download and unpack the llama.cpp runtime.
 *
 * The Vulkan build is preferred on Linux and Windows because it needs no CUDA
 * toolkit while still using the GPU, which is exactly the situation of a
 * machine that has a driver but no development install.
 *
 * @param options - pinned build and release-scan preferences.
 * @param report - progress reporter.
 * @param signal - cancellation.
 * @returns the absolute path of the `llama-server` binary.
 */
export async function fetchEngine(
  options: FetchOptions,
  report: (done: number, total: number | null, detail?: string) => void,
  signal: AbortSignal
): Promise<string> {
  const target = engineTarget()
  report(0, null, '查询最新 llama.cpp 版本…')
  const asset = await resolveEngineAsset(target, options, signal)

  const archive = path.join(engineDir(), `.download-${path.basename(asset.url)}`)
  fs.mkdirSync(engineDir(), { recursive: true })
  report(0, asset.size ?? null, `下载 ${path.basename(asset.url)}`)
  await downloadToFile(asset.url, archive, report, signal)

  if (signal.aborted) throw new Error('已取消')
  report(asset.size ?? 0, asset.size ?? null, '解压运行时…')

  const bytes = fs.readFileSync(archive)
  fs.rmSync(archive, { force: true })

  // Stage into a clean directory, then move into place, so a failed extraction
  // never leaves a half-populated runtime that looks installed.
  const staging = path.join(engineDir(), '.staging')
  fs.rmSync(staging, { recursive: true, force: true })
  fs.mkdirSync(staging, { recursive: true })
  extractTarGz(bytes, staging)

  // Release archives wrap everything in one top-level directory; flatten it.
  const entries = fs.readdirSync(staging)
  const root = entries.length === 1 && fs.statSync(path.join(staging, entries[0])).isDirectory()
    ? path.join(staging, entries[0])
    : staging
  for (const entry of fs.readdirSync(root)) {
    const from = path.join(root, entry)
    const to = path.join(engineDir(), entry)
    fs.rmSync(to, { recursive: true, force: true })
    fs.renameSync(from, to)
  }
  fs.rmSync(staging, { recursive: true, force: true })

  const binary = path.join(engineDir(), target.binary)
  if (!fs.existsSync(binary)) {
    throw new Error(`解压完成但未找到 ${target.binary}，请检查 ${engineDir()} 的内容`)
  }
  fs.chmodSync(binary, 0o755)
  report(asset.size ?? 0, asset.size ?? null, '运行时已就绪')
  return binary
}

/** Which release asset a platform needs. */
interface EngineTarget {
  /** Regexes tried in order against release asset names. */
  patterns: RegExp[]
  /** Binary name inside the archive. */
  binary: string
  /** Human-readable platform description. */
  label: string
}

/**
 * Decide which llama.cpp release asset fits this machine.
 *
 * @returns the asset patterns, in preference order.
 */
export function engineTarget(): EngineTarget {
  const platform = process.platform
  const arch = process.arch
  const binary = platform === 'win32' ? 'llama-server.exe' : 'llama-server'

  if (platform === 'linux' && arch === 'x64') {
    return {
      patterns: [/bin-ubuntu-vulkan-x64\.tar\.gz$/, /bin-ubuntu-x64\.tar\.gz$/],
      binary,
      label: 'Linux x64'
    }
  }
  if (platform === 'linux' && arch === 'arm64') {
    return { patterns: [/bin-ubuntu-arm64\.tar\.gz$/], binary, label: 'Linux arm64' }
  }
  if (platform === 'darwin') {
    return { patterns: [new RegExp(`bin-macos-${arch}\\.tar\\.gz$`)], binary, label: `macOS ${arch}` }
  }
  if (platform === 'win32' && arch === 'x64') {
    return { patterns: [/bin-win-cpu-x64\.zip$/, /bin-win-vulkan-x64\.zip$/], binary, label: 'Windows x64' }
  }
  throw new Error(`暂不支持自动获取推理引擎的平台：${platform}/${arch}，请手动安装 llama.cpp 并在设置中指定 engine.binary`)
}

/** One resolved release asset. */
interface ResolvedAsset {
  url: string
  size: number | null
  build: string
}

/**
 * Find the newest release asset matching this platform.
 *
 * `/releases/latest` is deliberately not used: llama.cpp's newest *non-*prerelease
 * tag is a version marker that carries no binaries, so the newest release that
 * actually ships the needed asset is selected from the recent list instead.
 *
 * @param target - platform asset requirements.
 * @param options - pinned build, when the user named one.
 * @param signal - cancellation.
 * @returns the chosen asset.
 */
async function resolveEngineAsset(
  target: EngineTarget,
  options: FetchOptions,
  signal: AbortSignal
): Promise<ResolvedAsset> {
  const pinned = options.engineBuild
  if (typeof pinned === 'string' && pinned.length > 0) {
    const release = await githubJson(`https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/${pinned}`, signal)
    const found = pickAsset(release, target)
    if (found === null) throw new Error(`llama.cpp ${pinned} 没有适用于 ${target.label} 的产物`)
    return { ...found, build: pinned }
  }

  const depth = options.releaseScanDepth ?? 15
  const releases = await githubJson(
    `https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=${depth}`,
    signal
  )
  if (!Array.isArray(releases)) throw new Error('无法读取 llama.cpp 发布列表')
  for (const release of releases) {
    const found = pickAsset(release, target)
    if (found !== null) return { ...found, build: String(release.tag_name ?? '') }
  }
  throw new Error(`最近 ${depth} 个 llama.cpp 发布中都没有适用于 ${target.label} 的产物，请用 engine.build 指定版本`)
}

/** Choose the first matching asset from one release. */
function pickAsset(release: any, target: EngineTarget): { url: string; size: number | null } | null {
  const assets: any[] = Array.isArray(release?.assets) ? release.assets : []
  for (const pattern of target.patterns) {
    const match = assets.find(asset => pattern.test(String(asset?.name ?? '')))
    if (match) {
      return {
        url: String(match.browser_download_url),
        size: Number.isFinite(match.size) ? Number(match.size) : null
      }
    }
  }
  return null
}

/** GET one GitHub API URL as JSON, with a helpful error on refusal. */
async function githubJson(url: string, signal: AbortSignal): Promise<any> {
  const response = await fetch(url, {
    headers: { accept: 'application/vnd.github+json', 'user-agent': 'dsh-minicpm' },
    signal
  })
  if (!response.ok) {
    if (response.status === 403) {
      throw new Error('GitHub API 拒绝了请求（可能是速率限制）。请稍后重试，或用 engine.build 指定版本号。')
    }
    throw new Error(`GitHub API 返回 ${response.status}`)
  }
  return response.json()
}

/**
 * Stream one URL into a file, resuming a partial download when possible.
 *
 * A `.part` sibling holds the bytes until the transfer completes, so an
 * interrupted download resumes from where it stopped instead of restarting a
 * gigabyte of transfer. The partial is removed on any failure, which keeps a
 * truncated file from ever being mistaken for a usable one.
 *
 * @param url - source URL.
 * @param destination - final path.
 * @param report - progress reporter.
 * @param signal - cancellation.
 */
export async function downloadToFile(
  url: string,
  destination: string,
  report: (done: number, total: number | null, detail?: string) => void,
  signal: AbortSignal
): Promise<void> {
  const part = `${destination}.part`
  let offset = 0
  try {
    offset = fs.statSync(part).size
  } catch {
    offset = 0
  }

  const headers: Record<string, string> = { 'user-agent': 'dsh-minicpm' }
  if (offset > 0) headers.range = `bytes=${offset}-`

  let response: Response
  try {
    response = await fetch(url, { headers, signal, redirect: 'follow' })
  } catch (error) {
    if (signal.aborted) throw new Error('已取消')
    throw new Error(`下载失败：${error instanceof Error ? error.message : String(error)}`)
  }

  if (!response.ok) {
    throw new Error(`下载 ${url} 返回 ${response.status} ${response.statusText}`)
  }

  // A server that ignores the range answers 200 with the whole body, so the
  // partial must be discarded rather than appended to.
  if (offset > 0 && response.status !== 206) {
    offset = 0
    fs.rmSync(part, { force: true })
  }

  const lengthHeader = response.headers.get('content-length')
  const remaining = lengthHeader === null ? null : Number(lengthHeader)
  const total = remaining === null || !Number.isFinite(remaining) ? null : offset + remaining

  const stream = fs.createWriteStream(part, { flags: offset > 0 ? 'a' : 'w' })
  let written = offset
  report(written, total)

  try {
    if (response.body === null) throw new Error('响应没有内容')
    for await (const chunk of response.body as unknown as AsyncIterable<Uint8Array>) {
      if (signal.aborted) throw new Error('已取消')
      const buffer = Buffer.from(chunk)
      if (!stream.write(buffer)) {
        await new Promise<void>(resolve => stream.once('drain', resolve))
      }
      written += buffer.length
      report(written, total)
    }
    await new Promise<void>((resolve, reject) => {
      stream.end((error?: Error | null) => (error ? reject(error) : resolve()))
    })
  } catch (error) {
    stream.destroy()
    fs.rmSync(part, { force: true })
    throw error instanceof Error && error.message === '已取消' ? error : new Error(describe(error))
  }

  if (total !== null && written < total) {
    fs.rmSync(part, { force: true })
    throw new Error(`下载不完整：${written}/${total} 字节`)
  }
  fs.renameSync(part, destination)
}

/** Turn a stream error into something a settings card can show. */
function describe(error: unknown): string {
  return `下载中断：${error instanceof Error ? error.message : String(error)}`
}

/**
 * Run a command, resolving to its stdout.
 *
 * @param file - executable.
 * @param args - arguments.
 * @returns stdout text.
 */
export async function runCommand(file: string, args: string[]): Promise<string> {
  const { stdout } = await run(file, args, { timeout: 10_000 })
  return String(stdout)
}
