#!/usr/bin/env node
/**
 * `dsh-minicpm` — standalone control of the local MiniCPM engine.
 *
 * The settings card is the normal way to drive this plugin, but two situations
 * want a terminal: verifying an installation before a session exists, and
 * driving the engine from a script. This CLI shares the plugin's own engine
 * manager and fetchers, so what it reports is exactly what a session would see.
 *
 * It deliberately touches no settings file: `--port` and the `DSH_MINICPM_*`
 * environment variables are the only configuration, and the defaults match the
 * plugin's.
 *
 * @module dsh-minicpm/bin
 */
import fs from 'node:fs'
import path from 'node:path'
import { EngineManager } from './engine.js'
import type { EngineConfig } from './engine.js'
import { JobRegistry, fetchEngine, fetchModel } from './fetch.js'
import { MODEL_CATALOG, absoluteFileOf, resolveModelSpec } from './models.js'
import type { ModelSpec } from './models.js'
import { engineDir, modelsDir, rootDir } from './paths.js'

/** Resolve the engine configuration from flags and environment. */
function engineConfig(overrides: Partial<EngineConfig> = {}): EngineConfig {
  const env = process.env
  return {
    mode: overrides.mode ?? (env.DSH_MINICPM_MODE === 'external' ? 'external' : 'managed'),
    baseURL: overrides.baseURL ?? env.DSH_MINICPM_BASE_URL,
    binary: overrides.binary ?? env.DSH_MINICPM_BINARY,
    host: overrides.host ?? env.DSH_MINICPM_HOST ?? '127.0.0.1',
    port: overrides.port ?? numberOr(env.DSH_MINICPM_PORT, 8081),
    contextSize: overrides.contextSize ?? numberOr(env.DSH_MINICPM_CTX, 32768),
    gpuLayers: overrides.gpuLayers ?? numberOr(env.DSH_MINICPM_GPU_LAYERS, 999),
    threads: overrides.threads,
    idleUnloadSeconds: overrides.idleUnloadSeconds ?? numberOr(env.DSH_MINICPM_IDLE_UNLOAD, 900),
    extraArgs: overrides.extraArgs ?? [],
    preserveReasoning: overrides.preserveReasoning ?? true
  }
}

/** Parse a positive integer from the environment, or fall back. */
function numberOr(raw: string | undefined, fallback: number): number {
  const parsed = Number(raw)
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback
}

/** Options parsed from the command line. */
interface Flags {
  command: string
  positionals: string[]
  options: Map<string, string>
}

/** Parse `--key value`, `--key=value` and bare positionals. */
function parseFlags(argv: string[]): Flags {
  const options = new Map<string, string>()
  const positionals: string[] = []
  let command = ''
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index]
    if (token.startsWith('--')) {
      const equals = token.indexOf('=')
      if (equals > 0) {
        options.set(token.slice(2, equals), token.slice(equals + 1))
      } else {
        const next = argv[index + 1]
        if (next !== undefined && !next.startsWith('--')) {
          options.set(token.slice(2), next)
          index += 1
        } else {
          options.set(token.slice(2), 'true')
        }
      }
      continue
    }
    if (command === '' && positionals.length === 0) command = token
    else positionals.push(token)
  }
  return { command, positionals, options }
}

/** The catalogue entry a command should act on. */
function pickModel(flags: Flags, explicit?: string): ModelSpec {
  const id = explicit ?? flags.options.get('model') ?? MODEL_CATALOG[0].id
  const catalog = flags.options.get('file')
    ? [
        {
          id: flags.options.get('id') || 'custom',
          name: flags.options.get('name') || path.basename(flags.options.get('file')!),
          file: flags.options.get('file')!
        }
      ]
    : MODEL_CATALOG
  return resolveModelSpec(id, catalog)
}

/** Print usage. */
function usage(): void {
  process.stdout.write(
    [
      'dsh-minicpm — MiniCPM5-2B 本地引擎管理',
      '',
      '用法：dsh-minicpm <命令> [选项]',
      '',
      '命令：',
      '  status                显示引擎与显存状态',
      '  start                 启动引擎（加载默认模型）',
      '  stop                  停止引擎并释放显存',
      '  models                列出目录中的权重文件',
      '  fetch-model [id]      下载权重（默认 Q4_K_M，约 1.6 GB）',
      '  fetch-engine          下载 llama.cpp 运行时',
      '  chat <文本>           直接对本地引擎发一次请求，用于验证链路',
      '  doctor                检查运行环境（GPU、运行时、权重、端口）',
      '',
      '选项：',
      '  --port N              引擎端口（默认 8081）',
      '  --ctx N               上下文长度（默认 32768）',
      '  --model PATH          指定 .gguf 绝对路径',
      '  --base-url URL        外部模式下的端点',
      '  --mode managed|external',
      '  --build bNNNNN        固定 llama.cpp 版本',
      '  --hf-endpoint URL     Hugging Face 镜像',
      '',
      '环境变量：DSH_MINICPM_HOME / _PORT / _CTX / _MODEL / _BINARY / _BASE_URL / _IDLE_UNLOAD',
      ''
    ].join('\n')
  )
}

/** `status` — engine state plus GPU memory. */
async function commandStatus(flags: Flags): Promise<number> {
  const config = engineConfig({ port: flagPort(flags) })
  const engine = new EngineManager(() => config, { warn: message => process.stderr.write(`${message}\n`) })
  try {
    await engine.refreshHealth()
    const status = engine.status()
    process.stdout.write(`引擎状态：${status.running ? '运行中' : status.starting ? '启动中' : '未启动'}\n`)
    process.stdout.write(`  端点      ${status.baseURL}\n`)
    process.stdout.write(`  进程      ${status.pid ?? '—'}\n`)
    process.stdout.write(`  已载模型  ${status.loadedModel ?? '—'}\n`)
    process.stdout.write(`  运行时    ${status.binaryPresent ? status.binaryPath : `缺失（${status.binaryPath}）`}\n`)
    if (status.lastError) process.stdout.write(`  最近错误  ${status.lastError}\n`)
    const { gpuMemory } = await import('./engine.js')
    const gpu = await gpuMemory()
    if (gpu.totalMiB !== null) {
      process.stdout.write(`显存      ${gpu.name ?? ''} ${formatMiB(gpu.usedMiB)} / ${formatMiB(gpu.totalMiB)}\n`)
    } else {
      process.stdout.write('显存      不可用（未找到 nvidia-smi）\n')
    }
    return 0
  } finally {
    await engine.dispose()
  }
}

/** `start` — ensure the engine is serving the chosen weight. */
async function commandStart(flags: Flags): Promise<number> {
  const spec = pickModel(flags)
  const config = engineConfig({ port: flagPort(flags) })
  const engine = new EngineManager(() => config, { warn: message => process.stderr.write(`${message}\n`) })
  try {
    const baseURL = await engine.ensure(absoluteFileOf(spec))
    process.stdout.write(`引擎已就绪：${baseURL}（模型 ${path.basename(spec.file)}）\n`)
    process.stdout.write('注意：本命令启动的进程随本命令退出而结束；常驻请改用 DSH 设置页的「启动引擎」。\n')
    return 0
  } finally {
    // A CLI-started engine is torn down with the CLI: leaving a detached
    // server behind would hold VRAM that nothing tracks.
    await engine.dispose()
  }
}

/** `stop` — nothing to do beyond reporting, since the CLI owns no daemon. */
async function commandStop(): Promise<number> {
  process.stdout.write('dsh-minicpm 不维护后台守护进程：请在 DSH 设置页点击「停止引擎」，或结束当前的 DSH 进程。\n')
  return 0
}

/** `models` — list local weights. */
function commandModels(): number {
  process.stdout.write(`权重目录：${modelsDir()}\n`)
  let entries: string[] = []
  try {
    entries = fs.readdirSync(modelsDir()).filter(name => name.toLowerCase().endsWith('.gguf'))
  } catch {
    entries = []
  }
  if (entries.length === 0) {
    process.stdout.write('  （空）运行 dsh-minicpm fetch-model 下载默认权重\n')
    return 0
  }
  for (const name of entries) {
    const size = fs.statSync(path.join(modelsDir(), name)).size
    const id = MODEL_CATALOG.find(spec => path.basename(spec.file) === name)?.id ?? '—'
    process.stdout.write(`  ${name}  ${(size / 1024 ** 3).toFixed(2)} GB  模型 id: ${id}\n`)
  }
  return 0
}

/** `fetch-model` — download weights with a live progress line. */
async function commandFetchModel(flags: Flags): Promise<number> {
  const spec = pickModel(flags, flags.positionals[0])
  const jobs = new JobRegistry()
  process.stdout.write(`下载 ${spec.name} → ${absoluteFileOf(spec)}\n`)
  const handle = jobs.start('model', spec.name, (report, signal) =>
    fetchModel(
      spec,
      { hfEndpoint: flags.options.get('hf-endpoint') || process.env.HF_ENDPOINT },
      report,
      signal
    )
  )
  const stop = watch(handle.id, jobs)
  try {
    const result = await handle.completion
    stop()
    process.stdout.write(`\n完成：${result}\n`)
    return 0
  } catch (error) {
    stop()
    process.stderr.write(`\n失败：${error instanceof Error ? error.message : String(error)}\n`)
    return 1
  }
}

/** `fetch-engine` — download and unpack the llama.cpp runtime. */
async function commandFetchEngine(flags: Flags): Promise<number> {
  const jobs = new JobRegistry()
  process.stdout.write(`下载 llama.cpp 运行时 → ${engineDir()}\n`)
  const handle = jobs.start('engine', 'llama.cpp', (report, signal) =>
    fetchEngine(
      { engineBuild: flags.options.get('build') || undefined, hfEndpoint: flags.options.get('hf-endpoint') },
      report,
      signal
    )
  )
  const stop = watch(handle.id, jobs)
  try {
    const result = await handle.completion
    stop()
    process.stdout.write(`\n完成：${result}\n`)
    return 0
  } catch (error) {
    stop()
    process.stderr.write(`\n失败：${error instanceof Error ? error.message : String(error)}\n`)
    return 1
  }
}

/** Render one job's progress on a single rewriting line. */
function watch(id: string, jobs: JobRegistry): () => void {
  const timer = setInterval(() => {
    const job = jobs.get(id)
    if (job === null || job.status !== 'running') return
    const percent = job.totalBytes ? ` ${Math.round((job.doneBytes / job.totalBytes) * 100)}%` : ''
    const rate = job.bytesPerSecond ? ` ${(job.bytesPerSecond / 1024 ** 2).toFixed(1)} MB/s` : ''
    process.stdout.write(`\r  ${formatMiB(job.doneBytes / 1024 ** 2)}${job.totalBytes ? ` / ${formatMiB(job.totalBytes / 1024 ** 2)}` : ''}${percent}${rate}   `)
  }, 500)
  timer.unref?.()
  return () => clearInterval(timer)
}

/** `chat` — one request through the engine, streaming to stdout. */
async function commandChat(flags: Flags): Promise<number> {
  const prompt = flags.positionals.join(' ').trim()
  if (prompt === '') {
    process.stderr.write('用法：dsh-minicpm chat <文本>\n')
    return 2
  }
  const spec = pickModel(flags)
  const config = engineConfig({ port: flagPort(flags) })
  const engine = new EngineManager(() => config, { warn: message => process.stderr.write(`${message}\n`) })
  try {
    const baseURL = await engine.ensure(absoluteFileOf(spec))
    const response = await fetch(`${baseURL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: spec.id,
        messages: [{ role: 'user', content: prompt }],
        stream: true,
        stream_options: { include_usage: true },
        max_tokens: numberOr(flags.options.get('max-tokens'), 512)
      })
    })
    if (!response.ok || response.body === null) {
      process.stderr.write(`引擎返回 ${response.status}\n`)
      return 1
    }
    const decoder = new TextDecoder()
    let buffer = ''
    for await (const chunk of response.body as unknown as AsyncIterable<Uint8Array>) {
      buffer += decoder.decode(chunk, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed.startsWith('data:')) continue
        const raw = trimmed.slice(5).trim()
        if (raw === '' || raw === '[DONE]') continue
        try {
          const event = JSON.parse(raw)
          const delta = event.choices?.[0]?.delta
          if (delta?.content) process.stdout.write(String(delta.content))
          if (event.usage) {
            process.stdout.write(
              `\n\n[用量] 输入 ${event.usage.prompt_tokens ?? 0} · 输出 ${event.usage.completion_tokens ?? 0}\n`
            )
          }
        } catch {
          /* a partial frame; the next read completes it */
        }
      }
    }
    return 0
  } finally {
    await engine.dispose()
  }
}

/** `doctor` — check every precondition in one pass. */
async function commandDoctor(flags: Flags): Promise<number> {
  const config = engineConfig({ port: flagPort(flags) })
  const engine = new EngineManager(() => config, { warn: message => process.stderr.write(`${message}\n`) })
  const { gpuMemory } = await import('./engine.js')
  let failures = 0

  process.stdout.write(`DSH_MINICPM 根目录  ${rootDir()}\n`)
  process.stdout.write(`权重目录            ${modelsDir()}${fs.existsSync(modelsDir()) ? '' : '（不存在）'}\n`)
  process.stdout.write(`运行时目录          ${engineDir()}${fs.existsSync(engineDir()) ? '' : '（不存在）'}\n`)

  const binary = engine.binaryPath()
  process.stdout.write(`llama-server        ${binary ?? '缺失 ✗'}\n`)
  if (binary === null) failures += 1

  const defaultSpec = pickModel(flags)
  const modelPath = absoluteFileOf(defaultSpec)
  const present = fs.existsSync(modelPath)
  process.stdout.write(`默认权重            ${modelPath} ${present ? '已就绪 ✓' : '缺失 ✗'}\n`)
  if (!present) failures += 1

  const gpu = await gpuMemory()
  process.stdout.write(
    `GPU                 ${gpu.name ?? '不可用 ✗'}${gpu.totalMiB === null ? '' : ` · 空闲 ${formatMiB(gpu.freeMiB)} / ${formatMiB(gpu.totalMiB)}`}\n`
  )
  if (gpu.totalMiB === null) failures += 1

  const port = config.port
  const inUse = await portBusy(port)
  process.stdout.write(`端口 ${port}          ${inUse ? '已被占用（若为本插件引擎则正常）' : '空闲'}\n`)

  await engine.dispose()
  process.stdout.write(failures === 0 ? '\n全部就绪，可在 DSH 中选择 minicpm-local 模型。\n' : `\n${failures} 项待处理。\n`)
  return failures === 0 ? 0 : 1
}

/** Whether something answers on a local port. */
async function portBusy(port: number): Promise<boolean> {
  try {
    const response = await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(1200) })
    return response.ok
  } catch {
    return false
  }
}

/** Read `--port` when present. */
function flagPort(flags: Flags): number | undefined {
  const raw = flags.options.get('port')
  if (raw === undefined) return undefined
  const parsed = Number(raw)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined
}

/** Format MiB as GB. */
function formatMiB(mib: number | null): string {
  if (mib === null || !Number.isFinite(mib)) return '—'
  return `${(mib / 1024).toFixed(1)} GB`
}

/** Entry point. */
async function main(): Promise<number> {
  const flags = parseFlags(process.argv.slice(2))
  switch (flags.command) {
    case 'status':
      return commandStatus(flags)
    case 'start':
      return commandStart(flags)
    case 'stop':
      return commandStop()
    case 'models':
      return commandModels()
    case 'fetch-model':
      return commandFetchModel(flags)
    case 'fetch-engine':
      return commandFetchEngine(flags)
    case 'chat':
      return commandChat(flags)
    case 'doctor':
      return commandDoctor(flags)
    case '':
    case 'help':
    case '--help':
    case '-h':
      usage()
      return 0
    default:
      process.stderr.write(`未知命令：${flags.command}\n\n`)
      usage()
      return 2
  }
}

main()
  .then(code => {
    process.exitCode = code
  })
  .catch(error => {
    process.stderr.write(`dsh-minicpm: ${error instanceof Error ? error.message : String(error)}\n`)
    process.exitCode = 1
  })
