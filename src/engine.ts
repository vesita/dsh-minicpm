/**
 * Lifecycle owner for the local `llama-server` process.
 *
 * The engine is a *managed resource*, not a service the user is expected to
 * babysit. This manager spawns it on first use, reloads it when a different
 * weight file is requested, releases the GPU when nothing has called for a
 * while, and guarantees the child dies with the plugin's fiber. Every decision
 * it makes is readable afterwards through {@link EngineManager.status}, which
 * is what the settings card renders.
 *
 * Two modes:
 * - `managed`: this plugin owns the child process (the default);
 * - `external`: a server the user started is merely pointed at, never touched.
 *
 * @module dsh-minicpm/engine
 */
import { spawn } from 'node:child_process'
import type { ChildProcess } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { execFile } from 'node:child_process'
import { engineDir } from './paths.js'

/** Knobs the manager reads; every one is re-resolved per operation. */
export interface EngineConfig {
  /** `managed` spawns the child; `external` only talks to `baseURL`. */
  mode: 'managed' | 'external'
  /** Endpoint used in `external` mode, and the target in `managed` mode. */
  baseURL?: string
  /** Explicit `llama-server` path; discovered automatically when unset. */
  binary?: string
  host: string
  port: number
  /** `--ctx-size` the server is started with. */
  contextSize: number
  /** `--n-gpu-layers`; a large number means "all of them". */
  gpuLayers: number
  /** CPU threads for generation; unset lets the engine choose. */
  threads?: number
  /**
   * Seconds of inactivity after which the GPU is released by stopping the
   * server. `0` keeps it resident until DSH exits.
   */
  idleUnloadSeconds: number
  /** Extra CLI arguments appended verbatim, for power users. */
  extraArgs: string[]
  /** Keep the model's reasoning trace across turns (`--reasoning-preserve`). */
  preserveReasoning: boolean
}

/** Live, observable state of the engine. */
export interface EngineStatus {
  /** Whether a healthy endpoint is currently answering. */
  running: boolean
  /** Whether a start attempt is in flight. */
  starting: boolean
  /** `managed` or `external`, as configured. */
  mode: 'managed' | 'external'
  /** Endpoint the adapter would call. */
  baseURL: string
  /** OS process id of the managed child, when one is alive. */
  pid: number | null
  /** Epoch ms the current process started at. */
  startedAt: number | null
  /** Absolute path of the weight file the running server loaded. */
  loadedModel: string | null
  /** Absolute path of the runtime binary that would be used. */
  binaryPath: string
  /** Whether that binary exists and is executable. */
  binaryPresent: boolean
  /** Most recent failure, cleared on a successful start. */
  lastError: string | null
  /** Requests currently being streamed, which block idle unload. */
  inFlight: number
  /** Seconds until idle unload fires; null when nothing is scheduled. */
  idleSecondsLeft: number | null
  /** Tail of the child's merged stdout/stderr, newest last. */
  log: string[]
}

/** Default configuration, used before any settings section is installed. */
export const DEFAULT_ENGINE: EngineConfig = {
  mode: 'managed',
  host: '127.0.0.1',
  port: 8081,
  contextSize: 32768,
  gpuLayers: 999,
  idleUnloadSeconds: 900,
  extraArgs: [],
  preserveReasoning: true
}

/** Anything the manager reports a problem through. */
export interface EngineLogger {
  info?(message: string): void
  warn?(message: string): void
}

/** Raised for every engine failure the caller should see verbatim. */
export class EngineError extends Error {
  /** Stable machine code, mirroring the codes DSH's own adapters use. */
  readonly code: string

  constructor(message: string, code: string) {
    super(message)
    this.name = 'EngineError'
    this.code = code
  }
}

/** Cap on the retained log tail, in lines. */
const LOG_LIMIT = 200

/** How long a fresh server may take to become healthy. */
const READY_TIMEOUT_MS = 180_000

/** Grace period between SIGTERM and SIGKILL. */
const KILL_GRACE_MS = 5_000

/**
 * Owns one optional child process plus the health state that describes it.
 *
 * Instances are single-shot per plugin fiber: {@link dispose} stops the child
 * and permanently refuses further starts.
 */
export class EngineManager {
  /** Reads the live configuration on every operation. */
  readonly #config: () => EngineConfig
  readonly #logger: EngineLogger
  /** The managed child, when one is alive. */
  #child: ChildProcess | null = null
  /** Absolute path of the weight file the child loaded. */
  #loadedModel: string | null = null
  /** The in-flight start attempt, so concurrent callers share one spawn. */
  #starting: Promise<void> | null = null
  /** Pending idle-unload timer. */
  #idleTimer: NodeJS.Timeout | null = null
  /** Epoch ms the idle timer will fire, for the UI. */
  #idleDeadline: number | null = null
  /** Number of streams currently using the engine. */
  #inFlight = 0
  /** Most recent failure text. */
  #lastError: string | null = null
  /** Epoch ms the current child spawned at. */
  #startedAt: number | null = null
  /** Bounded tail of the child's output. */
  #log: string[] = []
  /** Set once the fiber is torn down; no further spawn is allowed. */
  #disposed = false
  /** Cached probe result, so `status()` stays synchronous. */
  #healthy = false
  /** Why the most recent `/health` probe failed, for diagnostics. */
  #lastProbeError: string | null = null
  /** Health probes attempted during the current start. */
  #probeCount = 0

  constructor(config: () => EngineConfig, logger: EngineLogger = {}) {
    this.#config = config
    this.#logger = logger
  }

  /** The endpoint calls should be sent to. */
  baseURL(): string {
    const config = this.#config()
    if (config.mode === 'external' && config.baseURL) return config.baseURL.replace(/\/+$/, '')
    return `http://${config.host}:${config.port}`
  }

  /**
   * Resolve the `llama-server` binary this manager would run.
   *
   * Discovery order is explicit setting, then the plugin's own engine
   * directory the fetch command fills, then whatever the user put on `PATH` —
   * so a system-wide llama.cpp installation is used without configuration, and
   * the self-managed copy wins when both exist.
   *
   * @returns the chosen absolute path, or `null` when nothing is executable.
   */
  binaryPath(): string | null {
    const explicit = this.#config().binary
    if (typeof explicit === 'string' && explicit.length > 0) {
      return executable(explicit) ? explicit : null
    }
    const managed = path.join(engineDir(), process.platform === 'win32' ? 'llama-server.exe' : 'llama-server')
    if (executable(managed)) return managed
    for (const directory of (process.env.PATH || '').split(path.delimiter)) {
      if (!directory) continue
      const candidate = path.join(directory, process.platform === 'win32' ? 'llama-server.exe' : 'llama-server')
      if (executable(candidate)) return candidate
    }
    return null
  }

  /**
   * Snapshot the engine's state for the settings card and the CLI.
   *
   * The health flag is the last observed value rather than a fresh probe: this
   * method is synchronous by design so the UI never waits on a socket.
   *
   * @returns a detached, JSON-safe status object.
   */
  status(): EngineStatus {
    const config = this.#config()
    const binary = this.binaryPath()
    return {
      running: this.#healthy,
      starting: this.#starting !== null,
      mode: config.mode,
      baseURL: this.baseURL(),
      pid: this.#child?.pid ?? null,
      startedAt: this.#startedAt,
      loadedModel: this.#loadedModel,
      binaryPath: binary ?? path.join(engineDir(), 'llama-server'),
      binaryPresent: binary !== null,
      lastError: this.#lastError,
      inFlight: this.#inFlight,
      idleSecondsLeft:
        this.#idleDeadline === null ? null : Math.max(0, Math.round((this.#idleDeadline - Date.now()) / 1000)),
      log: [...this.#log]
    }
  }

  /**
   * Make sure a server is answering for `modelPath`, starting or reloading it
   * when necessary.
   *
   * Concurrent callers are the normal case, not an edge case: one agent turn
   * routinely issues a title request and a main request together. A caller that
   * arrives while a start is in flight must *wait for it*, never probe it — a
   * server that is still loading answers `503 loading model`, and treating that
   * as "dead" would kill the very process the first caller is waiting for.
   *
   * @param modelPath - absolute path of the weight file the caller needs.
   * @returns the base URL to send the request to.
   * @throws {EngineError} when the runtime or the weight file is missing, or the
   *   server fails to become healthy.
   */
  async ensure(modelPath: string): Promise<string> {
    const config = this.#config()
    if (this.#disposed) throw new EngineError('MiniCPM 引擎已随插件卸载而关闭', 'ENGINE_DISPOSED')

    if (config.mode === 'external') {
      if (!(await this.#probe())) {
        throw new EngineError(
          `外部推理端点 ${this.baseURL()} 无响应。请确认已启动 llama-server，或把 llm-minicpm.engine.mode 改回 managed。`,
          'ENGINE_UNREACHABLE'
        )
      }
      return this.baseURL()
    }

    // Join an in-flight start rather than racing it. The owner of that start
    // reports its own failure, so a rejection here is not re-thrown: this
    // caller re-evaluates the outcome below and starts its own attempt if what
    // settled does not serve the weight it asked for.
    if (this.#starting !== null) {
      await this.#starting.catch(() => {})
      if (this.#disposed) throw new EngineError('MiniCPM 引擎已随插件卸载而关闭', 'ENGINE_DISPOSED')
    }

    if (this.#serving(modelPath)) {
      this.#healthy = true
      return this.baseURL()
    }

    await this.#start(modelPath)
    return this.baseURL()
  }

  /**
   * Whether a live child is already serving exactly this weight file.
   *
   * Health is deliberately *not* re-probed here: the only reason to distrust a
   * running child is a request that actually failed, and that failure reaches
   * the caller as a transport error it can act on. Probing instead would make
   * a loading server look dead.
   *
   * @param modelPath - absolute path of the weight file.
   */
  #serving(modelPath: string): boolean {
    const child = this.#child
    return (
      child !== null &&
      child.exitCode === null &&
      child.signalCode === null &&
      this.#loadedModel === modelPath
    )
  }

  /**
   * Restart the managed server around `modelPath`, or start it if it is not
   * running. Concurrent callers share one attempt.
   *
   * @param modelPath - absolute path of the weight file to load.
   */
  async #start(modelPath: string): Promise<void> {
    if (this.#starting !== null) return this.#starting

    const attempt = (async () => {
      if (this.#disposed) throw new EngineError('MiniCPM 引擎已随插件卸载而关闭', 'ENGINE_DISPOSED')

      const config = this.#config()
      const binary = this.binaryPath()
      if (binary === null) {
        throw new EngineError(
          `未找到 llama-server 运行时（已查找 ${engineDir()} 与 PATH）。请在设置页点击「下载推理引擎」，或运行 dsh-minicpm fetch-engine。`,
          'ENGINE_MISSING'
        )
      }
      if (!fs.existsSync(modelPath)) {
        throw new EngineError(
          `模型权重不存在：${modelPath}。请在设置页点击「下载模型」，或把 llm-minicpm.engine.modelPath 指向已有的 .gguf 文件。`,
          'MODEL_MISSING'
        )
      }

      // A reload is a stop followed by a start; doing it here keeps the two
      // halves ordered even when a stream is still draining.
      await this.#stopChild()

      const args = buildArgs(config, modelPath)
      this.#log = []
      this.#lastError = null
      this.#appendLog(`$ ${binary} ${args.join(' ')}`)

      const child = spawn(binary, args, {
        cwd: path.dirname(binary),
        stdio: ['ignore', 'pipe', 'pipe'],
        // Not detached: the child must die with DSH, and the plugin's own
        // teardown is what sends the signal.
        detached: false
      })
      this.#child = child
      this.#loadedModel = modelPath
      this.#startedAt = Date.now()

      const forward = (chunk: Buffer) => this.#appendLog(chunk.toString('utf8'))
      child.stdout?.on('data', forward)
      child.stderr?.on('data', forward)
      child.on('exit', (code, signal) => {
        this.#appendLog(`[exit] code=${code} signal=${signal}`)
        if (this.#child === child) {
          this.#child = null
          this.#loadedModel = null
          this.#startedAt = null
          this.#healthy = false
        }
      })
      child.on('error', error => {
        this.#lastError = error.message
        this.#appendLog(`[error] ${error.message}`)
      })

      await this.#awaitReady(child)
      this.#healthy = true
      this.#logger.info?.(`dsh-minicpm: 引擎已就绪 ${this.baseURL()} (pid ${child.pid})`)
      this.#scheduleIdle()
    })()

    this.#starting = attempt
    try {
      await attempt
    } catch (error) {
      this.#healthy = false
      this.#lastError = error instanceof Error ? error.message : String(error)
      // A failed start must not leave a half-initialised child behind holding
      // VRAM that nothing will ever release.
      await this.#stopChild()
      throw error
    } finally {
      this.#starting = null
    }
  }

  /**
   * Poll `/health` until the server reports readiness, the child exits, or the
   * timeout expires.
   *
   * The wait races the health poll against the child's own termination. That
   * race is the whole point: `exitCode` stays `null` when a process dies from a
   * signal, so a loop that only reads `exitCode` would sit out the entire
   * timeout after a child was already gone, and report "never became ready"
   * instead of the signal that killed it.
   *
   * @param child - the process that was just spawned.
   */
  async #awaitReady(child: ChildProcess): Promise<void> {
    if (child.exitCode !== null || child.signalCode !== null) {
      throw new EngineError(`llama-server 启动后立即退出（${describeExit(child)}）：${this.#tail(6)}`, 'ENGINE_FAILED')
    }

    let died: Error | null = null
    const onExit = (code: number | null, signal: NodeJS.Signals | null) => {
      died = new EngineError(
        `llama-server 在就绪前退出（code=${code} signal=${signal}，已探测 ${this.#probeCount} 次${this.#lastProbeError === null ? '' : `，最后一次探测：${this.#lastProbeError}`}）：${this.#tail(8)}`,
        'ENGINE_FAILED'
      )
    }
    const onError = (error: Error) => {
      died = new EngineError(`llama-server 无法启动：${error.message}`, 'ENGINE_FAILED')
    }
    child.once('exit', onExit)
    child.once('error', onError)

    this.#probeCount = 0
    this.#lastProbeError = null
    const deadline = Date.now() + READY_TIMEOUT_MS
    try {
      while (Date.now() < deadline) {
        if (this.#disposed) throw new EngineError('MiniCPM 引擎已随插件卸载而关闭', 'ENGINE_DISPOSED')
        // The listener runs on the same event loop, so a check here is enough
        // to notice a death that happened while the probe was in flight.
        if (died !== null) throw died
        if (child.exitCode !== null || child.signalCode !== null) {
          throw new EngineError(`llama-server 启动后立即退出（${describeExit(child)}）：${this.#tail(6)}`, 'ENGINE_FAILED')
        }
        if (await this.#probe()) return
        if (died !== null) throw died
        await delay(250)
      }
    } finally {
      child.off('exit', onExit)
      child.off('error', onError)
    }
    throw new EngineError(
      `llama-server 在 ${READY_TIMEOUT_MS / 1000}s 内未就绪（已探测 ${this.#probeCount} 次${this.#lastProbeError === null ? '' : `，最后一次探测：${this.#lastProbeError}`}）：${this.#tail(8)}`,
      'ENGINE_TIMEOUT'
    )
  }

  /**
   * Ask the endpoint whether it is serving.
   *
   * @returns true when `/health` answers 200 with a readable body.
   */
  async #probe(): Promise<boolean> {
    this.#probeCount += 1
    try {
      const response = await fetch(`${this.baseURL()}/health`, {
        signal: AbortSignal.timeout(2000)
      })
      if (!response.ok) {
        this.#lastProbeError = `HTTP ${response.status}`
        return false
      }
      const payload: any = await response.json().catch(() => ({}))
      // llama.cpp reports "ok" once loaded and "loading model" beforehand.
      if (payload?.status !== undefined && payload.status !== 'ok') {
        this.#lastProbeError = `status=${payload.status}`
        return false
      }
      this.#lastProbeError = null
      return true
    } catch (error) {
      this.#lastProbeError = error instanceof Error ? error.message : String(error)
      return false
    }
  }

  /**
   * Re-check the endpoint and cache the answer. Used by the status route so the
   * card reflects a server that was killed behind the plugin's back.
   *
   * @returns the freshly probed health.
   */
  async refreshHealth(): Promise<boolean> {
    if (this.#child === null && this.#config().mode === 'managed') {
      this.#healthy = false
      return false
    }
    this.#healthy = await this.#probe()
    return this.#healthy
  }

  /**
   * Mark one stream as started. Idle unload is suspended while any stream runs,
   * so a long generation is never cut off by the release timer.
   */
  begin(): void {
    this.#inFlight += 1
    this.#clearIdle()
  }

  /** Mark one stream as finished and restart the idle countdown. */
  end(): void {
    this.#inFlight = Math.max(0, this.#inFlight - 1)
    if (this.#inFlight === 0) this.#scheduleIdle()
  }

  /**
   * Stop the managed server, releasing its VRAM.
   *
   * A no-op in `external` mode: a process this plugin did not start is not one
   * it may kill.
   */
  async stop(): Promise<void> {
    this.#clearIdle()
    if (this.#config().mode === 'external') return
    await this.#stopChild()
    this.#logger.info?.('dsh-minicpm: 引擎已停止，显存已释放')
  }

  /** Terminate the child and wait for it to actually exit. */
  async #stopChild(): Promise<void> {
    const child = this.#child
    this.#child = null
    this.#loadedModel = null
    this.#startedAt = null
    this.#healthy = false
    if (child === null) return
    // An `exit` event that already fired never fires again, so waiting on it
    // would stall until the SIGKILL timer for a process that is long gone.
    if (child.exitCode !== null || child.signalCode !== null) return
    if (child.pid === undefined) return

    await new Promise<void>(resolve => {
      const done = () => {
        clearTimeout(kill)
        resolve()
      }
      const kill = setTimeout(() => {
        try {
          child.kill('SIGKILL')
        } catch {
          /* already gone */
        }
        resolve()
      }, KILL_GRACE_MS)
      child.once('exit', done)
      try {
        child.kill('SIGTERM')
      } catch {
        done()
      }
    })
  }

  /** Arm the idle-unload timer, when one is configured. */
  #scheduleIdle(): void {
    this.#clearIdle()
    const seconds = Number(this.#config().idleUnloadSeconds)
    if (!Number.isFinite(seconds) || seconds <= 0) return
    if (this.#child === null) return
    this.#idleDeadline = Date.now() + seconds * 1000
    this.#idleTimer = setTimeout(() => {
      this.#idleTimer = null
      this.#idleDeadline = null
      if (this.#inFlight > 0) {
        this.#scheduleIdle()
        return
      }
      void this.stop().catch(error => this.#logger.warn?.(`dsh-minicpm: 空闲卸载失败：${error.message}`))
    }, seconds * 1000)
    // The timer must never keep the DSH process alive on its own.
    this.#idleTimer.unref?.()
  }

  /** Cancel a pending idle unload. */
  #clearIdle(): void {
    if (this.#idleTimer !== null) clearTimeout(this.#idleTimer)
    this.#idleTimer = null
    this.#idleDeadline = null
  }

  /** Append child output to the bounded tail. */
  #appendLog(text: string): void {
    for (const line of text.split('\n')) {
      const trimmed = line.trimEnd()
      if (!trimmed) continue
      this.#log.push(trimmed)
    }
    if (this.#log.length > LOG_LIMIT) this.#log.splice(0, this.#log.length - LOG_LIMIT)
  }

  /** Last few log lines, for a one-line failure message. */
  #tail(count: number): string {
    return this.#log.slice(-count).join(' | ')
  }

  /**
   * Stop the child for good. Called from the plugin's teardown so no orphan
   * keeps the GPU while DSH thinks the plugin is gone.
   */
  async dispose(): Promise<void> {
    this.#disposed = true
    this.#clearIdle()
    await this.#stopChild()
  }
}

/**
 * Translate one configuration plus weight file into `llama-server` arguments.
 *
 * @param config - resolved engine configuration.
 * @param modelPath - absolute path of the weight file.
 * @returns the argument vector, without the binary.
 */
export function buildArgs(config: EngineConfig, modelPath: string): string[] {
  const args = [
    '-m',
    modelPath,
    '--host',
    config.host,
    '--port',
    String(config.port),
    '-c',
    String(config.contextSize),
    '-ngl',
    String(config.gpuLayers),
    // The Jinja chat template is what gives this model its tool-call and
    // reasoning vocabulary; without it tool use degrades to plain text.
    '--jinja',
    '--no-ui'
  ]
  if (typeof config.threads === 'number' && config.threads > 0) args.push('-t', String(config.threads))
  if (config.preserveReasoning === false) args.push('--no-reasoning-preserve')
  if (Array.isArray(config.extraArgs)) args.push(...config.extraArgs)
  return args
}

/** Whether a path is an existing executable file. */
function executable(candidate: string): boolean {
  try {
    fs.accessSync(candidate, fs.constants.X_OK)
    return fs.statSync(candidate).isFile()
  } catch {
    return false
  }
}

/** Describe how a child process ended, covering the silent signal case. */
function describeExit(child: ChildProcess): string {
  if (child.signalCode !== null && child.signalCode !== undefined) return `signal ${child.signalCode}`
  if (child.exitCode !== null) return `code ${child.exitCode}`
  return 'code unknown'
}

/** Promise-based delay. */
function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

/** One GPU's memory picture, as far as the plugin can observe it. */
export interface GpuMemory {
  /** MiB currently allocated, or null when unknown. */
  usedMiB: number | null
  /** MiB on the card, or null when unknown. */
  totalMiB: number | null
  /** Free MiB, or null when unknown. */
  freeMiB: number | null
  /** The adapter name reported by the driver. */
  name: string | null
}

/**
 * Read GPU memory from `nvidia-smi`.
 *
 * The engine reports nothing about its own VRAM, and the settings card's whole
 * point is showing what the local route costs, so this shells out to the one
 * tool that knows. Every failure degrades to nulls rather than an error:
 * a machine without NVIDIA tooling still gets a working plugin.
 *
 * @returns the first GPU's memory picture.
 */
export async function gpuMemory(): Promise<GpuMemory> {
  return new Promise(resolve => {
    execFile(
      'nvidia-smi',
      ['--query-gpu=name,memory.used,memory.total,memory.free', '--format=csv,noheader,nounits'],
      { timeout: 3000 },
      (error, stdout) => {
        if (error) {
          resolve({ usedMiB: null, totalMiB: null, freeMiB: null, name: null })
          return
        }
        const first = String(stdout).trim().split('\n')[0] || ''
        const [name, used, total, free] = first.split(',').map(part => part.trim())
        resolve({
          usedMiB: numberOrNull(used),
          totalMiB: numberOrNull(total),
          freeMiB: numberOrNull(free),
          name: name || null
        })
      }
    )
  })
}

/** Parse a non-negative finite number, or null. */
function numberOrNull(value: unknown): number | null {
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : null
}
