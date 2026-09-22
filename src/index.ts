/**
 * DSH host plugin for the local MiniCPM route.
 *
 * Ownership boundaries:
 * - the plugin owns the `llm-minicpm` settings namespace and the
 *   `minicpm-local` provider route, registered natively through
 *   `ctx.llm.registerAdapter` and never through `llm-pi-ai`;
 * - the `llama-server` child process is owned by {@link EngineManager}, which
 *   starts it on first use, reloads it when a different weight is requested, and
 *   releases the GPU after an idle period;
 * - long acquisitions (weights, runtime) are owned by {@link JobRegistry} and
 *   exposed to the browser half through loopback routes;
 * - the browser half renders the management card and drives those routes, so no
 *   settings write is ever needed for the common path.
 *
 * @module dsh-minicpm
 */
import z from '@deepseek-ai/schemastery'
import { RetryPolicySchema } from '@deepseek-ai/dsh-llm'
import type { IncomingMessage, ServerResponse } from 'node:http'
import fs from 'node:fs'
import path from 'node:path'
import { MiniCPMAdapter, PROVIDER, PROVIDER_NAME } from './adapter.js'
import { EngineManager, gpuMemory } from './engine.js'
import type { EngineConfig, EngineStatus, GpuMemory } from './engine.js'
import { JobRegistry, fetchEngine, fetchModel } from './fetch.js'
import type { Job } from './fetch.js'
import { MODEL_CATALOG, absoluteFileOf, resolveModelSpec } from './models.js'
import type { ModelSpec } from './models.js'
import { engineDir, modelsDir } from './paths.js'

export const name = 'dsh-minicpm'
export const inject = ['llm']

/**
 * Settings namespace this plugin owns — the profile entry id its Config lives
 * under.
 *
 * The settings service resolves an entry by `entry.options.id === ns`, so this
 * must equal the `id:` the profile patch declares for this package. It is read
 * off the live fiber at runtime (`settingsNs`) and this constant is the fallback
 * for a host with no entry (the standalone CLI, a bare `apply()` in a test). The
 * two must stay in step: the bundled `cordis.patch.yml` declares
 * `id: minicpm`.
 */
const NS = 'minicpm'
/** Loopback route prefix for the browser half. */
const ROUTE_PREFIX = '/dsh-minicpm'

/** One catalogue entry as settings carry it. */
const modelSchema = z.object({
  id: z.string().required(),
  name: z.string().default(''),
  description: z.string().default(''),
  /** GGUF file name inside the models directory, or an absolute path. */
  file: z.string().required()
})

/** Engine knobs, all optional so the defaults below apply. */
const engineSchema = z
  .object({
    mode: z.union(['managed', 'external']).default('managed'),
    baseURL: z.string().default(''),
    binary: z.string().default(''),
    host: z.string().default('127.0.0.1'),
    port: z.number().step(1).min(1).max(65535).default(8081),
    contextSize: z.number().step(1).min(512).default(32768),
    gpuLayers: z.number().step(1).min(0).default(999),
    threads: z.number().step(1).min(0).default(0),
    idleUnloadSeconds: z.number().step(1).min(0).default(900),
    extraArgs: z.array(z.string()).default([]),
    preserveReasoning: z.boolean().default(true)
  })
  .default({
    mode: 'managed',
    baseURL: '',
    binary: '',
    host: '127.0.0.1',
    port: 8081,
    contextSize: 32768,
    gpuLayers: 999,
    threads: 0,
    idleUnloadSeconds: 900,
    extraArgs: [],
    preserveReasoning: true
  })

/** Acquisition knobs. */
const fetchSchema = z
  .object({
    /** Hugging Face base URL; a mirror keeps large pulls off the public CDN. */
    hfEndpoint: z.string().default('https://huggingface.co'),
    /** Pinned llama.cpp build tag, e.g. `b10951`; newest is discovered when empty. */
    build: z.string().default('')
  })
  .default({ hfEndpoint: 'https://huggingface.co', build: '' })

/**
 * The plugin's Config schema.
 *
 * DSH 0.1.7 moved plugin settings to the entry's own Config: the Plugins page
 * renders a form for exactly the schema nodes marked `.volatile()`, and saving
 * one writes the new value into the running fiber and emits
 * `loader/volatile-update` — no plugin reload.
 *
 * The engine and fetch knobs are exposed **flat** (`enginePort`, `engineHost`, …)
 * rather than only nested: a Plugins-page form addresses a field by a single key
 * (`value?.[field]`, `path: [field]`), so `engine.port` would look up a literal
 * top-level key of that name. The nested `engine` / `fetch` objects remain the
 * profile-patch spelling and the fallback — see `engineConfig`, which reconciles
 * the two.
 */
export const Config = z.object({
  models: z.array(modelSchema).default(MODEL_CATALOG.map(spec => ({ ...spec, name: spec.name, description: spec.description ?? '' }))),
  engine: engineSchema,
  fetch: fetchSchema,
  retryPolicy: RetryPolicySchema,
  // Flat, form-addressable spellings of the engine settings.
  engineMode: z.union(['managed', 'external']).default('managed').volatile(),
  engineBaseURL: z.string().default('').volatile(),
  engineBinary: z.string().default('').volatile(),
  engineHost: z.string().default('127.0.0.1').volatile(),
  enginePort: z.number().step(1).min(1).max(65535).default(8081).volatile(),
  engineContextSize: z.number().step(1).min(512).default(32768).volatile(),
  engineGpuLayers: z.number().step(1).min(0).default(999).volatile(),
  engineThreads: z.number().step(1).min(0).default(0).volatile(),
  engineIdleUnloadSeconds: z.number().step(1).min(0).default(900).volatile(),
  enginePreserveReasoning: z.boolean().default(true).volatile(),
  // Same reasoning for the fetch settings.
  fetchHfEndpoint: z.string().default('https://huggingface.co').volatile(),
  fetchBuild: z.string().default('').volatile()
})

/** The section shape after schema defaulting. */
interface MiniCPMSettings {
  models?: Array<{ id: string; name?: string; description?: string; file: string }>
  engine?: {
    mode?: string
    baseURL?: string
    binary?: string
    host?: string
    port?: number
    contextSize?: number
    gpuLayers?: number
    threads?: number
    idleUnloadSeconds?: number
    extraArgs?: string[]
    preserveReasoning?: boolean
  }
  fetch?: { hfEndpoint?: string; build?: string }
  /**
   * Flat, form-addressable spellings of the fields above.
   *
   * The Plugins page addresses a field by a single key, so what a user edits
   * there lands here rather than inside the nested objects; `current()`
   * reconciles the two, the flat one winning.
   */
  engineMode?: string
  engineBaseURL?: string
  engineBinary?: string
  engineHost?: string
  enginePort?: number
  engineContextSize?: number
  engineGpuLayers?: number
  engineThreads?: number
  engineIdleUnloadSeconds?: number
  enginePreserveReasoning?: boolean
  fetchHfEndpoint?: string
  fetchBuild?: string
  retryPolicy?: unknown
}

/** One weight file as the card's download list shows it. */
interface ModelRow {
  id: string
  name: string
  description: string
  file: string
  path: string
  present: boolean
  /** On-disk size in bytes, when the file exists. */
  sizeBytes: number | null
  /** Expected size in bytes, when the catalogue states one. */
  expectedBytes: number | null
}

/** Everything the browser half renders. */
interface StatusPayload {
  engine: EngineStatus
  gpu: GpuMemory
  models: ModelRow[]
  jobs: Job[]
  activeModel: string
  provider: string
  providerName: string
  engineDir: string
  modelsDir: string
  hfEndpoint: string
}

/**
 * Mount the plugin.
 *
 * @param ctx - Cordis context; optional seams are resolved through `ctx.get`
 *   so a deployment without a web server still gets a working model route.
 * @param config - the `llm-minicpm` section resolved at load time.
 */
export function apply(ctx: any, config: MiniCPMSettings = {}): void {
  let current: () => MiniCPMSettings = () => config

  /**
   * Read one schema field, resolving a DSH 0.1.7 volatile reference.
   *
   * A `.volatile()` field is a stable reference whose `get()` returns the live
   * value (`JSON.stringify` of it yields `{}`); the Loader rewrites that value in
   * place when the user saves the Plugins-page form. Non-volatile fields arrive
   * as plain values, so both shapes are recognised here.
   */
  const read = <K extends keyof MiniCPMSettings>(field: K): MiniCPMSettings[K] => {
    const raw: any = (config as any)?.[field]
    if (raw !== null && typeof raw === 'object' && typeof raw.get === 'function') {
      return raw.get() as MiniCPMSettings[K]
    }
    return raw as MiniCPMSettings[K]
  }

  /**
   * The plugin's live configuration, with volatile fields resolved.
   *
   * The flat `engine*` / `fetch*` fields win over the nested objects, because two
   * editors write two spellings: the profile patch and existing installs use the
   * nested `engine` / `fetch`; the Plugins-page form can only address a single
   * top-level key, so it writes the flat ones. Both default identically, so
   * "which is set" only matters once a user has touched one — and the one they
   * touched must win.
   */
  current = (): MiniCPMSettings => {
    const nestedEngine = read('engine') ?? {}
    const nestedFetch = read('fetch') ?? {}
    return {
      models: read('models'),
      retryPolicy: read('retryPolicy'),
      engine: {
        mode: read('engineMode') ?? nestedEngine.mode,
        baseURL: read('engineBaseURL') ?? nestedEngine.baseURL,
        binary: read('engineBinary') ?? nestedEngine.binary,
        host: read('engineHost') ?? nestedEngine.host,
        port: read('enginePort') ?? nestedEngine.port,
        contextSize: read('engineContextSize') ?? nestedEngine.contextSize,
        gpuLayers: read('engineGpuLayers') ?? nestedEngine.gpuLayers,
        threads: read('engineThreads') ?? nestedEngine.threads,
        idleUnloadSeconds: read('engineIdleUnloadSeconds') ?? nestedEngine.idleUnloadSeconds,
        extraArgs: nestedEngine.extraArgs,
        preserveReasoning: read('enginePreserveReasoning') ?? nestedEngine.preserveReasoning
      },
      fetch: {
        hfEndpoint: read('fetchHfEndpoint') ?? nestedFetch.hfEndpoint,
        build: read('fetchBuild') ?? nestedFetch.build
      }
    }
  }

  /** Normalise the settings section into the engine's own shape. */
  const engineConfig = (): EngineConfig => {
    const settings = current().engine ?? {}
    return {
      mode: settings.mode === 'external' ? 'external' : 'managed',
      baseURL: settings.baseURL || undefined,
      binary: settings.binary || undefined,
      host: settings.host || '127.0.0.1',
      port: Number(settings.port) > 0 ? Number(settings.port) : 8081,
      contextSize: Number(settings.contextSize) > 0 ? Number(settings.contextSize) : 32768,
      gpuLayers: Number.isFinite(Number(settings.gpuLayers)) ? Number(settings.gpuLayers) : 999,
      threads: Number(settings.threads) > 0 ? Number(settings.threads) : undefined,
      idleUnloadSeconds: Number.isFinite(Number(settings.idleUnloadSeconds)) ? Number(settings.idleUnloadSeconds) : 900,
      extraArgs: Array.isArray(settings.extraArgs) ? settings.extraArgs : [],
      preserveReasoning: settings.preserveReasoning !== false
    }
  }

  /** The active catalogue, defaulted when settings supplied none. */
  const catalog = (): ModelSpec[] => {
    const models = current().models
    if (!Array.isArray(models) || models.length === 0) return MODEL_CATALOG
    return models.map(entry => ({
      id: entry.id,
      name: entry.name || entry.id,
      ...(entry.description ? { description: entry.description } : {}),
      file: entry.file
    }))
  }

  const engine = new EngineManager(engineConfig, {
    info: (message: string) => ctx.logger?.info?.(message),
    warn: (message: string) => ctx.logger?.warn?.(message)
  })
  const jobs = new JobRegistry()

  /** File names present in the models directory, for catalogue availability notes. */
  const availableFiles = (): Set<string> => {
    try {
      return new Set(fs.readdirSync(modelsDir()).filter(entry => entry.toLowerCase().endsWith('.gguf')))
    } catch {
      return new Set()
    }
  }

  // ---------------------------------------------------------------------------
  // Provider route (native adapter)
  // ---------------------------------------------------------------------------
  const adapter = new MiniCPMAdapter({
    resolveModels: catalog,
    resolveEngine: engineConfig,
    engine,
    resolveAvailable: availableFiles
  })

  const adapterHandle = ctx.llm.registerAdapter([PROVIDER], adapter)
  /**
   * The profile entry id this plugin's Config lives under.
   *
   * The settings service resolves an entry by `entry.options.id === ns`, and that
   * id is whatever the profile patch declared (`cordis.patch.yml` →
   * `id: minicpm`) — not a name this package can fix in advance. So it is read
   * off the live fiber, exactly as the official `llm-deepseek` adapter does
   * (`settingsNs: ctx.fiber.entry?.options.id ?? NS`), with the constant kept as
   * the fallback for a host that has no entry (the standalone CLI, a bare
   * `apply()` in a test).
   *
   * Everything that addresses this plugin's own settings goes through this: a
   * namespace the profile did not declare makes the service refuse every write.
   */
  const settingsNs = (): string => ctx.fiber?.entry?.options?.id ?? NS
  const directoryHandle = ctx.llm.registerConfigurableProviders([
    {
      provider: PROVIDER,
      displayName: PROVIDER_NAME,
      settingsNs: settingsNs(),
      // An empty path means the whole section is this provider's profile, so the
      // row is configured from the start — a local engine needs no account
      // marker to be meaningful, and the card exists to set it up.
      settingsPath: [],
      declared: false
    }
  ])

  // ---------------------------------------------------------------------------
  // Status assembly
  // ---------------------------------------------------------------------------
  /** Build the card's whole view model. */
  const statusPayload = async (): Promise<StatusPayload> => {
    const present = availableFiles()
    const list = catalog()
    const models: ModelRow[] = list.map(spec => {
      const filePath = absoluteFileOf(spec)
      const name = path.basename(spec.file)
      const rows = MODEL_CATALOG.find(spec2 =>
        path.basename(spec2.file) === name || spec2.id === spec.id
      )
      let sizeBytes: number | null = null
      try {
        sizeBytes = fs.statSync(filePath).size
      } catch {
        sizeBytes = null
      }
      return {
        id: spec.id,
        name: spec.name || spec.id,
        description: spec.description ?? '',
        file: name,
        path: filePath,
        present: present.has(name) || fs.existsSync(filePath),
        sizeBytes,
        expectedBytes: rows?.sizeBytes ?? null
      }
    })
    return {
      engine: engine.status(),
      gpu: await gpuMemory(),
      models,
      jobs: jobs.list(),
      activeModel: list[0]?.id ?? 'minicpm5-2b-q4',
      provider: PROVIDER,
      providerName: PROVIDER_NAME,
      engineDir: engineDir(),
      modelsDir: modelsDir(),
      hfEndpoint: current().fetch?.hfEndpoint || 'https://huggingface.co'
    }
  }

  // ---------------------------------------------------------------------------
  // Settings section
  // ---------------------------------------------------------------------------
  /**
   * Fingerprint of the engine-relevant settings as of the last notification.
   *
   * `onChange` is documented to fire **at attach and at detach** as well as on a
   * committed change, so treating every notification as "settings moved" would
   * stop an engine the very first request just started — the attach arrives
   * right after the plugin mounts, which is typically mid-first-call. Comparing
   * the resolved engine configuration is what separates a real change from
   * those lifecycle notifications.
   */
  let engineFingerprint: string | null = null

  // ---------------------------------------------------------------------------
  // Settings
  // ---------------------------------------------------------------------------
  // DSH 0.1.7 removed `ctx.settings.installSection` (the whole section API is
  // gone; `SettingsForms` now only describes the Config of each profile entry).
  // Plugin preferences are the entry's own volatile Config fields, so there is
  // no section to install — this plugin only has to notice when the Loader
  // commits an edited field and re-run the effects that depend on it.
  //
  // `loader/volatile-update` is that signal: the Loader rewrites the volatile
  // reference in place and emits the event, so `current()` already sees the new
  // value by the time this fires. Nothing reloads.
  engineFingerprint = JSON.stringify(engineConfig())
  ctx.on('loader/volatile-update', () => {
    const next = JSON.stringify(engineConfig())
    const previous = engineFingerprint
    engineFingerprint = next
    // The first notification is the attach; nothing was running under the
    // previous value, so there is nothing to invalidate.
    if (previous === null || previous === next) return
    // A changed port, context size, binary or mode makes the running child
    // stale; releasing it means the next call starts one that matches.
    void engine.stop().catch((error: Error) => ctx.logger?.warn?.(`dsh-minicpm: ${error.message}`))
  })

  // ---------------------------------------------------------------------------
  // Loopback routes for the browser half
  // ---------------------------------------------------------------------------
  ctx.inject(['webServer'], (webCtx: any) => {
    /**
     * Reject anything that did not come from this browser session.
     *
     * @returns true when the request was already refused and answered.
     */
    const guard = (req: IncomingMessage, res: ServerResponse): boolean => {
      const connection = ctx.get('connection')
      if (connection === undefined) return false
      const rejection = connection.requestRejection(req)
      if (rejection === undefined) return false
      res.statusCode = rejection
      res.end()
      return true
    }

    const register = (method: string, suffix: string, handler: (req: IncomingMessage, res: ServerResponse) => void | Promise<void>): void => {
      const route = `${ROUTE_PREFIX}${suffix}`
      webCtx.effect(
        () =>
          webCtx.webServer.register({
            kind: 'exact',
            path: route,
            handler: async (req: IncomingMessage, res: ServerResponse) => {
              if (guard(req, res)) return
              if (req.method !== method) return methodNotAllowed(res, method)
              try {
                await handler(req, res)
              } catch (error) {
                sendJson(res, 500, { error: error instanceof Error ? error.message : String(error) })
              }
            }
          }),
        `dsh-minicpm: ${method} ${route}`
      )
    }

    // The card polls this one route for everything it renders.
    register('GET', '/status', async (_req, res) => {
      // A server killed outside the plugin must not keep reporting healthy.
      await engine.refreshHealth()
      sendJson(res, 200, await statusPayload())
    })

    register('POST', '/engine/start', async (req, res) => {
      const body = await readJson(req)
      const list = catalog()
      const spec = resolveModelSpec(typeof body.model === 'string' && body.model ? body.model : list[0]?.id ?? '', list)
      await engine.ensure(absoluteFileOf(spec))
      sendJson(res, 200, await statusPayload())
    })

    register('POST', '/engine/stop', async (_req, res) => {
      await engine.stop()
      sendJson(res, 200, await statusPayload())
    })

    register('POST', '/download/model', async (req, res) => {
      const body = await readJson(req)
      if (jobs.busy('model')) {
        sendJson(res, 409, { error: '已有一个模型下载正在进行' })
        return
      }
      const list = catalog()
      const spec = resolveModelSpec(typeof body.id === 'string' && body.id ? body.id : list[0]?.id ?? '', list)
      const fetchOptions = {
        hfEndpoint: current().fetch?.hfEndpoint,
        engineBuild: current().fetch?.build || undefined
      }
      const handle = jobs.start('model', spec.name || spec.id, (report, signal) =>
        fetchModel(spec, fetchOptions, report, signal)
      )
      handle.completion.catch((error: Error) => ctx.logger?.warn?.(`dsh-minicpm: 模型下载失败：${error.message}`))
      sendJson(res, 200, { jobId: handle.id })
    })

    register('POST', '/download/engine', async (_req, res) => {
      if (jobs.busy('engine')) {
        sendJson(res, 409, { error: '已有一个引擎下载正在进行' })
        return
      }
      const handle = jobs.start('engine', 'llama.cpp 运行时', (report, signal) =>
        fetchEngine(
          { hfEndpoint: current().fetch?.hfEndpoint, engineBuild: current().fetch?.build || undefined },
          report,
          signal
        )
      )
      handle.completion.catch((error: Error) => ctx.logger?.warn?.(`dsh-minicpm: 引擎下载失败：${error.message}`))
      sendJson(res, 200, { jobId: handle.id })
    })

    register('POST', '/jobs/cancel', async (req, res) => {
      const body = await readJson(req)
      const id = typeof body.id === 'string' ? body.id : ''
      // Cancellation is cooperative: the job aborts, the partial file is
      // removed, and the next status poll shows it settled.
      if (!jobs.cancel(id)) {
        sendJson(res, 404, { error: '没有这个正在进行的任务' })
        return
      }
      sendJson(res, 200, { cancelled: true })
    })

    register('GET', '/models', async (_req, res) => {
      sendJson(res, 200, { models: (await statusPayload()).models })
    })
  })

  // ---------------------------------------------------------------------------
  // Teardown
  // ---------------------------------------------------------------------------
  ctx.effect(
    () => () => {
      try {
        adapterHandle?.()
      } catch {
        /* already released */
      }
      try {
        directoryHandle?.()
      } catch {
        /* already released */
      }
      jobs.dispose()
      // The child must not outlive the plugin: an orphan would keep several
      // gigabytes of VRAM while DSH believes the route is gone.
      void engine.dispose().catch((error: Error) => ctx.logger?.warn?.(`dsh-minicpm: ${error.message}`))
    },
    'dsh-minicpm: registration teardown'
  )
}

/**
 * Read a JSON body, tolerating an absent one.
 */
async function readJson(req: IncomingMessage): Promise<any> {
  const chunks: Buffer[] = []
  for await (const chunk of req) chunks.push(Buffer.from(chunk))
  if (chunks.length === 0) return {}
  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8'))
  } catch {
    return {}
  }
}

/** Send a JSON response. */
function sendJson(res: ServerResponse, statusCode: number, payload: unknown): void {
  if (res.headersSent) return
  res.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store'
  })
  res.end(JSON.stringify(payload))
}

/** Answer a method mismatch. */
function methodNotAllowed(res: ServerResponse, allowed: string): void {
  res.setHeader('Allow', allowed)
  sendJson(res, 405, { error: `仅支持 ${allowed}` })
}

export { MiniCPMAdapter, PROVIDER, PROVIDER_NAME } from './adapter.js'
export { EngineManager } from './engine.js'
export default { name, inject, apply, Config }
