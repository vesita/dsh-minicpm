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

/** Settings namespace this plugin owns. */
const NS = 'llm-minicpm'
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

/** The resolved `llm-minicpm` settings section. */
export const Config = z.object({
  models: z.array(modelSchema).default(MODEL_CATALOG.map(spec => ({ ...spec, name: spec.name, description: spec.description ?? '' }))),
  engine: engineSchema,
  fetch: fetchSchema,
  retryPolicy: RetryPolicySchema
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
  const directoryHandle = ctx.llm.registerConfigurableProviders([
    {
      provider: PROVIDER,
      displayName: PROVIDER_NAME,
      settingsNs: NS,
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

  ctx.inject(['settings'], (settingsCtx: any) => {
    settingsCtx.settings.installSection(ctx, NS, Config, config, {
      setSource: (source: () => MiniCPMSettings) => {
        current = source
      },
      onChange: () => {
        const next = JSON.stringify(engineConfig())
        const previous = engineFingerprint
        engineFingerprint = next
        // The first notification is the attach; nothing was running under the
        // previous value, so there is nothing to invalidate.
        if (previous === null || previous === next) return
        // A changed port, context size, binary or mode makes the running child
        // stale; releasing it means the next call starts one that matches.
        void engine.stop().catch((error: Error) => ctx.logger?.warn?.(`dsh-minicpm: ${error.message}`))
      }
    })
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
