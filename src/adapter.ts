/**
 * Native DSH `LlmAdapter` over a local `llama-server` endpoint.
 *
 * The adapter owns no settings and no process: it asks the engine manager for
 * an endpoint serving the weight file the requested model id names, then speaks
 * the OpenAI-compatible protocol to it. That split is what lets the same class
 * work against a plugin-managed child, a system `llama-server`, or a remote
 * OpenAI-compatible box.
 *
 * @module dsh-minicpm/adapter
 */
import { LlmAdapter, LlmError, attributionHeaders } from '@deepseek-ai/dsh-llm'
import type {
  GenerateOptions,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  ResolvedRetryPolicy,
  StreamChunk
} from '@deepseek-ai/dsh-llm'
import { MODEL_CATALOG, absoluteFileOf, modelInfoOf, resolveModelSpec, resolvedModelInfoOf } from './models.js'
import type { ModelSpec } from './models.js'
import { buildRequest, parseStream } from './wire.js'
import type { EngineConfig, EngineManager } from './engine.js'

/** Every fact the adapter reads, supplied by the mounting plugin. */
export interface AdapterOptions {
  /** Active catalogue, read fresh so settings edits apply to the next call. */
  resolveModels?: () => ModelSpec[]
  /** Live engine configuration, read fresh for the context window it implies. */
  resolveEngine?: () => EngineConfig
  /** Lifecycle owner that guarantees an endpoint for a weight file. */
  engine: EngineManager
  /** File names currently present in the models directory, for availability notes. */
  resolveAvailable?: () => Set<string>
}

/**
 * Provider route served by this adapter.
 *
 * The suffix states the locality, so a user selecting it in the model picker
 * cannot mistake it for a hosted route with the same weights.
 */
export const PROVIDER = 'minicpm-local'

/** Display name shown in provider selectors. */
export const PROVIDER_NAME = 'MiniCPM (本地)'

/** Per-request output cap applied when the caller states none. */
const DEFAULT_MAX_TOKENS = 8192

export class MiniCPMAdapter extends LlmAdapter {
  /** Resolvers installed by the mounting plugin. */
  readonly options: AdapterOptions

  /**
   * @param options - catalogue, engine configuration, and the lifecycle owner.
   */
  constructor(options: AdapterOptions) {
    super()
    this.options = options
  }

  /** The active catalogue, defaulted when settings supplied none. */
  #catalog(): ModelSpec[] {
    const models = this.options.resolveModels?.()
    return Array.isArray(models) && models.length > 0 ? models : MODEL_CATALOG
  }

  providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: PROVIDER_NAME }
  }

  providerRetryPolicy(): ResolvedRetryPolicy | undefined {
    // The policy is left entirely to the harness defaults. Returning a
    // half-written literal here is the documented trap: `dsh-llm-retry` reads
    // `policy.retryableCodes.includes(...)` on the first failure, so a policy
    // missing that field turns a real transport fault into a TypeError.
    return undefined
  }

  async listModels(provider: string): Promise<LlmModelInfo[]> {
    const available = this.options.resolveAvailable?.()
    return this.#catalog().map(spec => modelInfoOf(spec, provider, available))
  }

  async resolveModel(provider: string, modelId: string): Promise<LlmResolvedModelInfo> {
    const spec = resolveModelSpec(modelId, this.#catalog())
    // The advertised window is the engine's own `--ctx-size`, never the model's
    // architectural ceiling: promising 128k from a server started at 32k would
    // let the loop build requests the engine has to truncate.
    const engine = this.options.resolveEngine?.()
    const contextWindow = engine?.contextSize && engine.contextSize > 0 ? engine.contextSize : 32768
    return resolvedModelInfoOf(spec, provider, contextWindow, DEFAULT_MAX_TOKENS)
  }

  async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    const spec = resolveModelSpec(options.model, this.#catalog())
    const modelPath = absoluteFileOf(spec)

    const baseURL = await this.options.engine.ensure(modelPath)
    const body = buildRequest(options, spec.id)

    this.options.engine.begin()
    try {
      const response = await this.#post(baseURL, body, options.signal)
      yield* parseStream(response)
    } finally {
      // The idle clock only restarts once nothing is streaming, so a long
      // generation can never be cut short by the release timer.
      this.options.engine.end()
    }
  }

  /**
   * Open the completion stream, turning a non-2xx answer into a typed failure.
   *
   * @param baseURL - endpoint the engine resolved.
   * @param body - the assembled OpenAI-compatible request.
   * @param signal - caller cancellation.
   * @returns the open SSE response.
   */
  async #post(baseURL: string, body: unknown, signal?: AbortSignal): Promise<Response> {
    let response: Response
    try {
      response = await fetch(`${baseURL}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
          ...attributionHeaders()
        },
        body: JSON.stringify(body),
        signal
      })
    } catch (error) {
      if (signal?.aborted) throw new LlmError('MiniCPM 请求已被取消', 'ABORTED', { cause: error })
      throw new LlmError(
        `无法连接本地 MiniCPM 引擎 ${baseURL}：${error instanceof Error ? error.message : String(error)}`,
        'TRANSPORT',
        { cause: error }
      )
    }

    if (response.ok && response.body) return response

    const detail = await response.text().catch(() => '')
    throw new LlmError(
      `MiniCPM 引擎返回 ${response.status}：${detail.slice(0, 500) || response.statusText}`,
      httpCode(response.status),
      { status: response.status }
    )
  }
}

/**
 * Map an HTTP status onto DSH's failure taxonomy so retry policy can act on it.
 *
 * @param status - status observed at the engine boundary.
 * @returns a stable machine code.
 */
function httpCode(status: number): string {
  if (status === 401 || status === 403) return 'INVALID_CREDENTIAL'
  if (status === 404) return 'MODEL_NOT_FOUND'
  if (status === 429) return 'RATE_LIMIT'
  if (status === 400 || status === 422) return 'INVALID_REQUEST'
  if (status >= 500) return 'TRANSPORT'
  return 'UNKNOWN'
}
