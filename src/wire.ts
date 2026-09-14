/**
 * Translation between DSH's provider-neutral vocabulary and the
 * OpenAI-compatible wire that `llama-server` serves.
 *
 * The two halves are deliberately pure functions over plain data: request
 * construction never touches the network, and the stream parser only reads an
 * already-open `Response`. That keeps the mapping testable without a model and
 * leaves the adapter itself responsible only for process and lifecycle facts.
 *
 * @module dsh-minicpm/wire
 */
import { EMPTY_RESPONSE_CODE } from '@deepseek-ai/dsh-llm'
import type {
  ContentBlock,
  FinishReason,
  GenerateOptions,
  Message,
  StreamChunk,
  TokenUsage,
  ToolCallBlock,
  ToolCallId,
  ToolResultBlock
} from '@deepseek-ai/dsh-llm'

/**
 * Adopt a provider-issued call id into DSH's branded type.
 *
 * The id is opaque to this plugin — it is echoed back on the matching tool
 * result and never interpreted — so branding is the whole operation.
 */
function asCallId(value: string): ToolCallId {
  return value as unknown as ToolCallId
}

/** A content block as the text projection reads it. */
type TextBearingBlock = ContentBlock & { text?: string }

/** One OpenAI-compatible message. */
export interface WireMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string
  tool_calls?: WireToolCall[]
  tool_call_id?: string
}

/** One OpenAI-compatible tool invocation on an assistant message. */
export interface WireToolCall {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

/** One OpenAI-compatible tool declaration. */
export interface WireTool {
  type: 'function'
  function: { name: string; description: string; parameters: Record<string, unknown> }
}

/** The body sent to `/v1/chat/completions`. */
export interface WireRequest {
  model: string
  messages: WireMessage[]
  stream: true
  stream_options: { include_usage: true }
  tools?: WireTool[]
  temperature?: number
  max_tokens?: number
  stop?: string[]
}

/**
 * Translate one DSH request into an OpenAI-compatible chat completion body.
 *
 * Three projections differ from a naive field-for-field copy:
 *
 * - every `system`-role message is hoisted into a single leading system
 *   message, because that is the only system slot OpenAI-compatible servers
 *   uniformly honour;
 * - a user-role message carrying tool results becomes one `tool` message per
 *   result, which is the shape a tool-calling template expects;
 * - `reasoning` blocks are dropped, since a model's private trace is not
 *   replayed back to it.
 *
 * @param options - the assembled DSH request.
 * @param wireModel - model id to put on the wire.
 * @returns the JSON body for `/v1/chat/completions`.
 */
export function buildRequest(options: GenerateOptions, wireModel: string): WireRequest {
  const systemParts: string[] = []
  if (typeof options.system === 'string' && options.system.length > 0) systemParts.push(options.system)

  const messages: WireMessage[] = []

  for (const message of options.messages || []) {
    if (message.role === 'system') {
      const text = textOf(message.content)
      if (text) systemParts.push(text)
      continue
    }
    if (message.role === 'assistant') {
      messages.push(assistantMessage(message.content))
      continue
    }
    messages.push(...userMessages(message))
  }

  const body: WireRequest = {
    model: wireModel,
    messages: systemParts.length > 0 ? [{ role: 'system', content: systemParts.join('\n\n') }, ...messages] : messages,
    stream: true,
    stream_options: { include_usage: true }
  }

  if (Array.isArray(options.tools) && options.tools.length > 0) {
    body.tools = options.tools.map(tool => ({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description || '',
        parameters: normaliseParameters(tool.parameters)
      }
    }))
  }
  if (typeof options.temperature === 'number') body.temperature = options.temperature
  if (typeof options.maxTokens === 'number' && options.maxTokens > 0) body.max_tokens = options.maxTokens
  if (Array.isArray(options.stop) && options.stop.length > 0) body.stop = [...options.stop]

  // A conversation cannot be empty on the wire; a one-shot caller that supplied
  // neither history nor a system prompt still needs a user turn.
  if (body.messages.every(message => message.role === 'system')) {
    body.messages.push({ role: 'user', content: 'Hello' })
  }
  return body
}

/** Project one assistant message, keeping its tool calls in correlation order. */
function assistantMessage(content: Message['content'] | string): WireMessage {
  const blocks = Array.isArray(content) ? content : []
  const text = blocks
    .filter(block => block.type === 'text')
    .map(block => (block as TextBearingBlock).text || '')
    .join('')
  const calls = blocks
    .filter((block): block is ToolCallBlock => block.type === 'tool-call')
    .map(block => ({
      id: block.id,
      type: 'function' as const,
      function: { name: block.name, arguments: block.arguments || '{}' }
    }))

  if (calls.length === 0) return { role: 'assistant', content: text }
  // OpenAI requires content to be null-or-string; an empty string is accepted
  // by llama-server and avoids a null-vs-missing ambiguity in older templates.
  return { role: 'assistant', content: text, tool_calls: calls }
}

/**
 * Project one user-role DSH message.
 *
 * Tool results are split out into their own `tool` messages; any surrounding
 * text stays in a `user` message so a mixed turn keeps its order.
 */
function userMessages(message: Message): WireMessage[] {
  const blocks = Array.isArray(message.content) ? message.content : []
  const results = blocks.filter((block): block is ToolResultBlock => block.type === 'tool-result')
  const out: WireMessage[] = []

  if (results.length === 0) {
    const text = textOf(blocks)
    if (text) out.push({ role: 'user', content: text })
    return out
  }

  // Text that arrived beside tool results precedes them, matching the order the
  // model produced the calls in.
  const text = blocks
    .filter(block => block.type !== 'tool-result')
    .map(block => projectionOf(block))
    .filter(Boolean)
    .join('\n')
  if (text) out.push({ role: 'user', content: text })

  for (const result of results) {
    out.push({
      role: 'tool',
      tool_call_id: result.toolCallId,
      content: resultText(result)
    })
  }
  return out
}

/** Flatten every text-bearing block of a message. */
function textOf(content: Message['content'] | null | undefined): string {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  return content.map(block => projectionOf(block)).filter(Boolean).join('\n')
}

/**
 * Project one block into the text it contributes to a request.
 *
 * `image` and `file` blocks become an inert placeholder rather than being
 * dropped silently: this model declares text-only input, and a conversation
 * whose attachment vanished without a trace is harder to reason about than one
 * that says so.
 */
function projectionOf(block: ContentBlock): string {
  if (block === null || block === undefined) return ''
  switch (block.type) {
    case 'text':
      return block.text || ''
    case 'image':
      return '[图片附件：本模型仅支持文本输入，未随请求发送]'
    case 'file':
      return '[文件附件：本模型仅支持文本输入，未随请求发送]'
    default:
      return ''
  }
}

/** Flatten one tool result's content into the string a `tool` message carries. */
function resultText(block: ToolResultBlock): string {
  const body = Array.isArray(block.content)
    ? block.content
        .map(part => (part?.type === 'text' ? part.text || '' : ''))
        .filter(Boolean)
        .join('\n')
    : ''
  if (block.isError) return body ? `ERROR: ${body}` : 'ERROR'
  return body
}

/**
 * Guarantee a syntactically valid parameter schema.
 *
 * llama.cpp converts the schema into a grammar, and an empty or missing
 * `properties` object is the one shape that reliably trips the conversion;
 * normalising here keeps a tool with no arguments usable.
 */
function normaliseParameters(parameters: Record<string, unknown> | undefined): Record<string, unknown> {
  if (parameters === null || parameters === undefined || typeof parameters !== 'object') {
    return { type: 'object', properties: {} }
  }
  if (parameters.type === undefined) return { type: 'object', properties: {}, ...parameters }
  return parameters
}

/** One parsed SSE event from the completion stream. */
interface WireChunk {
  choices?: Array<{
    index?: number
    delta?: {
      content?: string | null
      reasoning_content?: string | null
      tool_calls?: Array<{
        index?: number
        id?: string
        function?: { name?: string; arguments?: string }
      }>
    }
    finish_reason?: string | null
  }>
  usage?: {
    prompt_tokens?: number
    completion_tokens?: number
    total_tokens?: number
    prompt_tokens_details?: { cached_tokens?: number }
    completion_tokens_details?: { reasoning_tokens?: number }
  }
}

/** A tool call still being assembled across deltas. */
interface PendingCall {
  /** DSH block index this call occupies. */
  blockIndex: number
  /** Provider-issued id, defaulted when the server sent none. */
  id: string
  /** Function name, observed on the first delta. */
  name: string
  /** Raw JSON argument text accumulated so far. */
  args: string
  /** Whether the block-start chunk has been emitted. */
  opened: boolean
}

/**
 * Translate one OpenAI-compatible SSE response into DSH `StreamChunk`s.
 *
 * Block indices follow DSH's contract: a `block-start` precedes that block's
 * deltas, and `block-end` carries the assembled block. Text and reasoning are
 * tracked as one active block at a time (they alternate), while each tool call
 * keeps its own block for the whole stream.
 *
 * @param response - the `fetch` response whose body is `text/event-stream`.
 * @param newId - id factory, injected so tests can be deterministic.
 * @yields DSH stream chunks, ending with exactly one `finish`.
 */
export async function* parseStream(
  response: Response,
  newId: () => string = () => `call_${Math.random().toString(36).slice(2, 10)}`
): AsyncGenerator<StreamChunk, void, unknown> {
  const reader = response.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let nextIndex = 0
  let active: { type: 'text' | 'reasoning'; index: number; text: string } | null = null
  const calls = new Map<number, PendingCall>()
  let order: number[] = []
  let sawToolCall = false
  let finishReason: 'stop' | 'max-tokens' = 'stop'
  let usage: TokenUsage | null = null

  /** Close the currently open text or reasoning block. */
  const endActive = (): StreamChunk | null => {
    if (active === null) return null
    const block = { type: active.type, text: active.text } as ContentBlock
    const chunk: StreamChunk = { type: 'block-end', index: active.index, block }
    active = null
    return chunk
  }

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed.startsWith('data:')) continue
        const raw = trimmed.slice(5).trim()
        if (!raw || raw === '[DONE]') continue

        let event: WireChunk
        try {
          event = JSON.parse(raw)
        } catch {
          continue
        }

        if (event.usage) usage = mapUsage(event.usage)
        const choice = event.choices?.[0]
        if (!choice) continue

        if (choice.finish_reason === 'length') finishReason = 'max-tokens'
        const delta = choice.delta
        if (!delta) continue

        // Tool calls first: a call closes whatever text block was open, because
        // DSH correlates blocks by index and a call must not inherit one.
        for (const call of delta.tool_calls || []) {
          const slot = typeof call.index === 'number' ? call.index : 0
          let pending = calls.get(slot)
          if (pending === undefined) {
            const ended = endActive()
            if (ended) yield ended
            pending = { blockIndex: nextIndex++, id: call.id || newId(), name: call.function?.name || '', args: '', opened: false }
            calls.set(slot, pending)
            order.push(slot)
            sawToolCall = true
          }
          if (call.id) pending.id = call.id
          if (call.function?.name) pending.name = pending.name || call.function.name
          if (!pending.opened) {
            pending.opened = true
            yield { type: 'block-start', index: pending.blockIndex, blockType: 'tool-call' }
          }
          const fragment = call.function?.arguments
          if (typeof fragment === 'string' && fragment.length > 0) {
            pending.args += fragment
            yield {
              type: 'tool-call-delta',
              index: pending.blockIndex,
              id: asCallId(pending.id),
              // The name is repeated on the first delta only; later deltas leave
              // it unset so the assembler keeps the value it already learned.
              ...(call.function?.name ? { name: call.function.name } : {}),
              argumentsDelta: fragment
            }
          }
        }

        for (const [kind, text] of [
          ['reasoning', delta.reasoning_content],
          ['text', delta.content]
        ] as const) {
          if (typeof text !== 'string' || text.length === 0) continue
          if (active === null || active.type !== kind) {
            const ended = endActive()
            if (ended) yield ended
            active = { type: kind, index: nextIndex++, text: '' }
            yield { type: 'block-start', index: active.index, blockType: kind }
          }
          active.text += text
          yield kind === 'reasoning'
            ? { type: 'reasoning-delta', index: active.index, text }
            : { type: 'text-delta', index: active.index, text }
        }
      }
    }
  } finally {
    reader.releaseLock()
  }

  const ended = endActive()
  if (ended) yield ended

  // Every tool call must be closed with its assembled block, in first-seen
  // order, so the assembler can hand the loop a complete invocation.
  for (const slot of order) {
    const pending = calls.get(slot)!
    if (!pending.opened) continue
    yield {
      type: 'block-end',
      index: pending.blockIndex,
      block: { type: 'tool-call', id: asCallId(pending.id), name: pending.name, arguments: pending.args || '{}' }
    }
  }

  if (usage) yield { type: 'usage', usage }

  // A response that produced nothing at all must become a retryable failure
  // rather than a successful empty turn: a zero-content `stop` would let the
  // loop commit an empty assistant message and mark the turn complete.
  if (nextIndex === 0 && finishReason === 'stop') {
    yield {
      type: 'finish',
      reason: {
        kind: 'error',
        failure: { message: 'model returned a completed response with no content', code: EMPTY_RESPONSE_CODE }
      } as FinishReason
    }
    return
  }

  // A completed tool call must run even when the same turn also hit the output
  // cap, so tool-calls outranks max-tokens.
  yield { type: 'finish', reason: { kind: sawToolCall ? 'tool-calls' : finishReason } as FinishReason }
}

/**
 * Map OpenAI usage counters onto DSH's disjoint accounting.
 *
 * DSH reports uncached input separately from cache reads, while OpenAI folds
 * them into `prompt_tokens`, so the cached count is subtracted out.
 *
 * @param usage - the wire usage object.
 * @returns a `TokenUsage` whose buckets do not overlap.
 */
export function mapUsage(usage: NonNullable<WireChunk['usage']>): TokenUsage {
  const cached = usage.prompt_tokens_details?.cached_tokens || 0
  const prompt = usage.prompt_tokens || 0
  const completion = usage.completion_tokens || 0
  const reasoning = usage.completion_tokens_details?.reasoning_tokens || 0
  const inputTokens = Math.max(0, prompt - cached)
  return {
    inputTokens,
    outputTokens: completion,
    ...(cached > 0 ? { cacheReadTokens: cached } : {}),
    ...(reasoning > 0 ? { reasoningTokens: reasoning } : {}),
    totalTokens: usage.total_tokens || inputTokens + cached + completion
  }
}
