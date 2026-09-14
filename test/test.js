/**
 * Unit tests for the pure halves of `dsh-minicpm`.
 *
 * Everything here runs without a model, without a GPU, and without a network:
 * the wire translation, the archive reader and the catalogue are all functions
 * over plain data, which is precisely why they were factored that way.
 *
 * Run with `npm test` (which builds first, since the tests import `lib/`).
 */
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { extractTar, extractTarGz } from '../lib/archive.js'
import { buildRequest, mapUsage, parseStream } from '../lib/wire.js'
import { MODEL_CATALOG, resolveModelSpec, modelInfoOf, resolvedModelInfoOf } from '../lib/models.js'
import { buildArgs, DEFAULT_ENGINE, EngineManager } from '../lib/engine.js'

let passed = 0
let failed = 0

/** Run one named assertion block, recording the outcome. */
async function test(name, body) {
  try {
    await body()
    passed += 1
    process.stdout.write(`  ✓ ${name}\n`)
  } catch (error) {
    failed += 1
    process.stdout.write(`  ✗ ${name}\n    ${error.message}\n`)
  }
}

/** Build a `Response` whose body yields the given SSE frames. */
function sseResponse(frames) {
  const encoder = new TextEncoder()
  const stream = new ReadableStream({
    start(controller) {
      for (const frame of frames) controller.enqueue(encoder.encode(frame))
      controller.close()
    }
  })
  return new Response(stream, { status: 200 })
}

/** Collect a `parseStream` run into one array. */
async function collect(response, newId) {
  const chunks = []
  for await (const chunk of parseStream(response, newId)) chunks.push(chunk)
  return chunks
}

/** One OpenAI-compatible delta frame. */
function frame(delta, finish) {
  return `data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: finish ?? null }] })}\n\n`
}

process.stdout.write('wire: buildRequest\n')

await test('hoists every system message into one leading system turn', () => {
  const body = buildRequest(
    {
      provider: 'minicpm-local',
      model: 'm',
      messages: [
        { id: '1', role: 'system', content: [{ type: 'text', text: 'first' }], source: { kind: 'plugin', plugin: 'p' } },
        { id: '2', role: 'user', content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' } },
        { id: '3', role: 'system', content: [{ type: 'text', text: 'second' }], source: { kind: 'plugin', plugin: 'p' } }
      ]
    },
    'wire-model'
  )
  assert.equal(body.model, 'wire-model')
  assert.equal(body.messages.length, 2)
  assert.equal(body.messages[0].role, 'system')
  assert.equal(body.messages[0].content, 'first\n\nsecond')
  assert.equal(body.messages[1].role, 'user')
  assert.equal(body.stream, true)
  assert.deepEqual(body.stream_options, { include_usage: true })
})

await test('splits tool results into role:tool messages keyed by call id', () => {
  const body = buildRequest(
    {
      provider: 'minicpm-local',
      model: 'm',
      messages: [
        {
          id: '1',
          role: 'user',
          content: [{ type: 'tool-result', toolCallId: 'call_a', content: [{ type: 'text', text: '42' }] }],
          source: { kind: 'tool', callId: 'call_a' }
        }
      ]
    },
    'm'
  )
  assert.equal(body.messages.length, 1)
  assert.equal(body.messages[0].role, 'tool')
  assert.equal(body.messages[0].tool_call_id, 'call_a')
  assert.equal(body.messages[0].content, '42')
})

await test('marks an errored tool result in the text the model receives', () => {
  const body = buildRequest(
    {
      provider: 'minicpm-local',
      model: 'm',
      messages: [
        {
          id: '1',
          role: 'user',
          content: [
            { type: 'tool-result', toolCallId: 'c', content: [{ type: 'text', text: 'boom' }], isError: true }
          ],
          source: { kind: 'tool', callId: 'c' }
        }
      ]
    },
    'm'
  )
  assert.equal(body.messages[0].content, 'ERROR: boom')
})

await test('projects assistant tool calls back onto the wire', () => {
  const body = buildRequest(
    {
      provider: 'minicpm-local',
      model: 'm',
      messages: [
        {
          id: '1',
          role: 'assistant',
          content: [
            { type: 'text', text: 'let me check' },
            { type: 'tool-call', id: 'call_1', name: 'get_weather', arguments: '{"city":"Paris"}' }
          ],
          source: { kind: 'model', provider: 'minicpm-local', model: 'm' }
        }
      ]
    },
    'm'
  )
  assert.equal(body.messages[0].role, 'assistant')
  assert.equal(body.messages[0].content, 'let me check')
  assert.equal(body.messages[0].tool_calls.length, 1)
  assert.equal(body.messages[0].tool_calls[0].id, 'call_1')
  assert.equal(body.messages[0].tool_calls[0].function.name, 'get_weather')
  assert.equal(body.messages[0].tool_calls[0].function.arguments, '{"city":"Paris"}')
})

await test('drops reasoning blocks instead of replaying them', () => {
  const body = buildRequest(
    {
      provider: 'minicpm-local',
      model: 'm',
      messages: [
        {
          id: '1',
          role: 'assistant',
          content: [
            { type: 'reasoning', text: 'private trace' },
            { type: 'text', text: 'answer' }
          ],
          source: { kind: 'model', provider: 'minicpm-local', model: 'm' }
        }
      ]
    },
    'm'
  )
  assert.equal(body.messages[0].content, 'answer')
})

await test('replaces an image block with an explicit text placeholder', () => {
  const body = buildRequest(
    {
      provider: 'minicpm-local',
      model: 'm',
      messages: [
        {
          id: '1',
          role: 'user',
          content: [{ type: 'image', attachment: { id: 'x' } }],
          source: { kind: 'user' }
        }
      ]
    },
    'm'
  )
  assert.match(body.messages[0].content, /图片附件/)
})

await test('maps tools and normalises a missing parameter schema', () => {
  const body = buildRequest(
    {
      provider: 'minicpm-local',
      model: 'm',
      messages: [{ id: '1', role: 'user', content: [{ type: 'text', text: 'x' }], source: { kind: 'user' } }],
      tools: [
        { name: 'noargs', description: '', parameters: {} },
        { name: 'args', description: 'd', parameters: { type: 'object', properties: { a: { type: 'string' } } } }
      ]
    },
    'm'
  )
  assert.equal(body.tools.length, 2)
  assert.deepEqual(body.tools[0].function.parameters, { type: 'object', properties: {} })
  assert.equal(body.tools[1].function.parameters.properties.a.type, 'string')
})

await test('never sends an empty conversation', () => {
  const body = buildRequest(
    {
      provider: 'minicpm-local',
      model: 'm',
      messages: [
        { id: '1', role: 'system', content: [{ type: 'text', text: 'only a prompt' }], source: { kind: 'plugin', plugin: 'p' } }
      ]
    },
    'm'
  )
  assert.equal(body.messages.length, 2)
  assert.equal(body.messages[1].role, 'user')
})

await test('passes generation controls through, omitting absent ones', () => {
  const body = buildRequest(
    {
      provider: 'minicpm-local',
      model: 'm',
      messages: [{ id: '1', role: 'user', content: [{ type: 'text', text: 'x' }], source: { kind: 'user' } }],
      temperature: 0.2,
      maxTokens: 128,
      stop: ['</s>']
    },
    'm'
  )
  assert.equal(body.temperature, 0.2)
  assert.equal(body.max_tokens, 128)
  assert.deepEqual(body.stop, ['</s>'])
  assert.equal('temperature' in buildRequest({ provider: 'p', model: 'm', messages: [] }, 'm'), false)
})

process.stdout.write('wire: parseStream\n')

await test('emits text block start, deltas, block end, then finish', async () => {
  const chunks = await collect(
    sseResponse([frame({ content: 'Hel' }), frame({ content: 'lo' }), frame({}, 'stop'), 'data: [DONE]\n\n'])
  )
  assert.deepEqual(chunks, [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text: 'Hel' },
    { type: 'text-delta', index: 0, text: 'lo' },
    { type: 'block-end', index: 0, block: { type: 'text', text: 'Hello' } },
    { type: 'finish', reason: { kind: 'stop' } }
  ])
})

await test('keeps reasoning and text in separate indexed blocks', async () => {
  const chunks = await collect(
    sseResponse([frame({ reasoning_content: 'think' }), frame({ content: 'say' }), frame({}, 'stop')])
  )
  const blocks = chunks.filter(chunk => chunk.type === 'block-start').map(chunk => chunk.blockType)
  assert.deepEqual(blocks, ['reasoning', 'text'])
  assert.deepEqual(
    chunks.filter(chunk => chunk.type === 'block-end').map(chunk => chunk.index),
    [0, 1]
  )
})

await test('assembles a fragmented tool call and reports tool-calls', async () => {
  const chunks = await collect(
    sseResponse([
      frame({ tool_calls: [{ index: 0, id: 'call_z', function: { name: 'get_weather', arguments: '{"ci' } }] }),
      frame({ tool_calls: [{ index: 0, function: { arguments: 'ty":"Paris"}' } }] }),
      frame({}, 'tool_calls')
    ])
  )
  assert.deepEqual(chunks[0], { type: 'block-start', index: 0, blockType: 'tool-call' })
  const deltas = chunks.filter(chunk => chunk.type === 'tool-call-delta')
  assert.equal(deltas.length, 2)
  assert.equal(deltas[0].id, 'call_z')
  assert.equal(deltas[0].name, 'get_weather')
  // The name rides only the first delta; repeating it would overwrite a value
  // the assembler already learned from a later frame that omits it.
  assert.equal('name' in deltas[1], false)
  const end = chunks.find(chunk => chunk.type === 'block-end')
  assert.equal(end.block.name, 'get_weather')
  assert.equal(end.block.arguments, '{"city":"Paris"}')
  assert.equal(end.block.id, 'call_z')
  assert.deepEqual(chunks.at(-1), { type: 'finish', reason: { kind: 'tool-calls' } })
})

await test('lets tool-calls outrank max-tokens on the same turn', async () => {
  const chunks = await collect(
    sseResponse([
      frame({ tool_calls: [{ index: 0, id: 'c', function: { name: 'f', arguments: '{}' } }] }, 'length')
    ])
  )
  assert.deepEqual(chunks.at(-1), { type: 'finish', reason: { kind: 'tool-calls' } })
})

await test('maps length to max-tokens when no call was made', async () => {
  const chunks = await collect(sseResponse([frame({ content: 'x' }, 'length')]))
  assert.deepEqual(chunks.at(-1), { type: 'finish', reason: { kind: 'max-tokens' } })
})

await test('turns a zero-content stop into a retryable EMPTY_RESPONSE failure', async () => {
  const chunks = await collect(sseResponse([frame({}, 'stop')]))
  assert.equal(chunks.length, 1)
  assert.equal(chunks[0].type, 'finish')
  assert.equal(chunks[0].reason.kind, 'error')
  assert.equal(chunks[0].reason.failure.code, 'EMPTY_RESPONSE')
})

await test('surfaces usage before the terminal finish', async () => {
  const chunks = await collect(
    sseResponse([
      frame({ content: 'hi' }),
      `data: ${JSON.stringify({
        choices: [],
        usage: {
          prompt_tokens: 100,
          completion_tokens: 20,
          total_tokens: 120,
          prompt_tokens_details: { cached_tokens: 40 },
          completion_tokens_details: { reasoning_tokens: 5 }
        }
      })}\n\n`,
      frame({}, 'stop')
    ])
  )
  const usageIndex = chunks.findIndex(chunk => chunk.type === 'usage')
  const finishIndex = chunks.findIndex(chunk => chunk.type === 'finish')
  assert.ok(usageIndex > 0 && usageIndex < finishIndex, 'usage must precede finish')
  const usage = chunks[usageIndex].usage
  assert.equal(usage.inputTokens, 60)
  assert.equal(usage.cacheReadTokens, 40)
  assert.equal(usage.outputTokens, 20)
  assert.equal(usage.reasoningTokens, 5)
  assert.equal(usage.totalTokens, 120)
})

await test('survives a data frame split across reads', async () => {
  const payload = `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: 'split' } }] })}\n\n`
  const chunks = await collect(sseResponse([payload.slice(0, 20), payload.slice(20), frame({}, 'stop')]))
  assert.equal(chunks.find(chunk => chunk.type === 'block-end').block.text, 'split')
})

await test('ignores unparsable frames rather than failing the stream', async () => {
  const chunks = await collect(sseResponse(['data: {not json}\n\n', frame({ content: 'ok' }, 'stop')]))
  assert.equal(chunks.find(chunk => chunk.type === 'block-end').block.text, 'ok')
})

await test('uses the injected id factory when the server omits one', async () => {
  const chunks = await collect(
    sseResponse([frame({ tool_calls: [{ index: 0, function: { name: 'f', arguments: '{}' } }] })]),
    () => 'generated-id'
  )
  assert.equal(chunks.find(chunk => chunk.type === 'block-end').block.id, 'generated-id')
})

process.stdout.write('wire: mapUsage\n')

await test('derives disjoint buckets when the provider folds cache into prompt', () => {
  const usage = mapUsage({ prompt_tokens: 10, completion_tokens: 3, total_tokens: 13, prompt_tokens_details: { cached_tokens: 4 } })
  assert.equal(usage.inputTokens, 6)
  assert.equal(usage.cacheReadTokens, 4)
  assert.equal(usage.outputTokens, 3)
  assert.equal('reasoningTokens' in usage, false)
})

process.stdout.write('models\n')

await test('resolves catalogue ids, bare file names and unknown ids permissively', () => {
  assert.equal(resolveModelSpec('minicpm5-2b-q8', MODEL_CATALOG).id, 'minicpm5-2b-q8')
  assert.equal(resolveModelSpec('MiniCPM5-2B-Q8_0.gguf', MODEL_CATALOG).id, 'minicpm5-2b-q8')
  assert.equal(resolveModelSpec('totally-unknown', MODEL_CATALOG).id, MODEL_CATALOG[0].id)
  assert.equal(resolveModelSpec('/tmp/custom.gguf', MODEL_CATALOG).file, '/tmp/custom.gguf')
})

await test('marks an entry whose weights are missing', () => {
  const spec = MODEL_CATALOG[0]
  assert.match(modelInfoOf(spec, 'p', new Set()).description, /未下载/)
  assert.equal(modelInfoOf(spec, 'p', new Set([path.basename(spec.file)])).description, spec.description)
})

await test('advertises the engine context window, not the architectural ceiling', () => {
  const info = resolvedModelInfoOf(MODEL_CATALOG[0], 'p', 8192, 4096)
  assert.equal(info.context.contextWindow, 8192)
  assert.equal(info.defaultMaxTokens, 4096)
  assert.deepEqual(info.inputModalities, ['text'])
})

process.stdout.write('engine: buildArgs\n')

await test('always passes the flags the route depends on', () => {
  const args = buildArgs(DEFAULT_ENGINE, '/tmp/m.gguf')
  for (const flag of ['-m', '/tmp/m.gguf', '--jinja', '--no-ui', '-ngl', '-c', '--host', '--port']) {
    assert.ok(args.includes(flag), `missing ${flag}`)
  }
  assert.equal(args[args.indexOf('-m') + 1], '/tmp/m.gguf')
})

await test('appends extra args verbatim and omits optional ones', () => {
  const args = buildArgs({ ...DEFAULT_ENGINE, extraArgs: ['--flash-attn', 'on'] }, '/tmp/m.gguf')
  assert.ok(args.includes('--flash-attn'))
  assert.equal(args.includes('-t'), false)
  assert.equal(args.includes('--no-reasoning-preserve'), false)
})

await test('honours an explicit thread count and reasoning switch', () => {
  const args = buildArgs({ ...DEFAULT_ENGINE, threads: 6, preserveReasoning: false }, '/tmp/m.gguf')
  assert.equal(args[args.indexOf('-t') + 1], '6')
  assert.ok(args.includes('--no-reasoning-preserve'))
})

process.stdout.write('engine: lifecycle\n')

/** Configuration pointing the engine at the fake server on a chosen port. */
function fakeEngineConfig(port, extra = {}) {
  return {
    ...DEFAULT_ENGINE,
    binary: path.join(import.meta.dirname, 'fixtures', 'fake-llama-server.mjs'),
    port,
    contextSize: 4096,
    idleUnloadSeconds: 0,
    ...extra
  }
}

/** Build a throwaway directory holding one stub weight file. */
function stubModel(work, name = 'm.gguf') {
  fs.mkdirSync(work, { recursive: true })
  const model = path.join(work, name)
  fs.writeFileSync(model, 'stub')
  return model
}

await test('serves a weight the running child already loaded, without respawning', async () => {
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'dsh-minicpm-engine-'))
  const model = stubModel(work)
  const config = fakeEngineConfig(8171)
  const engine = new EngineManager(() => config, {})
  try {
    const first = await engine.ensure(model)
    const pid = engine.status().pid
    const second = await engine.ensure(model)
    assert.equal(first, second)
    assert.equal(engine.status().pid, pid, 'a second ensure() must reuse the running child')
  } finally {
    await engine.dispose()
    fs.rmSync(work, { recursive: true, force: true })
  }
})

await test('concurrent callers wait for a starting server instead of killing it', async () => {
  // The regression this guards: a second ensure() arriving while the server is
  // still starting probes `/health`, is refused because the port is not open
  // yet, and concludes the child is dead. It then stops the very process the
  // first caller is waiting for, and that caller fails after the full ready
  // timeout. One agent turn issuing a title request alongside the main request
  // is enough to trigger it.
  //
  // The window only opens once the first caller has actually spawned its child
  // while the server is not yet answering, so the second call is issued at that
  // observed moment rather than alongside the first — issuing both in the same
  // microtask would let the second join the start before it ever spawns, and
  // the test would pass against the broken implementation.
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'dsh-minicpm-engine-'))
  const model = stubModel(work)
  const config = fakeEngineConfig(8172)
  const engine = new EngineManager(() => config, {})
  const previous = process.env.FAKE_READY_MS
  process.env.FAKE_READY_MS = '1500'
  try {
    const first = engine.ensure(model)
    const deadline = Date.now() + 5000
    while (engine.status().pid === null && Date.now() < deadline) await new Promise(r => setTimeout(r, 10))
    assert.ok(engine.status().pid !== null, 'the child should have spawned by now')

    const second = engine.ensure(model)
    const results = await Promise.all([first, second])
    assert.equal(new Set(results).size, 1, 'every concurrent caller must resolve to one endpoint')
    assert.equal(engine.status().running, true)
    assert.ok(engine.status().pid !== null, 'the child must still be alive')
  } finally {
    if (previous === undefined) delete process.env.FAKE_READY_MS
    else process.env.FAKE_READY_MS = previous
    await engine.dispose()
    fs.rmSync(work, { recursive: true, force: true })
  }
})

await test('reloads when a different weight is requested', async () => {
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'dsh-minicpm-engine-'))
  const first = stubModel(work, 'a.gguf')
  const second = stubModel(work, 'b.gguf')
  const config = fakeEngineConfig(8173)
  const engine = new EngineManager(() => config, {})
  const previous = process.env.FAKE_READY_MS
  process.env.FAKE_READY_MS = '50'
  try {
    await engine.ensure(first)
    const pid = engine.status().pid
    await engine.ensure(second)
    assert.equal(engine.status().loadedModel, second)
    assert.notEqual(engine.status().pid, pid, 'a different weight must be a new process')
  } finally {
    if (previous === undefined) delete process.env.FAKE_READY_MS
    else process.env.FAKE_READY_MS = previous
    await engine.dispose()
    fs.rmSync(work, { recursive: true, force: true })
  }
})

await test('reports a missing runtime as an actionable failure, not a timeout', async () => {
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'dsh-minicpm-engine-'))
  const model = stubModel(work)
  const config = fakeEngineConfig(8174, { binary: path.join(work, 'does-not-exist') })
  const engine = new EngineManager(() => config, {})
  try {
    await assert.rejects(
      () => engine.ensure(model),
      error => {
        assert.equal(error.code, 'ENGINE_MISSING')
        assert.match(error.message, /未找到 llama-server 运行时/)
        return true
      }
    )
  } finally {
    await engine.dispose()
    fs.rmSync(work, { recursive: true, force: true })
  }
})

await test('reports a missing weight file before spawning anything', async () => {
  const config = fakeEngineConfig(8175)
  const engine = new EngineManager(() => config, {})
  try {
    await assert.rejects(
      () => engine.ensure('/nonexistent/model.gguf'),
      error => {
        assert.equal(error.code, 'MODEL_MISSING')
        return true
      }
    )
  } finally {
    await engine.dispose()
  }
})

process.stdout.write('archive\n')

await test('extracts a tarball byte-for-byte, preserving the executable bit', () => {
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'dsh-minicpm-archive-'))
  const source = path.join(work, 'src')
  fs.mkdirSync(path.join(source, 'nested'), { recursive: true })
  fs.writeFileSync(path.join(source, 'a.txt'), 'alpha\n')
  fs.writeFileSync(path.join(source, 'nested', 'b.bin'), Buffer.from([0, 1, 2, 255, 0]))
  fs.writeFileSync(path.join(source, 'run.sh'), '#!/bin/sh\necho hi\n')
  fs.chmodSync(path.join(source, 'run.sh'), 0o755)
  // A name past the 100-byte header field, which forces the GNU long-name form.
  const longName = `${'d'.repeat(120)}.txt`
  fs.writeFileSync(path.join(source, longName), 'long\n')

  const archive = path.join(work, 'bundle.tar.gz')
  execFileSync('tar', ['-czf', archive, '-C', work, 'src'])
  const destination = path.join(work, 'out')
  extractTarGz(fs.readFileSync(archive), destination)

  assert.equal(fs.readFileSync(path.join(destination, 'src', 'a.txt'), 'utf8'), 'alpha\n')
  assert.deepEqual(
    fs.readFileSync(path.join(destination, 'src', 'nested', 'b.bin')),
    Buffer.from([0, 1, 2, 255, 0])
  )
  assert.equal(fs.readFileSync(path.join(destination, 'src', longName), 'utf8'), 'long\n')
  assert.equal(fs.statSync(path.join(destination, 'src', 'run.sh')).mode & 0o777, 0o755)
  fs.rmSync(work, { recursive: true, force: true })
})

await test('rejects an entry that would escape the destination', () => {
  // Hand-build a tar with a traversal path; no tar writer is needed for one
  // header plus its payload.
  const name = '../escaped.txt'
  const body = Buffer.from('nope')
  const header = Buffer.alloc(512)
  header.write(name, 0, 'utf8')
  header.write('0000644\0', 100, 'utf8')
  header.write('0000000\0', 108, 'utf8')
  header.write('0000000\0', 116, 'utf8')
  header.write(`${body.length.toString(8).padStart(11, '0')}\0`, 124, 'utf8')
  header.write(`${Math.floor(Date.now() / 1000).toString(8).padStart(11, '0')}\0`, 136, 'utf8')
  header.write('        ', 148, 'utf8')
  header.write('0', 156, 'utf8')
  header.write('ustar\0', 257, 'utf8')
  let sum = 0
  for (const byte of header) sum += byte
  header.write(`${sum.toString(8).padStart(6, '0')}\0 `, 148, 'utf8')

  const payload = Buffer.alloc(512)
  body.copy(payload)
  const archive = Buffer.concat([header, payload, Buffer.alloc(1024)])

  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'dsh-minicpm-escape-'))
  assert.throws(() => extractTar(archive, work), /越出目标目录/)
  assert.equal(fs.existsSync(path.join(path.dirname(work), 'escaped.txt')), false)
  fs.rmSync(work, { recursive: true, force: true })
})

process.stdout.write(`\n${passed} passed, ${failed} failed\n`)
process.exitCode = failed === 0 ? 0 : 1
