#!/usr/bin/env node
/**
 * A stand-in for `llama-server` used by the engine tests.
 *
 * It accepts the same argument shape the engine builds, ignores everything but
 * `--port` and `--host`, and — crucially — takes a configurable delay before it
 * starts answering `/health`. That delay is what makes the concurrency
 * regression reproducible: a second caller that arrives during the delay must
 * wait, not conclude the server is dead.
 */
import http from 'node:http'

const argv = process.argv.slice(2)

/** Read the value following a flag. */
function valueOf(flag, fallback) {
  const index = argv.indexOf(flag)
  return index >= 0 && index + 1 < argv.length ? argv[index + 1] : fallback
}

const port = Number(valueOf('--port', '0'))
const host = valueOf('--host', '127.0.0.1')
const readyAfterMs = Number(process.env.FAKE_READY_MS || '1500')

// Output before readiness mirrors the real server's startup banner, so the
// engine's log capture is exercised too.
process.stderr.write(`fake llama-server: pid ${process.pid}, ready in ${readyAfterMs}ms on ${host}:${port}\n`)

const server = http.createServer((request, response) => {
  if (request.url === '/health') {
    response.writeHead(200, { 'content-type': 'application/json' })
    response.end(JSON.stringify({ status: 'ok' }))
    return
  }
  response.writeHead(404)
  response.end()
})

setTimeout(() => {
  server.listen(port, host, () => {
    process.stderr.write(`fake llama-server: listening on http://${host}:${port}\n`)
  })
}, readyAfterMs)
