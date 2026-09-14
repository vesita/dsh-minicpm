/**
 * Model catalogue for the local MiniCPM route.
 *
 * One GGUF file is what a `llama-server` process actually serves, so a
 * catalogue entry is a *quantisation choice plus the file that satisfies it*.
 * Asking for a different entry is what makes the engine reload: the manager
 * compares the file behind the requested id with the file already loaded and
 * restarts only when they differ.
 *
 * @module dsh-minicpm/models
 */
import path from 'node:path'
import { modelsDir, DEFAULT_MODEL_FILE } from './paths.js'
import type { LlmModelInfo, LlmResolvedModelInfo } from '@deepseek-ai/dsh-llm'

/** One selectable local weight. */
export interface ModelSpec {
  /** DSH-facing model id. */
  id: string
  /** Human-readable name. */
  name: string
  /** One-line distinction shown in selectors. */
  description?: string
  /** GGUF file name inside the models directory, or an absolute path. */
  file: string
  /** Approximate on-disk size, for the UI. */
  sizeBytes?: number
}

/**
 * The official OpenBMB quantisations of MiniCPM5-2B, smallest first.
 *
 * The 4-bit file is first because it is the one that leaves the most headroom
 * on an 8 GB card that is also driving a desktop; the larger two are offered
 * for users who close everything else and want maximum fidelity.
 */
export const MODEL_CATALOG: ModelSpec[] = [
  {
    id: 'minicpm5-2b-q4',
    name: 'MiniCPM5-2B (Q4_K_M)',
    description: '4-bit 量化，约 1.6 GB，8 GB 显存下的默认选择',
    file: DEFAULT_MODEL_FILE,
    sizeBytes: 1_561_318_368
  },
  {
    id: 'minicpm5-2b-q8',
    name: 'MiniCPM5-2B (Q8_0)',
    description: '8-bit 量化，约 2.8 GB，接近原始精度',
    file: 'MiniCPM5-2B-Q8_0.gguf',
    sizeBytes: 2_800_000_000
  },
  {
    id: 'minicpm5-2b-f16',
    name: 'MiniCPM5-2B (F16)',
    description: '半精度原始权重，约 5 GB，需要腾空显存',
    file: 'MiniCPM5-2B-F16.gguf',
    sizeBytes: 5_100_000_000
  }
]

/** Upstream Hugging Face repository holding the official GGUF files. */
export const HF_REPO = 'openbmb/MiniCPM5-2B-GGUF'

/** Upstream file name for a catalogue entry. */
export function remoteFileOf(spec: ModelSpec): string {
  return path.basename(spec.file)
}

/** Absolute path of one entry's weight file. */
export function absoluteFileOf(spec: ModelSpec): string {
  return path.isAbsolute(spec.file) ? spec.file : path.join(modelsDir(), spec.file)
}

/**
 * Resolve one requested model id against a catalogue.
 *
 * Resolution is deliberately permissive: `llama-server` does not validate the
 * `model` field of a request (it serves whatever it loaded), and DSH's own
 * contract states that catalogue membership is advisory. An id naming a
 * `.gguf` file therefore resolves to that file, and anything else falls back to
 * the catalogue head so a stale id in durable history still routes.
 *
 * @param id - exact model id from `GenerateOptions.model`.
 * @param catalog - active catalogue, already defaulted.
 * @returns the matching entry, or the fallback head.
 */
export function resolveModelSpec(id: string, catalog: ModelSpec[]): ModelSpec {
  const list = catalog.length > 0 ? catalog : MODEL_CATALOG
  const exact = list.find(spec => spec.id === id)
  if (exact !== undefined) return exact
  // A bare catalogue file name is still a catalogue entry — resolving it here
  // rather than through the custom-path branch is what keeps its description
  // and expected size available to the download UI.
  const byFile = list.find(spec => path.basename(spec.file) === id)
  if (byFile !== undefined) return byFile
  if (typeof id === 'string' && /\.gguf$/i.test(id)) {
    return { id, name: path.basename(id), file: id }
  }
  return list[0]
}

/**
 * Project one entry into DSH's advisory catalogue shape.
 *
 * @param spec - the entry to describe.
 * @param provider - provider route that owns it.
 * @param available - file names actually present on disk, for availability notes.
 * @returns detached `LlmModelInfo`.
 */
export function modelInfoOf(spec: ModelSpec, provider: string, available?: Set<string>): LlmModelInfo {
  const present = available === undefined || available.has(path.basename(spec.file))
  return {
    provider,
    id: spec.id,
    name: spec.name,
    description: present ? spec.description : `${spec.description ?? ''}（权重未下载）`.trim(),
    inputModalities: ['text']
  }
}

/**
 * Project one entry into the exact-route metadata shape.
 *
 * The advertised context window is the engine's live `--ctx-size`, never the
 * model's architectural maximum: a 128k ceiling the server was not started with
 * would let the loop build requests the engine must truncate.
 *
 * @param spec - the entry to describe.
 * @param provider - provider route that owns it.
 * @param contextWindow - context the engine is (or will be) started with.
 * @param defaultMaxTokens - per-request output cap applied when the caller omits one.
 * @returns detached `LlmResolvedModelInfo`.
 */
export function resolvedModelInfoOf(
  spec: ModelSpec,
  provider: string,
  contextWindow: number,
  defaultMaxTokens: number
): LlmResolvedModelInfo {
  return {
    provider,
    id: spec.id,
    name: spec.name,
    ...(spec.description === undefined ? {} : { description: spec.description }),
    context: { contextWindow },
    defaultMaxTokens,
    inputModalities: ['text']
  }
}
