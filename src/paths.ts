/**
 * Filesystem layout owned by `dsh-minicpm`.
 *
 * Everything this plugin manages lives under one directory so a user can
 * inspect, back up, or delete the whole local-model installation as a unit:
 *
 * ```
 * ~/.dsh/minicpm/
 * ├── engine/            llama.cpp prebuilt runtime (llama-server + shared libs)
 * ├── models/            *.gguf weights
 * └── state.json         last selected model + engine preferences
 * ```
 *
 * @module dsh-minicpm/paths
 */
import os from 'node:os'
import path from 'node:path'

/** The harness home directory, matching every other DSH plugin. */
export function dshHome(): string {
  return process.env.DSH_HOME || path.join(os.homedir(), '.dsh')
}

/** Root of everything this plugin owns. */
export function rootDir(): string {
  return process.env.DSH_MINICPM_HOME || path.join(dshHome(), 'minicpm')
}

/** Where GGUF weights live. */
export function modelsDir(): string {
  return path.join(rootDir(), 'models')
}

/** Where the llama.cpp runtime lives. */
export function engineDir(): string {
  return path.join(rootDir(), 'engine')
}

/** Small JSON file holding selections that are not worth a settings entry. */
export function statePath(): string {
  return path.join(rootDir(), 'state.json')
}

/** Default file name of the recommended 4-bit weight. */
export const DEFAULT_MODEL_FILE = 'MiniCPM5-2B-Q4_K_M.gguf'

/** Absolute default path of the recommended 4-bit weight. */
export function defaultModelPath(): string {
  return path.join(modelsDir(), DEFAULT_MODEL_FILE)
}
