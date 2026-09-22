(window as any).__ModuleLoader__.load({
  id: 'dsh-minicpm',
  factory: (require: (id: string) => any) => {
    var module = { exports: {} as any }
    var exports = module.exports
    Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' })

    const React = require('react')
    const primitives = require('@deepseek-ai/dsh-client-ui-primitives')
    const { Button } = primitives
    const h = React.createElement

    /**
     * The profile entry id this package's Config lives under — the `id:` the
     * profile patch declares (`cordis.patch.yml` → `id: minicpm`), which is also
     * the namespace `configForms` addresses.
     *
     * Not to be confused with the historical `llm-minicpm` provider-settings
     * namespace: that one addresses the directory row in Settings → Models, and
     * it too is now the entry id (the host half reads it off its own fiber).
     */
    const CONFIG_NS = 'minicpm'
    /** The bundle package name the Plugins page keys this card by. */
    const BUNDLE_KEY = 'dsh-minicpm'

    /**
     * Browser half of `dsh-minicpm`: the management card inside
     * Settings → Models for the `llm-minicpm` namespace.
     *
     * The card is the whole user interface for the local route. It shows what
     * the engine is doing right now, lets the two long downloads be started and
     * cancelled, and never writes settings: everything it changes is host state
     * reached over the loopback routes the host half registers.
     *
     * Every element is built with `React.createElement`, whose trailing
     * arguments are children. The automatic `jsx`/`jsxs` runtime instead reads
     * children from `props.children` and treats its third argument as the key,
     * so calling it in `createElement` shape would silently render empty
     * elements.
     */

    /** `EngineStatus` as the host serialises it. */
    interface EngineStatus {
      running: boolean
      starting: boolean
      mode: string
      baseURL: string
      pid: number | null
      loadedModel: string | null
      binaryPath: string
      binaryPresent: boolean
      lastError: string | null
      inFlight: number
      idleSecondsLeft: number | null
      log: string[]
    }

    /** GPU memory as the host reads it from `nvidia-smi`. */
    interface Gpu {
      usedMiB: number | null
      totalMiB: number | null
      freeMiB: number | null
      name: string | null
    }

    /** One catalogue entry with its on-disk state. */
    interface ModelRow {
      id: string
      name: string
      description: string
      file: string
      path: string
      present: boolean
      sizeBytes: number | null
      expectedBytes: number | null
    }

    /** One acquisition in flight or recently finished. */
    interface Job {
      id: string
      kind: 'model' | 'engine'
      label: string
      status: 'running' | 'done' | 'error' | 'cancelled'
      doneBytes: number
      totalBytes: number | null
      error: string | null
      detail: string | null
      bytesPerSecond: number | null
    }

    /** The host's full view model. */
    interface Status {
      engine: EngineStatus
      gpu: Gpu
      models: ModelRow[]
      jobs: Job[]
      activeModel: string
      provider: string
      providerName: string
      engineDir: string
      modelsDir: string
      hfEndpoint: string
    }

    const ROUTE = '/dsh-minicpm'
    const PROVIDER = 'minicpm-local'
    /** Poll cadence while something is moving, and while nothing is. */
    const POLL_BUSY_MS = 1200
    const POLL_IDLE_MS = 6000

    const zh = typeof navigator !== 'undefined' && /^zh/i.test(navigator.language || '')
    const copy = zh
      ? {
          running: '引擎运行中',
          stopped: '引擎未启动',
          starting: '引擎启动中…',
          external: '外部端点',
          vram: '显存',
          port: '端口',
          start: '启动引擎',
          stop: '停止引擎',
          refresh: '刷新',
          idleIn: '空闲卸载倒计时',
          idleOff: '常驻',
          weights: '模型权重',
          runtime: '推理引擎',
          runtimeReady: '运行时已就绪',
          runtimeMissing: '未找到 llama-server 运行时',
          download: '下载',
          downloadRuntime: '下载推理引擎',
          downloading: '下载中',
          reinstall: '重新下载',
          present: '已就绪',
          absent: '未下载',
          cancel: '取消',
          done: '完成',
          failed: '失败',
          cancelled: '已取消',
          log: '引擎日志',
          noLog: '暂无日志',
          release: '立即释放显存',
          unsupported: '不支持的操作',
          rate: '/秒'
        }
      : {
          running: 'Engine running',
          stopped: 'Engine stopped',
          starting: 'Engine starting…',
          external: 'External endpoint',
          vram: 'VRAM',
          port: 'Port',
          start: 'Start engine',
          stop: 'Stop engine',
          refresh: 'Refresh',
          idleIn: 'Unloads in',
          idleOff: 'Stays resident',
          weights: 'Weights',
          runtime: 'Runtime',
          runtimeReady: 'Runtime ready',
          runtimeMissing: 'llama-server runtime not found',
          download: 'Download',
          downloadRuntime: 'Download runtime',
          downloading: 'Downloading',
          reinstall: 'Re-download',
          present: 'ready',
          absent: 'not downloaded',
          cancel: 'Cancel',
          done: 'done',
          failed: 'failed',
          cancelled: 'cancelled',
          log: 'Engine log',
          noLog: 'No output yet',
          release: 'Release VRAM now',
          unsupported: 'Unsupported action',
          rate: '/s'
        }

    /** One JSON call against the host's loopback routes. */
    async function call(path: string, init?: RequestInit): Promise<any> {
      const response = await fetch(`${ROUTE}${path}`, {
        headers: { accept: 'application/json', 'content-type': 'application/json' },
        ...init
      })
      let payload: any = {}
      try {
        payload = await response.json()
      } catch {
        payload = {}
      }
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`)
      return payload
    }

    /** Format a byte count for a human. */
    function fmtBytes(value: number | null | undefined): string {
      if (value === null || value === undefined || !isFinite(value)) return '—'
      const units = ['B', 'KB', 'MB', 'GB', 'TB']
      let size = Math.abs(value)
      let unit = 0
      while (size >= 1024 && unit < units.length - 1) {
        size /= 1024
        unit += 1
      }
      return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`
    }

    /** Format MiB as a compact GB string. */
    function fmtVram(mib: number | null): string {
      if (mib === null || !isFinite(mib)) return '—'
      return `${(mib / 1024).toFixed(1)} GB`
    }

    /** Styles, all built from tokens the Theme service publishes. */
    const styles = {
      wrap: { display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '8px' },
      row: { display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' as const },
      dot: { width: '8px', height: '8px', borderRadius: '50%', flex: 'none' },
      label: { fontSize: '12px', lineHeight: '18px', color: 'var(--dsw-alias-label-secondary)' },
      strong: { fontSize: '12px', lineHeight: '18px', color: 'var(--dsw-alias-label-primary)' },
      hint: { fontSize: '12px', lineHeight: '18px', color: 'var(--dsw-alias-label-secondary)', opacity: '0.75' },
      error: { fontSize: '12px', lineHeight: '18px', color: 'var(--dsw-alias-state-error-primary)', margin: 0 },
      panel: {
        display: 'flex',
        flexDirection: 'column' as const,
        gap: '6px',
        padding: '8px 10px',
        borderRadius: '8px',
        background: 'var(--dsw-alias-bg-secondary)'
      },
      panelTitle: { fontSize: '12px', lineHeight: '18px', fontWeight: 600, color: 'var(--dsw-alias-label-primary)' },
      item: { display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' as const },
      bar: {
        position: 'relative' as const,
        height: '4px',
        borderRadius: '2px',
        background: 'var(--dsw-alias-bg-tertiary, rgba(127,127,127,0.25))',
        overflow: 'hidden',
        width: '100%'
      },
      barFill: {
        position: 'absolute' as const,
        left: 0,
        top: 0,
        bottom: 0,
        background: 'var(--dsw-alias-brand-primary)',
        borderRadius: '2px'
      },
      log: {
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
        fontSize: '11px',
        lineHeight: '15px',
        maxHeight: '160px',
        overflow: 'auto',
        whiteSpace: 'pre-wrap' as const,
        color: 'var(--dsw-alias-label-secondary)',
        margin: 0
      },
      mono: { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace', fontSize: '11px' }
    }

    /** A labelled progress bar for one running acquisition. */
    function JobBar(props: { job: Job }) {
      const job = props.job
      const ratio = job.totalBytes && job.totalBytes > 0 ? Math.min(1, job.doneBytes / job.totalBytes) : null
      const percent = ratio === null ? null : Math.round(ratio * 100)
      return h(
        'div',
        { style: { display: 'flex', flexDirection: 'column', gap: '4px' } },
        h(
          'div',
          { style: styles.row },
          h('span', { style: styles.strong }, job.label),
          h(
            'span',
            { style: styles.hint },
            `${fmtBytes(job.doneBytes)}${job.totalBytes ? ` / ${fmtBytes(job.totalBytes)}` : ''}${
              percent === null ? '' : ` · ${percent}%`
            }${job.bytesPerSecond ? ` · ${fmtBytes(job.bytesPerSecond)}${copy.rate}` : ''}`
          ),
          job.detail ? h('span', { style: styles.hint }, job.detail) : null
        ),
        h('div', { style: styles.bar }, h('div', { style: Object.assign({}, styles.barFill, { width: `${percent ?? 8}%` }) }))
      )
    }

    /**
     * The provider management card.
     *
     * @param props - the slot's standard props, including the directory entry
     *   this card is keyed by.
     */
    function MiniCPMCard(props: any) {
      const [status, setStatus] = React.useState(null)
      const [busy, setBusy] = React.useState(false)
      const [error, setError] = React.useState(null)
      const [showLog, setShowLog] = React.useState(false)

      const refresh = React.useCallback(async () => {
        try {
          const next: Status = await call('/status')
          setStatus(next)
          setError(null)
        } catch (cause: any) {
          setError(cause.message)
        }
      }, [])

      React.useEffect(() => {
        let alive = true
        ;(async () => {
          try {
            const next: Status = await call('/status')
            if (alive) {
              setStatus(next)
              setError(null)
            }
          } catch (cause: any) {
            if (alive) setError(cause.message)
          }
        })()
        return () => {
          alive = false
        }
      }, [])

      // Poll faster while a download or a start is moving, slowly otherwise.
      const moving =
        status != null &&
        (status.engine.starting ||
          status.jobs.some(job => job.status === 'running') ||
          status.engine.running !== true)
      React.useEffect(() => {
        const timer = setInterval(refresh, moving ? POLL_BUSY_MS : POLL_IDLE_MS)
        return () => clearInterval(timer)
      }, [moving, refresh])

      const run = React.useCallback(
        async (action: () => Promise<any>) => {
          setBusy(true)
          setError(null)
          try {
            await action()
            await refresh()
          } catch (cause: any) {
            setError(cause.message)
          } finally {
            setBusy(false)
          }
        },
        [refresh]
      )

      if (props != null && props.provider != null && props.provider.provider !== PROVIDER) return null

      const engine = status?.engine
      const gpu = status?.gpu
      const runningJobs = (status?.jobs || []).filter(job => job.status === 'running')
      const finishedJobs = (status?.jobs || []).filter(job => job.status !== 'running').slice(0, 3)
      const activeJob = runningJobs[0] || null

      const stateText =
        engine == null
          ? '…'
          : engine.mode === 'external'
            ? `${copy.external} · ${engine.baseURL}`
            : engine.starting
              ? copy.starting
              : engine.running
                ? copy.running
                : copy.stopped

      const dotColor =
        engine == null
          ? 'var(--dsw-alias-label-secondary)'
          : engine.running
            ? 'var(--dsw-alias-state-success-primary)'
            : engine.starting
              ? 'var(--dsw-alias-state-warn-primary)'
              : engine.lastError
                ? 'var(--dsw-alias-state-error-primary)'
                : 'var(--dsw-alias-label-secondary)'

      const head = h(
        'div',
        { style: styles.row },
        h('span', { key: 'dot', style: Object.assign({}, styles.dot, { background: dotColor }) }),
        h('span', { key: 'state', style: styles.strong }, stateText),
        engine?.running && engine.pid
          ? h('span', { key: 'pid', style: styles.hint }, `PID ${engine.pid}`)
          : null,
        engine?.running && engine.loadedModel
          ? h('span', { key: 'file', style: styles.hint }, engine.loadedModel.split('/').pop())
          : null,
        gpu && gpu.totalMiB
          ? h(
              'span',
              { key: 'vram', style: styles.hint },
              `${copy.vram} ${fmtVram(gpu.usedMiB)} / ${fmtVram(gpu.totalMiB)}`
            )
          : null,
        engine?.running
          ? h(
              'span',
              { key: 'idle', style: styles.hint },
              engine.idleSecondsLeft === null
                ? copy.idleOff
                : `${copy.idleIn} ${engine.idleSecondsLeft}s`
            )
          : null
      )

      const controls = h(
        'div',
        { style: styles.row },
        engine?.running
          ? h(
              Button,
              {
                key: 'stop',
                variant: 'outline',
                size: 'sm',
                disabled: busy || engine.mode === 'external',
                onClick: () => run(() => call('/engine/stop', { method: 'POST' }))
              },
              copy.stop
            )
          : h(
              Button,
              {
                key: 'start',
                variant: 'primary',
                size: 'sm',
                disabled: busy || engine?.starting === true,
                onClick: () => run(() => call('/engine/start', { method: 'POST', body: '{}' }))
              },
              copy.start
            ),
        h(
          Button,
          { key: 'refresh', variant: 'outline', size: 'sm', disabled: busy, onClick: () => run(async () => {}) },
          copy.refresh
        ),
        engine?.running
          ? h(
              Button,
              { key: 'release', variant: 'outline', size: 'sm', disabled: busy, onClick: () => run(() => call('/engine/stop', { method: 'POST' })) },
              copy.release
            )
          : null
      )

      // ---------------------------------------------------------------------
      // Weights
      // ---------------------------------------------------------------------
      const weightRows = (status?.models || []).map(row =>
        h(
          'div',
          { key: row.id, style: styles.item },
          h('span', { style: styles.strong }, row.name),
          h(
            'span',
            { style: styles.hint },
            row.present
              ? `${copy.present} · ${fmtBytes(row.sizeBytes)}`
              : `${copy.absent} · ${fmtBytes(row.expectedBytes)}`
          ),
          row.id === status?.activeModel ? h('span', { style: styles.hint }, '· 默认') : null,
          !row.present
            ? h(
                Button,
                {
                  key: 'dl',
                  variant: 'outline',
                  size: 'sm',
                  disabled: busy || activeJob != null,
                  onClick: () => run(() => call('/download/model', { method: 'POST', body: JSON.stringify({ id: row.id }) }))
                },
                copy.download
              )
            : null
        )
      )

      const runtimeRow = h(
        'div',
        { style: styles.item },
        h('span', { style: styles.strong }, copy.runtime),
        h(
          'span',
          { style: styles.hint },
          engine?.binaryPresent ? `${copy.runtimeReady} · ${engine.binaryPath}` : copy.runtimeMissing
        ),
        !engine?.binaryPresent
          ? h(
              Button,
              {
                key: 'dl',
                variant: 'primary',
                size: 'sm',
                disabled: busy || activeJob != null,
                onClick: () => run(() => call('/download/engine', { method: 'POST' }))
              },
              copy.downloadRuntime
            )
          : null
      )

      const progress = h(
        'div',
        { style: styles.panel },
        h('div', { style: styles.panelTitle }, copy.weights),
        h('div', { style: { display: 'flex', flexDirection: 'column', gap: '6px' } }, ...weightRows),
        h('div', { style: styles.panelTitle }, copy.runtime),
        runtimeRow
      )

      const jobsPanel =
        runningJobs.length > 0 || finishedJobs.length > 0
          ? h(
              'div',
              { style: styles.panel },
              h('div', { style: styles.panelTitle }, copy.downloading),
              ...runningJobs.map(job =>
                h(
                  'div',
                  { key: job.id, style: { display: 'flex', flexDirection: 'column', gap: '4px' } },
                  h(JobBar, { job }),
                  h(
                    'div',
                    { style: styles.row },
                    h(
                      Button,
                      {
                        variant: 'outline',
                        size: 'sm',
                        onClick: () => run(() => call('/jobs/cancel', { method: 'POST', body: JSON.stringify({ id: job.id }) }))
                      },
                      copy.cancel
                    )
                  )
                )
              ),
              ...finishedJobs.map(job =>
                h(
                  'div',
                  { key: job.id, style: styles.row },
                  h('span', { style: styles.label }, job.label),
                  h(
                    'span',
                    {
                      style:
                        job.status === 'error'
                          ? styles.error
                          : styles.hint
                    },
                    job.status === 'done'
                      ? copy.done
                      : job.status === 'cancelled'
                        ? copy.cancelled
                        : `${copy.failed}: ${job.error || ''}`
                  )
                )
              )
            )
          : null

      const notice = engine?.lastError || error

      const logPanel =
        engine && engine.log && engine.log.length > 0
          ? h(
              'div',
              { style: styles.panel },
              h(
                'div',
                { style: styles.row },
                h('div', { style: styles.panelTitle }, copy.log),
                h(
                  Button,
                  { variant: 'outline', size: 'sm', onClick: () => setShowLog(value => !value) },
                  showLog ? '收起' : '展开'
                )
              ),
              showLog ? h('pre', { style: styles.log }, engine.log.slice(-60).join('\n')) : null
            )
          : null

      return h(
        'div',
        { style: styles.wrap },
        head,
        controls,
        notice ? h('p', { style: styles.error }, String(notice)) : null,
        progress,
        jobsPanel,
        logPanel
      )
    }

    /**
     * The card's key must equal the provider row's `settingsNs`, which is the
     * profile **entry id** (`minicpm`) — the settings page dispatches this slot
     * with `{ entryKey: row.entry.settingsNs }`, so any other value renders
     * nothing at all.
     */
    const inject = ['slots', 'configForms']

    function apply(ctx: any) {
      ctx.slots.inject('settings.models.provider-card', () =>
        ctx.slots.register(
          {
            name: 'settings.models.provider-card',
            key: 'minicpm'
          },
          MiniCPMCard
        )
      )
      mountConfigCard(ctx)
    }

    /**
     * The Plugins-page configuration card for this package.
     *
     * DSH 0.1.7 replaced the old `settingsScope` service with `configForms` (the
     * browser-side controller for the active profile's plugin configuration). A
     * form renders only for a package whose profile entry carries a `config`
     * block, which is what the page's `ledger.bundles` collects.
     *
     * Only **flat scalars** can be edited: `SettingsFormModel` reads
     * `value?.[field]` and writes `path: [field]`, so a nested name like
     * `engine.port` would address a literal top-level key of that name. That is
     * why the Host schema exposes `enginePort` / `engineHost` / … alongside the
     * nested `engine` object.
     */
    function mountConfigCard(ctx: any) {
      const forms = ctx && ctx.configForms
      if (forms === undefined || forms === null || typeof forms.get !== 'function') return
      const scope = forms.get(CONFIG_NS)
      if (scope === undefined || scope === null) return

      /** A boolean rendered as a two-state text field (no boolean helper ships). */
      const booleanField = (field: string) => ({
        field,
        format: (value: any) => (value === true ? 'on' : 'off'),
        parse: (text: string) => ({ kind: 'set', value: String(text).trim() === 'on' })
      })
      /** A text field that keeps an empty draft as an explicit clear. */
      const textField = (field: string) => ({
        field,
        format: (value: any) => (typeof value === 'string' ? value : ''),
        parse: (text: string) => {
          const trimmed = String(text).trim()
          return trimmed === '' ? { kind: 'clear' } : { kind: 'set', value: trimmed }
        }
      })
      /** A whole-number field; an empty draft clears it. */
      const numberField = (field: string) => ({
        field,
        format: (value: any) => (typeof value === 'number' ? String(value) : ''),
        parse: (text: string) => {
          const trimmed = String(text).trim()
          if (trimmed === '') return { kind: 'clear' }
          const parsed = Number(trimmed)
          return Number.isFinite(parsed) ? { kind: 'set', value: parsed } : undefined
        }
      })
      /** A two-choice enum field, travelling as text. */
      const choiceField = (field: string, allowed: string[], fallback: string) => ({
        field,
        format: (value: any) => (allowed.indexOf(value) >= 0 ? value : ''),
        parse: (text: string) => {
          const trimmed = String(text).trim()
          if (trimmed === '') return { kind: 'clear' }
          return { kind: 'set', value: allowed.indexOf(trimmed) >= 0 ? trimmed : fallback }
        }
      })

      const ENGINE_FIELDS: Array<[string, string, string]> = [
        ['engineMode', '引擎模式', 'managed 由本插件托管 llama.cpp 进程；external 连接一个你自己起的端口。'],
        ['engineHost', '监听地址', '托管模式下引擎绑定的网卡地址。'],
        ['enginePort', '端口', '托管模式下引擎监听的端口，默认 8081。'],
        ['engineBinary', 'llama.cpp 可执行文件', '留空则用自动下载/检测到的那个。'],
        ['engineBaseURL', '外部引擎地址', 'external 模式下要连的基址，例如 http://127.0.0.1:8081。'],
        ['engineContextSize', '上下文长度', '每次请求分配的上下文窗口，默认 32768。'],
        ['engineGpuLayers', 'GPU 层数', '卸载到 GPU 的层数；999 表示尽量全部。'],
        ['engineThreads', '线程数', 'CPU 线程数；0 表示交给引擎自己决定。'],
        ['engineIdleUnloadSeconds', '空闲卸载秒数', '闲置这么久后释放引擎，0 表示一直保留。'],
        ['fetchHfEndpoint', '模型下载源', 'Hugging Face 基址；换成镜像可以避开公共 CDN。'],
        ['fetchBuild', 'llama.cpp 版本', '固定一个构建号，例如 b10951；留空表示取最新。']
      ]

      const formModel = new (primitives.SettingsFormModel)(scope, [
        choiceField('engineMode', ['managed', 'external'], 'managed'),
        textField('engineHost'),
        numberField('enginePort'),
        textField('engineBinary'),
        textField('engineBaseURL'),
        numberField('engineContextSize'),
        numberField('engineGpuLayers'),
        numberField('engineThreads'),
        numberField('engineIdleUnloadSeconds'),
        booleanField('enginePreserveReasoning'),
        textField('fetchHfEndpoint'),
        textField('fetchBuild')
      ])
      ctx.effect(() => () => formModel.dispose())

      const formLabels = {
        unavailable: '本部署没有提供这项配置。',
        readOnly: '本部署的配置为只读，无法在此修改。',
        save: '保存',
        saving: '保存中…',
        saveFailed: '保存失败，改动未生效。'
      }

      const formStore = formModel.bind(() => {
        const projection: any = { shell: formModel.shell() }
        for (const [field] of ENGINE_FIELDS) projection[field] = formModel.field(field)
        projection.enginePreserveReasoning = formModel.field('enginePreserveReasoning')
        return projection
      })
      ctx.effect(() => () => formStore.dispose())

      function MiniCPMConfigCard(props: any) {
        const state = props.useMinicpmConfig((snapshot: any) => snapshot)
        const children = ENGINE_FIELDS.map(([field, label, hint]) => {
          const node = state[field]
          return h(primitives.SettingsValueField, {
            key: field,
            id: field,
            label,
            hint,
            text: node.text,
            overridden: node.overridden,
            invalid: node.invalid,
            numeric: field.indexOf('engine') === 0 && field !== 'engineMode' && field !== 'engineHost'
              && field !== 'engineBinary' && field !== 'engineBaseURL',
            onEdit: (text: string) => props.edit(field, text),
            onReset: () => props.resetField(field)
          })
        })
        const preserve = state.enginePreserveReasoning
        children.push(
          h(primitives.SettingsValueField, {
            key: 'enginePreserveReasoning',
            id: 'enginePreserveReasoning',
            label: '保留思考过程',
            hint: 'on 时把模型的 reasoning 一起留在回复里。',
            text: preserve.text,
            overridden: preserve.overridden,
            invalid: preserve.invalid,
            onEdit: (text: string) => props.edit('enginePreserveReasoning', text),
            onReset: () => props.resetField('enginePreserveReasoning')
          })
        )
        return h(
          primitives.SettingsForm,
          { labels: formLabels, state: state.shell, onSave: props.save, onDiscard: props.discard },
          children
        )
      }

      ctx.slots.inject('plugins.bundle.config', () =>
        ctx.slots.register(
          {
            name: 'plugins.bundle.config',
            key: BUNDLE_KEY,
            inject: () => Object.assign({ hooks: { minicpmConfig: formStore } }, formModel.actions())
          },
          MiniCPMConfigCard
        )
      )
    }

    exports.apply = apply
    exports.inject = inject
    exports.MiniCPMCard = MiniCPMCard
    return module.exports
  }
})
