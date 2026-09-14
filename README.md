# dsh-minicpm

DeepSeek Harness (DSH) 的 **MiniCPM5-2B 本地推理**插件：原生 `LlmAdapter` + `llama.cpp` 引擎生命周期托管 + 设置页管理卡片。

把 4-bit 量化的 MiniCPM5-2B 变成一个和 `deepseek-official`、`google-antigravity` 平级的 DSH 提供商路由 `minicpm-local`：模型选择器里直接可选，工具调用、流式输出、思考过程、词元统计全部走 DSH 自己的通道，不经过 `llm-pi-ai`，也不需要任何账户或联网（除了首次下载权重）。

---

## 它解决了什么

| 直接跑 `llama-server` 的痛点 | 本插件的做法 |
| --- | --- |
| 要手动拉起进程、记住端口、退出时忘了关，显存一直被占 | 插件按需拉起，空闲 15 分钟自动释放显存，DSH 退出时必定回收子进程 |
| 端口、上下文长度、GPU 层数散落在各人的启动脚本里 | 全部是 `llm-minicpm.engine.*` 设置项，改完自动重启引擎 |
| 模型下载要自己找仓库、认文件名、放到对的目录 | 设置页一个按钮，断点续传，实时进度，可取消 |
| `const`、`$ref` 之类的 JSON Schema 关键字会让端点整包 400 | 工具 schema 在构造请求时归一化，缺 `properties` 也能用 |
| 空回答被当成正常结束，agent loop 提交一条空消息 | 零内容的 `stop` 转成可重试的 `EMPTY_RESPONSE` 失败 |
| 换模型要手动重启服务 | 请求的模型 id 对应不同权重文件时，引擎自动重载 |

---

## 架构

```
src/
├── index.ts      Host 插件：settings 命名空间 llm-minicpm、原生适配器注册、
│                 目录条目、回环路由（状态 / 启停 / 下载 / 取消）
├── adapter.ts    原生 LlmAdapter：为请求的模型 id 拿到一个服务中的端点，再走 OpenAI 协议
├── wire.ts       DSH 消息/工具 ↔ OpenAI 兼容线的双向翻译（纯函数）
├── engine.ts     llama-server 子进程生命周期：发现、启动、健康探测、重载、空闲卸载、显存释放
├── fetch.ts      权重与运行时的下载，带进度、断点续传、取消
├── archive.ts    自带的最小 tar.gz 解包器（无第三方依赖，拒绝路径穿越）
├── models.ts     模型目录：DSH 模型 id ↔ GGUF 文件
├── paths.ts      ~/.dsh/minicpm 目录布局
├── client.ts     浏览器半：设置 → 模型 → MiniCPM 卡片（状态、启停、下载、日志）
└── bin.ts        独立 CLI：status / start / chat / doctor / fetch-model / fetch-engine
```

浏览器半是**原样下发、不做转译**的经典脚本，所以单独用 `tsconfig.client.json`
编译，并显式声明 `moduleDetection: "legacy"`（否则 tsc 会追加 `export {}`，
在浏览器里直接是语法错误）。

---

## 安装

插件是一个 DSH profile bundle。装进某个 profile：

```bash
# 本地开发（软链接，改完 src 重新 build 即可）
dsh plugin --profile web add link:/home/vesita/coding/my/dsh-minicpm

# 或打包安装
cd /home/vesita/coding/my/dsh-minicpm && pnpm pack
dsh plugin --profile web add file:/path/to/dsh-minicpm-0.1.0.tgz
```

`dsh plugin add` 会把包加进 profile 的 `dependencies`。**还需要把它加进 bundle 列表**，
否则包在 `node_modules` 里但不会被装载 —— 编辑 `~/.dsh/profiles/<name>/package.json`：

```json
{
  "dsh": {
    "profile": {
      "bundles": [
        "@deepseek-ai/dsh-base",
        "@deepseek-ai/dsh-web-app",
        "dsh-minicpm"
      ]
    }
  }
}
```

改了 bundle 列表需要**重启 DSH**（`patchReload: live` 只覆盖 `cordis.patch.yml` 的热改）。

### 准备运行时与权重

首次使用需要两样东西，插件可以自己下载：

```bash
# 装进 profile 后，CLI 在 profile 的 node_modules/.bin 下
D=~/.dsh/profiles/web/node_modules/.bin/dsh-minicpm

$D fetch-engine     # llama.cpp 运行时（Linux x64 默认取 Vulkan 版，约 20 MB）
$D fetch-model      # MiniCPM5-2B Q4_K_M（约 1.6 GB）
$D doctor           # 一次性检查 GPU / 运行时 / 权重 / 端口
```

也可以全部在 **设置 → 模型 → MiniCPM (本地)** 卡片里点按钮完成。

---

## 设置项（`~/.dsh/settings.yaml` → `llm-minicpm`）

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `models` | 内置 3 个量化档 | 目录项：`id`（DSH 模型 id）、`name`、`description`、`file`（models 目录内的文件名或绝对路径） |
| `engine.mode` | `managed` | `managed` 由插件托管子进程；`external` 只连 `baseURL`，绝不启停 |
| `engine.baseURL` | 未设置 | `external` 模式的端点 |
| `engine.binary` | 自动发现 | 显式指定 `llama-server` 路径；留空则依次找 插件 engine 目录 → `PATH` |
| `engine.host` / `engine.port` | `127.0.0.1` / `8081` | 监听地址与端口 |
| `engine.contextSize` | `32768` | `--ctx-size`。**DSH 看到的上下文窗口就是它**，不是模型的 128k 架构上限 |
| `engine.gpuLayers` | `999` | `--n-gpu-layers`，`999` = 全部卸载到 GPU |
| `engine.threads` | `0` | `0` 让引擎自己决定 |
| `engine.idleUnloadSeconds` | `900` | 空闲多久后停止引擎、释放显存；`0` = 常驻 |
| `engine.extraArgs` | `[]` | 原样追加到命令行的额外参数 |
| `engine.preserveReasoning` | `true` | 是否跨轮保留思考轨迹 |
| `fetch.hfEndpoint` | `https://huggingface.co` | Hugging Face 镜像，大文件走镜像更稳 |
| `fetch.build` | 未设置 | 固定 llama.cpp 版本号（如 `b10951`），留空则自动选最新带产物的版本 |
| `retryPolicy` | 未设置 | 透传给 `dsh-llm-retry`；留空用 harness 默认 |

选为默认模型：

```yaml
agent-default-model:
  provider: minicpm-local
  model: minicpm5-2b-q4
```

> 注意：**用户层 `settings.yaml` 覆盖组合层**。如果你的 `settings.yaml` 里已有
> `agent-default-model` 指向别的路由，必须在用户层改，只改组合层不会生效。

### 环境变量

| 变量 | 作用 |
| --- | --- |
| `DSH_MINICPM_HOME` | 覆盖 `~/.dsh/minicpm`（运行时 + 权重 + 状态的根目录） |
| `HF_ENDPOINT` | 覆盖 Hugging Face 镜像（权重下载） |
| `DSH_MINICPM_MODE` / `_PORT` / `_CTX` / `_GPU_LAYERS` / `_IDLE_UNLOAD` / `_BINARY` / `_BASE_URL` | **仅 CLI** 读取；插件本身只认设置项 |

---

## CLI

```bash
dsh-minicpm status          # 引擎状态、端点、pid、已载模型、显存
dsh-minicpm start           # 拉起引擎加载默认权重（随本命令退出而结束）
dsh-minicpm stop            # 说明：CLI 不维护守护进程
dsh-minicpm models          # 列出本地 GGUF 及其模型 id
dsh-minicpm fetch-model [id]  # 下载权重，实时进度，断点续传
dsh-minicpm fetch-engine    # 下载并解包 llama.cpp 运行时
dsh-minicpm chat "你好"     # 直接打一次请求，验证链路
dsh-minicpm doctor          # 环境体检
```

---

## 为什么是 llama.cpp + Vulkan

- **不需要 CUDA toolkit。** 本机有驱动（`nvidia-smi` 报告 CUDA 13.4）但没有 `nvcc`，
  也没有 `cmake`。llama.cpp 的 Linux 预编译产物只提供 Vulkan / ROCm / CPU，
  Vulkan 版直接吃 NVIDIA 显卡，`--list-devices` 能认到 RTX 5060。
- **纯 C++，零 Python 环境污染。** 不需要 venv、不需要 `torch`，冷启动秒级。
- **OpenAI 兼容 + `--jinja`。** 模型的工具调用与思考模板由 Jinja chat template 提供，
  实测 `finish_reason: "tool_calls"` 与 `reasoning_content` 都正常。

如果你更想从源码编译 CUDA 版（装上 `cuda` + `cmake` 后），把编出来的
`llama-server` 路径写进 `engine.binary` 即可，插件其余部分不用改。

---

## 已验证的行为

单元测试（`npm test`，33 项，无网络、无 GPU）：

- **请求构造**：system 消息上提合并、tool-result 拆成 `role: "tool"` 并按 call id 关联、
  assistant 工具调用回投影、reasoning 块不回放、图片降级为文字占位、
  tools 参数 schema 归一化、空会话兜底、生成参数按需出现；
- **流解析**：块起止与索引、reasoning/text 分块、跨帧分片的工具调用参数拼接、
  `tool_calls` 优先于 `length`、零内容 `stop` → `EMPTY_RESPONSE`、
  usage 在 finish 之前、半帧拼回、坏帧跳过、id 工厂注入；
- **用量口径**：`prompt_tokens` 里的缓存命中被减出去，得到互不重叠的桶；
- **模型解析**：目录 id / 裸文件名 / 未知 id / 绝对路径四种输入；
- **引擎参数**：`--jinja --no-ui -ngl -c` 恒存在，可选参数按需出现；
- **引擎生命周期**（用一个假的 `llama-server` 存根，受控的「延迟 N 毫秒后才开始应答」）：
  已载权重不重复起进程、**并发调用等启动中的服务而不是把它杀掉**（见下）、
  换权重文件时重载、运行时缺失与权重缺失各自的报错码；
- **归档**：真实 tarball 逐字节比对（含 100 字节以上的长文件名与可执行位），
  以及路径穿越被拒绝。

### 端到端（真实引擎 + 真实权重 + 真实 DSH profile）

方法：在隔离的 `DSH_HOME`（`/tmp` 下一个全新 home，没有 `settings.yaml`）里跑 headless
profile，由 `--patch` 覆盖层把 `agent-default-model` 指向 `minicpm-local/minicpm5-2b-q4`。

1. **反向对照**：把 `llama-server` 藏起来后，会话以插件自己的可执行错误结束 ——
   `未找到 llama-server 运行时（已查找 …/minicpm/engine 与 PATH）。请在设置页点击「下载推理引擎」…`，
   且**不产出任何回答**。这一步是必须的：最初几次「成功」其实落在用户层
   `settings.yaml` 覆盖的远端模型上，只有反向对照才暴露了这一点。
2. 恢复运行时后，同一会话正常回答。
3. **工具调用闭环**：模型自主调用 `bash` 执行 `echo FINAL_VERIFICATION_OK`、读回结果、
   只回答命令输出；思考过程经 `reasoning_content` → `reasoning-delta` 正常回显。
4. 会话结束后无 `llama-server` 孤儿进程，显存回到基线。

本机实测：RTX 5060 (Vulkan) 生成约 **128 tok/s**，4-bit 权重加载约 **2 秒**。

### 过程中修掉的两个真实缺陷

这两个都是端到端跑出来的，不是推测：

1. **并发 `ensure()` 会杀掉正在启动的引擎。** 第二次 `ensure()` 在服务还在加载时探测
   `/health`，得到 503 就断定「进程无响应」，于是停掉了第一次调用正在等待的那个子进程。
   一次 agent turn 同时发出标题请求和主请求就足以触发。现在并发调用改为等待在途启动。
   `test/test.js` 里有对应的回归测试，并且验证过：**把修复回退后该测试会失败**。
2. **只看 `exitCode` 判不出信号导致的退出。** 进程被信号杀死时 `exitCode` 保持 `null`，
   于是一个早已死掉的子进程会让就绪轮询空等满 180 秒，最后报「未就绪」而不是真正的死因。
   现在就绪等待与子进程退出直接竞速，并报告信号、探测次数与最后一次探测失败的原因。


---

## 开发

```bash
pnpm install
pnpm run build        # tsc → lib/（Host 与浏览器半两份 tsconfig）
pnpm test             # 构建后跑 28 项单元测试
pnpm run test:types   # 仅类型检查
```

> 改了 `src/` 必须 `pnpm run build`：DSH 装载的是 `lib/`，不是 `src/`。
> profile 里用 `link:` 安装时，重新 build 后重启 DSH 即可生效。

## License

MIT
