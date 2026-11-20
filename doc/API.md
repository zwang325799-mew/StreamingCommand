# StreamingCommand 功能与 API

从 `streamingcommand` 导入 `StreamingCommand`、`StreamingASR`、`StreamingConfig`、`ASRConfig`、`CommandConfig`、`CandidateProfiles`、`audio_file` 和 `microphone`。本页也列出会话对象的可调用方法、资源管理函数和命令行入口。安装后的文档位于 Python 环境的 `share/doc/streamingcommand/API.md`。

## 用一份 JSON 切换场景

随包提供的 [settings.json](../src/streamingcommand/assets/settings.json) 使用内窥镜候选和默认模型。安装后可从 `importlib.resources.files("streamingcommand").joinpath("assets/settings.json")` 读取同一模板，复制到自己的目录后再编辑。

| 完整范例 | ASR | 命令模型 |
| --- | --- | --- |
| [local-latest.json](../src/streamingcommand/assets/presets/local-latest.json) | 本地 `v17-yue-precision`，JSON 列出主模型、参考基座、语种判别和 VAD 的实际目录。 | 本地 Qwen 基座 + `clarification-hybrid`（epoch 2 / step1392），启用 `semantic-pair`。迁移到其他机器时按实际位置改六个 `path`。 |
| [huggingface-release.json](../src/streamingcommand/assets/presets/huggingface-release.json) | `humoiww/PhonoCommand-ASR@ff8daaaf`，使用 `v17-yue-precision`。 | `Qwen/Qwen3.5-0.8B@2fc06364` + `humoiww/PhonoCommand-LLM-0.8B@d5217d0c` 子目录 `clarification-hybrid`（epoch 2 / step1392）。预设 I/T/P 采用 `endoscope-hybrid`。 |

表中提交号为缩写。JSON 给出完整 URL 和 40 位提交号。V17 使用同一套经哈希核验的 ASR 权重，粤语策略由库在推理时应用。 [Hugging Face 发布记录](https://huggingface.co/humoiww/PhonoCommand-ASR/blob/ff8daaafd4a7fd4b396c2ffcab8da230d3c17a4f/releases/20261120-hybrid/ASR_PROFILE.json)

### 指定模型来源

`asr.profile` 选择语音处理策略，`command.model` 选择命令解码器及同名权重项。`models` 可逐项指定 `asr`（语音识别主模型）、`asr-base`（参考基座）、`language`（语种模型）、`speech`（VAD）、`llm-base`（命令基座）和命令模型 ID。各项可以混用不同来源。省略的项使用包内默认来源。

| `source` | 必填字段与示例 |
| --- | --- |
| `huggingface` | `"url": "https://huggingface.co/humoiww/PhonoCommand-LLM-0.8B"`、`"revision": "d5217d0c486933822da027600d7ceb9ed2d9bfb7"`、`"subfolder": "clarification-hybrid"`。命令模型还可填 `"expected_step": 1392`。 |
| `local` | `"path": "./models/my-checkpoint"`。填写实际模型目录。相对路径以配置文件所在目录为起点。命令模型还可写 `"expected_step": 936`，启动时检查 `experiment.json` 的步数，用错目录会报错。 |
| `package` | `path` 为 `assets/models/clarification-new` 或 `assets/models/clarification-latest`，对应随包分发的两个澄清模型。 |

更换仓库或版本时保持对应模型的权重结构。旧候选排序器 `phonocommand-llm-0.8b` 在目录根部读取 PEFT 适配器、`head.safetensors` 和 `experiment.json`。澄清模型从 `adapter/` 读取 PEFT 权重，根部还需 `head.safetensors`、`clarify_head.pt` 和 `experiment.json`。同一资源的 `models` 条目和旧式本地路径字段只能填写一处。

### 更换候选

下面是一份可以直接保存为 `settings.json` 的器械护士配置。它选用 step936，只允许递交、停止和复位。这个场景没有方向，模型的方向输出为 JSON `null`。

```json
{
  "command": {
    "model": "clarification-latest",
    "candidates": {
      "I": ["give", "stop", "reset"],
      "T": ["grasper", "gauze", "needle holder"],
      "P": []
    }
  }
}
```

```python
import asyncio
from streamingcommand import StreamingCommand

async def main():
    async with StreamingCommand(config_path="settings.json") as sdk:
        info = sdk.info()["command"]
        print(info["model_id"], info["checkpoint_step"], info["default_candidates"])
        result = await sdk.parse("Please give me the grasper.")
        print(result["status"], result["slots"])

asyncio.run(main())
```

- `command.model` 可选 `clarification-new`（step3003，默认）、`clarification-latest`（step936）、`clarification-hybrid`（step1392）或 `phonocommand-llm-0.8b`（旧候选排序器）。三个澄清模型支持 H1、空 P 的器械护士配置和 `semantic-pair`。内窥镜新命令表使用 `clarification-hybrid` 和 `endoscope-hybrid.json`，新增 `zoom in`、`zoom out`、`spray water`、`spray air`。
- `command.candidates` 直接保存三组有序字符串。I 是意图，T 是目标，P 是方向或位置。每组内部的值须非空且唯一，I 和 T 至少各有一项。P 有值时，T 和 P 都须包含业务值 `"none"`。P 为空时，I 只支持 `give`、`stop`、`reset` 的非空子集。自转用 `self rotate`。模型从提供的候选中预测，新增名称的识别效果需要用自己的输入验证。
- 也可以用 `"candidates_file": "profiles/scrub.json"` 引用单独一个只含 I、T、P 的 JSON。它与内嵌 `candidates` 二选一。两者都省略时，使用所选模型的默认候选文件。显式传给 `parse` 或 `session` 的候选只覆盖本次调用或会话。
- JSON 中省略的字段使用配置类默认值。配置里的相对文件路径以该 JSON 所在目录为基准；构造器接受的配置文件路径本身相对于当前工作目录。

## 识别器

| Python 调用 | 用途与返回值 |
| --- | --- |
| `StreamingCommand(config=None, *, config_path=None, asr_backend=None, command_backend=None)` | 创建待启动实例。可以传 `StreamingConfig`、位置参数 JSON 路径，或关键字 `config_path`。同时提供 `config` 与 `config_path` 会报错。自定义后端可经关键字注入；注入 ASR 时也由调用方提供所需命令后端。 |
| `StreamingASR(config=None, **kwargs)` | 只做转写，接受相同的配置路径和 ASR 后端参数，常规启动时只加载 ASR。 |
| `await StreamingCommand.initialize(config_path=None, *, config=None)`、`await StreamingASR.initialize(config_path=None, *, config=None)` | 创建、启动并返回就绪实例。可传 JSON 路径或 `StreamingConfig` 对象，省略两者时使用包默认值。调用方负责 `await sdk.close()`。 |
| `await sdk.start()` | 在已有实例上启动模型并返回自身。已就绪时可重复调用。 |
| `async with StreamingCommand(...) as sdk` | 进入时启动，退出时关闭。也适用于 `StreamingASR`。 |
| `sdk.info()` | 返回就绪状态和后端状态。实际 ASR 的 `model_sources` 给出来源和解析目录。命令后端的 `model_id`、`base_model_source`、`checkpoint_source`、`checkpoint` 和 `default_candidates` 给出所用模型与候选。 |
| `await sdk.parse(text, *, candidates=None, history=None)` | 直接解析一条非空文本，返回含 `status`、`slots`、`resolved`、`execute` 等字段的字典。需要启用命令模型；音频会话占用实例时使用该会话的澄清接口。独立调用的历史由调用方传入。 |
| `sdk.listen(*, device=None, candidates=None, history=None)` | 用 `async for` 接收麦克风识别事件。`device` 为录音设备。 |
| `sdk.transcribe_file(path, *, realtime=True, candidates=None, history=None)` | 用 `async for` 接收文件识别事件。默认按音频时长回放。 |
| `sdk.stream(source, *, candidates=None, history=None)` | 接受异步可迭代的 PCM 字节源，返回事件异步迭代器。提供含静音的 16 kHz 单声道、小端 PCM16。 |
| `sdk.session(*, candidates=None, history=None, vad=None, interval_ms=None, endpoint_ms=None, max_segment_ms=None, recognition_logic=None)` | 返回待启动的 `CommandStream`。可注入实现 `is_speech(frame, rate)` 的 VAD。未提供的会话选项沿用实例配置。一个实例同时支持一个音频会话。 |
| `await sdk.switch_model(model)` | 在空闲时切换状态所列的可用命令模型，返回命令模型状态。ASR 保持驻留。 |
| `await sdk.close()` | 结束活动会话，释放实例拥有的后端，可重复调用。再次使用时请新建实例。 |

可用 `history={"H1": "I=bend;T=none;P=left"}` 给出初始历史，空历史可省略。一个音频会话会在动作确定后更新 H1。流式输出中 `transcript.partial` 和 `command.partial` 是可修订的预览。`transcript.final` 给出定稿文字，`command.final` 给出模型的决定。检查 `resolved` 和 `execute` 后再消费动作；`command.error` 携带解析失败信息。

## 手动输入 PCM 的会话

通过 `session = sdk.session(...)` 创建，等价的类为 `streamingcommand.pipeline.CommandStream(owner, *, candidates=None, history=None, vad=None)`。用不同的 asyncio 任务发送 PCM 和消费事件，避免有界队列填满。

| 调用 | 用途与返回值 |
| --- | --- |
| `async with session` 或 `await session.start()` | 启动 ASR 会话和可选的命令会话，产生 `session.started`。上下文退出时关闭。 |
| `await session.feed(pcm)` | 加入一块 PCM 字节。奇数字节长度或类型不符会报错，缓冲超限会抛出 `BufferError`。 |
| `session.events()` | 用 `async for` 连续读取事件，支持一个活动消费者。 |
| `session.configure_candidates(candidates)` | 只更新当前会话，返回生效候选和 `candidate_version`，先前待答的澄清失效。 |
| `await session.feedback(text, clarification_id)` | 用自然语言回答当前澄清。ID 取自先前结果，结果通过 `command.feedback` 事件返回。 |
| `await session.clear_memory()` | 清除待答澄清和模型侧临时状态。后续命令仍可使用 H1。 |
| `await session.finish()` | 通知输入结束，等待定稿和命令处理完成，发出 `session.completed` 并结束事件流。消费任务应保持运行。 |
| `await session.fail(exc)` | 将输入源异常转交给事件消费者，消费者会收到该异常。随后关闭会话。 |
| `await session.close()` | 结束会话并释放占用，可用于提前中断。对完整录音先调用 `finish()` 以取回末尾定稿。 |

```python
# sdk 已启动，pcm_source 是异步 PCM 字节源
async with sdk.session() as session:
    async def consume():
        async for event in session.events():
            print(event["type"])

    async def produce():
        async for pcm in pcm_source:
            await session.feed(pcm)
        await session.finish()

    await asyncio.gather(produce(), consume())
```

## 配置类和字段

`StreamingConfig(asr=..., command=..., ...)` 接受 `ASRConfig(...)`、`CommandConfig(...)` 和下表顶层字段。用 `StreamingConfig.load(path)` 读取 UTF-8 JSON，返回可传给识别器的配置对象。构造配置对象时，下表的文件路径可直接传绝对路径。

| JSON 字段 | 默认值及作用 |
| --- | --- |
| `models` | 可按资源配置 `source`、远端链接与版本或本地模型目录。写法见本页“指定模型来源”。 |
| `asr.backend`、`asr.device` | 默认为 `native`、`auto`。原生执行支持 `cpu` 或 `cuda:N`。设置 `CUDA_VISIBLE_DEVICES` 时，`N` 为调用方可见列表中的编号。命令模型采用同一规则。可选 `vllm`、`vllm-builtin`（需要兼容 CUDA 环境），或协议测试用 `fake`。 |
| `asr.profile` | 默认 `v15-canto-commit`，可选 `v17-yue-precision`。单 ASR 的历史配置还有 `v15`、`v15-realtime`、`v15-continuity`、`v15-temporal`、`v15-temporal-stream`、`v15-speech-aware`。 |
| `asr.python`、`asr.expected_gpu_uuid`、`asr.warmup` | 可指定 worker 解释器与预期 GPU 身份，默认均为 `null`。默认打开预热。 |
| `asr.model_dir`、`asr.alternate_model_dir`、`asr.language_model_dir`、`asr.speech_model_dir` | 分别覆盖 ASR 主模型、参考基座、语种模型和语音活动模型目录。默认按固定版本解析。 |
| `asr.compatibility_lock_file` | 可选 POSIX 锁路径。用于和已有部署共用一把额外的 ASR 锁。 |
| `command.enabled`、`command.model`、`command.device` | 默认 `true`、`clarification-new`、`auto`。可选模型见本页开头。 |
| `command.candidates`、`command.candidates_file` | 可选内嵌 I/T/P 或候选 JSON 路径，默认均为 `null`。 |
| `command.candidate_profiles_dir` | 可选候选配置目录。省略时使用包的用户配置目录。 |
| `command.recognition_logic` | 默认 `original`。流式会话可选 `semantic-pair`，比较当前片段和相邻未执行片段拼接后的语义。 |
| `command.python`、`command.base_model_dir`、`command.checkpoint` | 覆盖命令 worker 解释器、语言模型基座、与所选模型兼容的 checkpoint 目录。默认均为 `null`。 |
| `command.use_cuda_graphs`、`command.expected_gpu_uuid` | CUDA 图默认打开；可选预期 GPU 身份。 |
| `interval_ms`、`endpoint_ms`、`max_segment_ms` | 默认 800、240、16000 毫秒，分别设置流式更新间隔、定稿静音阈值和单段最长时长。 |
| `event_queue_size`、`max_pending_audio_ms` | 默认 256 个事件、30000 毫秒音频，限制事件与音频积压。 |
| `cache_dir`、`logs_dir`、`offline` | 默认使用系统用户缓存和日志目录。默认 `offline=false`；设为 `true` 时，缺失的缓存资源会报错。 |

## 音频源与候选配置文件管理

| Python 调用 | 用途与返回值 |
| --- | --- |
| `audio_file(path, *, realtime=True, block_ms=20)` | 异步迭代解码后的单声道 16 kHz PCM16 字节块，自动下混和连续重采样。 |
| `microphone(device=None, *, sample_rate=None, block_ms=20, queue_seconds=5)` | 异步迭代录音字节。默认采用输入设备采样率并转换成 16 kHz；溢出时抛出 `streamingcommand.audio.AudioOverflowError`。 |
| `CandidateProfiles(directory=..., registry=None)` | 打开具名候选配置目录和模型默认映射表，省略参数使用用户配置目录。 |
| `CandidateProfiles.filename(name)` | 校验名称并返回带 `.json` 的文件名。 |
| `profiles.list()` | 返回有效文件、校验错误和默认映射，键为 `configs`、`errors`、`defaults`。 |
| `profiles.read(name)` | 返回包含 `name` 和 `candidates` 的字典。 |
| `profiles.save(name, candidates, *, create=False)` | 更新已有文件，用 `create=True` 新建。另存为可将 `read()` 得到的候选传给新名字。返回保存的记录。 |
| `profiles.rename(name, new_name)` | 重命名并同步模型默认映射，返回新记录。 |
| `profiles.default_path(model, fallback)` | 返回该模型默认候选的 `Path`，缺少映射时使用传入的备用路径。 |

## 资源、集成和 CLI

安装 `streamingcommand[speech]` 后，可用 `streamingcommand.speech.ClarificationSpeech(logs_root=None, *, offline=False)` 朗读澄清提示。用 `await speech.speak(text)` 返回 WAV 字节，可用 `await speech.start()` 预热，用 `await speech.close()` 释放 CPU worker。

| 调用 | 用途与返回值 |
| --- | --- |
| `streamingcommand.resources.ModelStore(cache_dir=None, *, offline=False)` | 创建固定版本的模型资源管理器。 |
| `ModelStore.from_config(config)` | 按 `StreamingConfig` 创建资源管理器，与识别器启动时使用相同的路径、缓存和离线设置。 |
| `store.source(name)` | 返回资源生效的来源字段，无需启动模型或下载文件。 |
| `store.ensure(name, override=None)` | 解析资源并返回本地 `Path`。名称可用五个基础资源名和三个命令模型 ID。显式 `override` 使用调用方的本地目录。包清单内的远端资源还会校验文件哈希。 |
| `streamingcommand.resources.sha256(path)` | 返回文件的十六进制 SHA-256。 |
| `await streamingcommand.backends.build_backends(config)` | 按 `StreamingConfig` 解析资源，返回待启动的 `(asr, command)`；禁用命令时第二项为 `None`。普通调用使用管理生命周期的识别器。 |
| `streamingcommand.cli.main(argv=None)` | Python 命令行入口，可传参数列表。入口脚本为 `streaming-command`。 |

```bash
streaming-command --config settings.json doctor
streaming-command --config settings.json download
streaming-command --config settings.json run recording.wav
streaming-command --config settings.json run --asr-only recording.wav
streaming-command --config settings.json run
```

`doctor` 显示依赖、解释器、所选模型和各项配置来源，无需下载。`download` 解析当前配置启用的资源，也可以指定单项，例如 `download llm-base`。`run` 默认从麦克风收音，也可指定音频文件。`--config` 放在子命令前。
