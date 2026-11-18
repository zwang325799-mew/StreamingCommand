# StreamingCommand

独立的本地 Python 包：连续语音转写、定稿续接、命令解析、H1 历史与澄清。
麦克风采集、文件回放、重采样和音频分块由包处理。一个常驻 ASR 实例，LLM 使用独立的常驻进程。

## 下载和安装

```bash
git clone https://github.com/zwang325799-mew/StreamingCommand.git
cd StreamingCommand
python -m pip install .
streaming-command doctor
```

也可直接安装：

```bash
python -m pip install "git+https://github.com/zwang325799-mew/StreamingCommand.git"
```

Python 3.10–3.13。安装后可以在任何工作目录导入；无需原 PhonoCommandDialogue 仓库。
默认使用原生 PyTorch 后端，自动选择可用 CUDA 或 CPU。MPS 尚未验收，macOS 使用 CPU。
CPU 保留同一接口，实际速度和识别结果需要按设备验证，不承诺与 CUDA/vLLM 相同。
Linux 麦克风需要系统 PortAudio；例如 Debian/Ubuntu 安装 `libportaudio2`。
安装只安装代码和依赖；首次 `start()` 下载模型，需联网与足够磁盘空间。可预先下载：

```bash
streaming-command download
```

## 直接使用麦克风

```python
import asyncio
from streamingcommand import StreamingCommand

async def main():
    async with StreamingCommand() as recognizer:
        async for event in recognizer.listen():
            if event["type"] in ("transcript.final", "command.final"):
                print(event)

asyncio.run(main())
```

`async with` 内模型持续驻留，可连续识别任意多句。Ctrl+C / 退出上下文会释放自己拥有的模型进程和麦克风。
仅需转写时将 `StreamingCommand` 换成 `StreamingASR`，不加载 LLM。

## 文件、文字与已有音频流

```python
async with StreamingCommand() as recognizer:
    async for event in recognizer.transcribe_file("recording.wav"):
        print(event)

    result = await recognizer.parse("Bend left.")
    print(result["slots"])
```

文件自动解码、下混单声道和连续重采样；默认按真实时间回放。`realtime=False` 适合离线调用，慢设备遇到积压会明确报错。
已有采集系统可把 **PCM16 little-endian、单声道、16kHz** 的异步音频源传给 `recognizer.stream(source)`。
也可使用 `recognizer.session()` 的 `feed()` / `events()`；输入和事件消费应在不同 asyncio task 中进行。
输入必须包含静音；按词或句子切段不属于调用者职责。

`transcript.partial` 携带 segment_id/revision，用于替换同段旧假设；`transcript.final` 提交一段文本。
`command.partial` 仅供预览；`command.final` 中 `resolved` / `execute` 表示可执行性，`command.error` 表示解析失败。
最终文本不保证语义已经完整，不应把所有 final 直接执行为机器人动作。
每个会话串行维护历史与澄清。相同文字的重复指令不会自动去重。

## 配置

一份 JSON 可以同时指定语音识别模型、命令模型及候选动作、目标、方向。两份完整配置随包安装。

| 配置单 | 加载的模型与识别方式 |
| --- | --- |
| [本地最新测试版](src/streamingcommand/assets/presets/local-latest.json) | 从六个本地目录加载，ASR 使用 `v17-yue-precision`，命令解析使用 epoch 1 / step936，支持 H1、澄清和相邻片段比较。路径对应测试机，换机器时修改各项 `path`。 |
| [Hugging Face 发布版](src/streamingcommand/assets/presets/huggingface-release.json) | 六个资源逐项固定仓库 URL 和提交 SHA。ASR 使用 `humoiww/PhonoCommand-ASR`，命令解析使用 `humoiww/PhonoCommand-LLM-0.8B` 发布的 step4400 权重，预测命令与 I/T/P。 |

安装后可复制任一范例再编辑。下面的命令将发布版保存为当前目录下的 `settings.json`，使用本地版时把文件名改为 `local-latest.json`。

```bash
python -c "from importlib.resources import files; from pathlib import Path; Path('settings.json').write_bytes(files('streamingcommand').joinpath('assets/presets/huggingface-release.json').read_bytes())"
```

```python
import asyncio
from streamingcommand import StreamingCommand

async def main():
    recognizer = await StreamingCommand.initialize(config_path="settings.json")
    try:
        print(recognizer.info()["command"]["model_id"])
        print((await recognizer.parse("Bend left."))["slots"])
    finally:
        await recognizer.close()

asyncio.run(main())
```

编辑 `command.candidates` 中的 `I`、`T`、`P` 可以更换候选。`command.model` 选择命令模型，`models` 中的同名项指定权重来源。换远端版本就改 `url` 和 `revision`，使用本地目录就设 `"source": "local"` 并填 `path`。关闭旧实例后重新初始化即可生效。省略配置路径时使用 step3003，完整默认值见[默认配置单](src/streamingcommand/assets/settings.json)。完整调用签名和器械护士示例见[功能文档](doc/API.md)。

也可以在 Python 中构建配置对象。

```python
from streamingcommand import StreamingConfig, ASRConfig, CommandConfig, StreamingCommand

config = StreamingConfig(
    asr=ASRConfig(backend="native", device="auto", profile="v17-yue-precision"),
    command=CommandConfig(model="clarification-latest", recognition_logic="semantic-pair"),
)
```

也可以 `StreamingCommand("settings.json")`，JSON 字段对应上述配置类。路径相对于配置文件。
默认保留 `v15-canto-commit` 与 `clarification-new`；`clarification-latest` 为完整 epoch1/step936。
两套已训练 r16 适配器及决策头随包提供，基座按固定 revision 下载；没有生成式提示词替代这些分类头。
澄清提议、自然回答、保留 N、business none/null、tracking 和 scrub-nurse 空 P 使用现有契约。
候选文件位于用户配置目录，可通过 `CandidateProfiles` 新建、保存、重命名与另存为；示例见 `examples/`。
`original` 与 `semantic-pair` 控制定稿后的语义处理；后者比较 B 与未执行的 A+B，仅允许一致的可执行动作自动通过。
全部模式共享一个 ASR 引擎，不额外解码音频。历史最多 H1。

## vLLM 与 Windows

详见 [vLLM 安装与 Windows 迁移](docs/VLLM_WINDOWS.md)。vLLM 是可选的 ASR 后端，不是默认安装的跨平台依赖。
可设置 `ASRConfig(backend="vllm", python="/path/to/asr-python", device="cuda:0")` 沿用已验证的旧 SDK 路径，
或用 `backend="vllm-builtin"` 接现代内置 Qwen3-ASR 的 vLLM，包括 Windows 社区构建。
不同后端明确报告身份；选定 vLLM 失败时不会静默切换到 PyTorch。

## 离线与验证

模型缓存位于系统的 StreamingCommand 用户缓存目录。配置 `cache_dir`、`offline=True`，或提供
`model_dir/alternate_model_dir/language_model_dir/speech_model_dir/base_model_dir` 本地目录。
范例中的 Hugging Face 资源按完整提交 SHA 下载，并按随包清单核对运行所需文件的 SHA-256。自定义仓库或提交使用该版本的 Hugging Face 快照，需要提供与所选模型兼容的文件。
原始录音、训练集、历史实验目录和旧 Demo 不包含在本仓库。

```bash
python -m pip install ".[test]"
python -m pytest
```

平台和实测范围在 [验证记录](docs/VALIDATION.md) 中列出；Linux 上的结果不等于 Windows/macOS 实机验收。
