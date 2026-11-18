# 原项目 Demo 接入

在 PhonoCommandDialogue 工作区中，`RealtimeTranscriptDemo/run.sh` 新启动默认使用本包。
WebUI 左上角可以在 StreamingCommand 包与原版之间切换；录音期间禁止切换。
切换先关闭当前拥有的 ASR/LLM，再加载所选实现，沿用同一端口、候选文件与录音界面。

```bash
RealtimeTranscriptDemo/run.sh
RealtimeTranscriptDemo/run.sh --implementation original
```

已有服务会保持原状态并披露实现不同，不会被脚本自动终止或迁移。
普通的 LLM 模型切换保持同一个 ASR；切换整个识别实现会重新加载 ASR。
包内提供当前 H1 默认/最新模型；原版保留其他历史实验模型。
本 GitHub 仓库只发布独立包。原项目 WebUI 的修改、录音与其他工程文件不上传到这里。
