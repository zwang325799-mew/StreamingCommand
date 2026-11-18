"""Command-line diagnostics, downloads and file/microphone recognition."""
import argparse
import asyncio
import json
import platform
import sys
from .config import StreamingConfig


def main(argv=None):
    parser = argparse.ArgumentParser(prog="streaming-command")
    parser.add_argument("--config", help="JSON config; paths resolve relative to this file")
    commands = parser.add_subparsers(dest="operation", required=True)
    commands.add_parser("doctor", help="Report dependencies without loading models")
    download = commands.add_parser("download", help="Download and verify pinned model resources")
    download.add_argument("models", nargs="*", default=["asr", "asr-base", "language", "speech", "llm-base"])
    run = commands.add_parser("run", help="Recognize a file, or listen to the microphone")
    run.add_argument("audio", nargs="?")
    run.add_argument("--asr-only", action="store_true")
    args = parser.parse_args(argv)
    config = StreamingConfig.load(args.config) if args.config else StreamingConfig()
    if args.operation == "doctor":
        import importlib.metadata as metadata
        versions = {}
        for name in ("streamingcommand", "torch", "transformers", "peft", "funasr", "sounddevice", "vllm"):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
        print(json.dumps({"python": sys.executable, "platform": platform.platform(), "versions": versions,
                          "asr_backend": config.asr.backend, "asr_python": config.asr.python or sys.executable}, indent=2))
    elif args.operation == "download":
        from .resources import ModelStore
        store = ModelStore(config.cache_dir, offline=config.offline)
        for name in args.models:
            print(name, store.ensure(name), flush=True)
    else:
        from .pipeline import StreamingCommand, StreamingASR
        async def recognize():
            kind = StreamingASR if args.asr_only else StreamingCommand
            async with kind(config) as recognizer:
                events = recognizer.transcribe_file(args.audio) if args.audio else recognizer.listen()
                async for event in events:
                    print(json.dumps(event, ensure_ascii=False), flush=True)
        asyncio.run(recognize())


if __name__ == "__main__":
    main()
