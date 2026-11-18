import asyncio
import sys
from pathlib import Path
import pytest
from streamingcommand import StreamingASR, StreamingCommand, StreamingConfig
from streamingcommand._runtime.asr_backend import FakeASR


class VAD:
    def is_speech(self, frame, rate):
        return frame[0:2] != b"\0\0"


def config():
    value = StreamingConfig(interval_ms=800, endpoint_ms=120)
    value.command.enabled = False
    return value


@pytest.mark.asyncio
async def test_final_drain_and_one_active_session():
    asr = FakeASR()
    async with StreamingASR(config(), asr_backend=asr) as runtime:
        async with runtime.session(vad=VAD()) as session:
            collected = []
            async def consume():
                async for event in session.events():
                    collected.append(event)
            consumer = asyncio.create_task(consume())
            with pytest.raises(RuntimeError, match="Only one"):
                await runtime.session(vad=VAD()).start()
            await session.feed(b"\x01\x01" * 16000)
            await session.feed(bytes(16000))
            await session.finish()
            await consumer
            finals = [e for e in collected if e["type"] == "transcript.final"]
            assert len(finals) == 1
            assert finals[0]["complete_audio_coverage"] is True
            assert collected[-1]["type"] == "session.completed"
        assert not runtime._lock.locked()
    assert asr.status["state"] == "closed"


@pytest.mark.asyncio
async def test_source_error_and_early_stop_cleanup():
    runtime = StreamingASR(config(), asr_backend=FakeASR())
    async def broken_source():
        yield bytes(640)
        raise ValueError("input failed")
    with pytest.raises(ValueError, match="input failed"):
        async for event in runtime.stream(broken_source()):
            pass
    assert not runtime._lock.locked()
    await runtime.close()


@pytest.mark.asyncio
async def test_bounded_queue_cancellation_releases():
    value = config()
    value.event_queue_size = 1
    runtime = StreamingASR(value, asr_backend=FakeASR())
    await runtime.start()
    session = runtime.session(vad=VAD())
    await session.start()
    await session.feed(b"\x01\x01" * 16000)
    await asyncio.sleep(.02)
    await asyncio.wait_for(session.close(), 2)
    assert not runtime._lock.locked()
    await runtime.close()


@pytest.mark.asyncio
async def test_invalid_pcm_and_repeated_sessions():
    async with StreamingASR(config(), asr_backend=FakeASR()) as runtime:
        for _ in range(2):
            async with runtime.session(vad=VAD()) as session:
                with pytest.raises(ValueError, match="PCM16"):
                    await session.feed(b"x")
                drain = asyncio.create_task(session.finish())
                assert [e["type"] async for e in session.events()][-1] == "session.completed"
                await drain


@pytest.mark.asyncio
async def test_history_and_llm_are_integrated():
    class Parser:
        default_candidates = {"I": ["bend"], "T": ["none"], "P": ["none", "left"]}
        status = {"ready": True, "max_history_depth": 1}
        histories = []
        async def start(self): pass
        async def close(self): pass
        async def parse(self, text, candidates, history):
            self.histories.append(dict(history))
            return {"command": "Y", "resolved": True, "slots": {"I": "bend", "T": "none", "P": "left"}}
    parser = Parser()
    runtime = StreamingCommand(config(), asr_backend=FakeASR(), command_backend=parser)
    async with runtime:
        async with runtime.session(vad=VAD()) as session:
            events = []
            async def consume():
                async for e in session.events(): events.append(e)
            consumer = asyncio.create_task(consume())
            for _ in range(2):
                await session.feed(b"\x01\x01" * 16000 + bytes(16000))
                await session.audio._idle.wait()
                await session.audio._outgoing.join()
                await session.commands.finish()
            await session.finish()
            await consumer
            finals = [e for e in events if e["type"] == "command.final"]
            assert len(finals) == 2
            assert finals[-1]["history_after"] == {"H1": "I=bend;T=none;P=left"}
            assert any(h.get("H1") for h in parser.histories)


def test_import_is_lightweight_in_fresh_process():
    import subprocess
    result = subprocess.run([sys.executable, "-c", "import streamingcommand,sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_owner_close_stops_source_even_when_iterator_is_retained():
    closed = asyncio.Event()
    async def source():
        try:
            while True:
                yield bytes(640)
                await asyncio.sleep(.01)
        finally:
            closed.set()
    runtime = StreamingASR(config(), asr_backend=FakeASR())
    stream = runtime.stream(source())
    assert (await anext(stream))["type"] == "session.started"
    await asyncio.wait_for(runtime.close(), 2)
    assert closed.is_set()
    await stream.aclose()


@pytest.mark.asyncio
async def test_asr_only_accepts_a_command_enabled_config():
    value = config()
    value.command.recognition_logic = "semantic-pair"
    async with StreamingASR(value, asr_backend=FakeASR()) as runtime:
        async with runtime.session(vad=VAD()) as session:
            assert session.commands is None
