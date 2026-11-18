"""Persistent engines, single-owner streaming sessions and asynchronous events."""
import asyncio
from contextlib import suppress
import copy
from .config import StreamingConfig


class StreamingCommand:
    def __init__(self, config=None, *, asr_backend=None, command_backend=None):
        self.config = StreamingConfig.load(config) if isinstance(config, (str, __import__("pathlib").Path)) else config or StreamingConfig()
        self.asr = asr_backend
        self.commands = command_backend
        self._lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._session = None
        self._closed = False
        self._ready = False

    async def start(self):
        async with self._start_lock:
            if self._closed:
                raise RuntimeError("Runtime is closed; create a new instance")
            if self._ready:
                return self
            try:
                if self.asr is None:
                    from .backends import build_backends
                    self.asr, self.commands = await build_backends(self.config)
                await self.asr.start()
                if self.commands is not None:
                    await self.commands.start()
                self._ready = True
            except BaseException:
                await self.close()
                raise
        return self

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, *exc):
        await self.close()

    async def close(self):
        if self._closed:
            return
        self._closed = True
        self._ready = False
        try:
            if self._session is not None:
                await self._session.close()
        finally:
            try:
                if self.asr is not None:
                    await self.asr.close()
            finally:
                if self.commands is not None:
                    await self.commands.close()

    def info(self):
        return {"implementation": "streamingcommand", "ready": self._ready,
                "asr": copy.deepcopy(self.asr.status) if self.asr else None,
                "command": copy.deepcopy(self.commands.status) if self.commands else None}

    def session(self, *, candidates=None, history=None, vad=None):
        return CommandStream(self, candidates=candidates, history=history, vad=vad)

    async def stream(self, source, *, candidates=None, history=None):
        """Consume PCM16 mono16k source and yield independent transcript/command events."""
        async with self.session(candidates=candidates, history=history) as session:
            async def receive():
                try:
                    async for pcm in source:
                        await session.feed(pcm)
                    await session.finish()
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    await session.fail(exc)
                finally:
                    if hasattr(source, "aclose"):
                        await source.aclose()
            receiver = asyncio.create_task(receive())
            session._producer = receiver
            try:
                async for event in session.events():
                    yield event
            finally:
                receiver.cancel()
                with suppress(asyncio.CancelledError):
                    await receiver

    async def listen(self, *, device=None, candidates=None, history=None):
        from .audio import microphone
        async for event in self.stream(microphone(device), candidates=candidates, history=history):
            yield event

    async def transcribe_file(self, path, *, realtime=True, candidates=None, history=None):
        from .audio import audio_file
        async for event in self.stream(audio_file(path, realtime=realtime), candidates=candidates, history=history):
            yield event

    async def parse(self, text, *, candidates=None, history=None):
        await self.start()
        if self.commands is None:
            raise RuntimeError("Command parsing is disabled")
        if self._lock.locked():
            raise RuntimeError("A streaming session is active; use its feedback API")
        async with self._lock:
            return await self.commands.parse(text, candidates or self.commands.default_candidates, history)


class StreamingASR(StreamingCommand):
    """ASR-only convenience API, with the identical audio and lifecycle interfaces."""
    def __init__(self, config=None, **kwargs):
        config = StreamingConfig.load(config) if isinstance(config, (str, __import__("pathlib").Path)) else copy.deepcopy(config or StreamingConfig())
        config.command.enabled = False
        super().__init__(config, **kwargs)


class CommandStream:
    def __init__(self, owner, *, candidates=None, history=None, vad=None):
        self.owner = owner
        self.candidates, self.history, self.vad = candidates, history, vad
        self.queue = asyncio.Queue(maxsize=owner.config.event_queue_size)
        self.audio = self.commands = None
        self._started = self._closed = self._finished = False
        self._owns_lock = False
        self._consumer = False
        self._failure = None
        self._producer = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def start(self):
        if self._started:
            return
        await self.owner.start()
        if self.owner._lock.locked():
            raise RuntimeError("Only one active audio session is supported")
        await self.owner._lock.acquire()
        self._owns_lock = True
        self.owner._session = self
        try:
            from ._runtime.streaming import StreamingSession
            from ._runtime.endpoint_commit_bounded import BoundedYueEndpointCommitSession
            from ._runtime.session_profiles import endpoint_policy
            from ._runtime.command_stream import CommandSession
            from ._runtime.semantic_pair import PairedCommandSession, validate_logic
            config = self.owner.config
            logic = validate_logic(config.command.recognition_logic if self.owner.commands is not None else "original", self.owner.commands)
            if self.owner.commands is not None:
                kind = PairedCommandSession if logic == "semantic-pair" else CommandSession
                self.commands = kind(self.owner.commands, self._emit, self.candidates or self.owner.commands.default_candidates, self.history)
            kind = BoundedYueEndpointCommitSession if endpoint_policy(self.owner.asr, config.endpoint_ms)["continuation_enabled"] else StreamingSession
            self.audio = kind(self.owner.asr, self._transcript, interval_ms=config.interval_ms,
                endpoint_ms=config.endpoint_ms, max_segment_ms=config.max_segment_ms, vad=self.vad)
            await self.audio.start()
            self._started = True
            await self._emit({"type": "session.started", "sample_rate": 16000, "implementation": "streamingcommand"})
        except BaseException:
            await self.close()
            raise

    async def _emit(self, event):
        if not self._closed:
            await self.queue.put(event)

    async def _transcript(self, event):
        await self._emit(event)
        if self.commands and event.get("type") in {"transcript.partial", "transcript.final"}:
            self.commands.submit(event)

    async def feed(self, pcm):
        if not self._started or self._closed or self._finished:
            raise RuntimeError("Audio session is not accepting input")
        if not isinstance(pcm, bytes) or len(pcm) % 2:
            raise ValueError("Expected little-endian PCM16 mono16k bytes")
        if len(pcm) > self.owner.config.max_pending_audio_ms * 32:
            raise BufferError("Audio packet exceeds the configured buffer capacity")
        # Backpressure is explicit. Audio is never silently dropped.
        pending = sum(len(job.pcm) for job in self.audio._finals)
        if self.audio._outgoing.qsize() > self.owner.config.event_queue_size:
            raise BufferError("Event consumer cannot keep up; stop or consume events faster")
        if pending + len(pcm) > self.owner.config.max_pending_audio_ms * 32:
            raise BufferError("ASR cannot keep up with incoming audio; stop or pace the source")
        await self.audio.feed(pcm)

    async def feedback(self, text, clarification_id):
        if self.commands is None:
            raise RuntimeError("Command parsing is disabled")
        await self.commands.feedback(text, clarification_id)

    def configure_candidates(self, candidates):
        if self.commands is None:
            raise RuntimeError("Command parsing is disabled")
        return self.commands.configure(candidates)

    async def finish(self):
        if self._finished:
            return
        self._finished = True
        await self.audio.finish()
        if self.commands:
            await self.commands.finish()
        await self._emit({"type": "session.completed"})
        await self.queue.put(None)

    async def fail(self, exc):
        self._failure = exc
        await self.queue.put(None)

    async def events(self):
        if self._consumer:
            raise RuntimeError("Only one event consumer is supported")
        self._consumer = True
        try:
            while True:
                event = await self.queue.get()
                if event is None:
                    if self._failure:
                        raise self._failure
                    return
                yield event
        finally:
            self._consumer = False

    async def close(self):
        if self._closed:
            return
        self._closed = True
        if self._producer is not None and self._producer is not asyncio.current_task():
            self._producer.cancel()
            with suppress(asyncio.CancelledError):
                await self._producer
        # Draining the queue releases publishers when the consumer stops early.
        while not self.queue.empty():
            self.queue.get_nowait()
        self.queue.put_nowait(None)
        try:
            if self.audio:
                await self.audio.close()
        finally:
            try:
                if self.commands:
                    await self.commands.close()
            finally:
                if self._owns_lock:
                    self._owns_lock = False
                    self.owner._session = None
                    self.owner._lock.release()
