"""
synthesizer.py — TTSSynthesizer
Wraps PocketTTSStreaming / InstantStreamingTTS from tts.py and exposes
a simple synchronous .synthesize() → bytes interface for the FastAPI service.

The microservice is stateless per-request (no local audio playback).
Audio is returned as raw WAV bytes so the caller can stream / buffer it.
"""

import io
import wave
import struct
import logging
import os
import threading
from typing import Optional, Iterator

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ndarray_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array [-1, 1] → 16-bit mono WAV bytes."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _silence_wav(duration_s: float = 0.5, sample_rate: int = 22050) -> bytes:
    """Return a silent WAV clip of the given duration."""
    n = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Backend: PocketTTS
# ---------------------------------------------------------------------------

class _PocketBackend:
    """
    Thin synthesis-only adapter for pocket_tts.
    Does NOT use sounddevice — audio is captured to memory instead.
    """

    def __init__(self, voice: str = "alba"):
        from pocket_tts import TTSModel  # type: ignore
        self.model = TTSModel.load_model()
        self.voice_state = self.model.get_state_for_audio_prompt(voice)
        self.sample_rate: int = self.model.sample_rate
        self._lock = threading.Lock()
        logger.info("PocketTTS backend ready (voice=%s, sr=%d)", voice, self.sample_rate)

    def synthesize(self, text: str, rate: float = 1.0) -> bytes:
        chunks: list[np.ndarray] = []
        with self._lock:
            for chunk in self.model.generate_audio_stream(self.voice_state, text):
                audio = chunk.numpy()
                if rate != 1.0:
                    indices = np.arange(0, len(audio), rate)
                    indices = indices[indices < len(audio)].astype(int)
                    audio = audio[indices]
                chunks.append(audio)
        if not chunks:
            return _silence_wav(sample_rate=self.sample_rate)
        return _ndarray_to_wav_bytes(np.concatenate(chunks), self.sample_rate)

    def available_voices(self) -> list[str]:
        try:
            return list(self.model.list_voices())
        except Exception:
            return ["alba"]


# ---------------------------------------------------------------------------
# Backend: Piper
# ---------------------------------------------------------------------------

class _PiperBackend:
    """Synthesis-only adapter for piper-tts."""

    def __init__(self, model_path: str, config_path: Optional[str] = None):
        from piper.voice import PiperVoice  # type: ignore
        cfg = config_path or (model_path + ".json")
        self.voice = PiperVoice.load(model_path, config_path=cfg)
        self.sample_rate: int = self.voice.config.sample_rate
        logger.info("Piper backend ready (model=%s, sr=%d)", model_path, self.sample_rate)

    def synthesize(self, text: str, rate: float = 1.0) -> bytes:
        audio_bytes = b"".join(
            self.voice.synthesize_stream_raw(text, length_scale=1.0 / max(rate, 0.1))
        )
        if not audio_bytes:
            return _silence_wav(sample_rate=self.sample_rate)
        pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return _ndarray_to_wav_bytes(pcm, self.sample_rate)

    def available_voices(self) -> list[str]:
        return ["default"]


# ---------------------------------------------------------------------------
# Backend: Coqui TTS (fallback)
# ---------------------------------------------------------------------------

class _CoquiBackend:
    """Synthesis-only adapter for Coqui TTS (original synthesizer.py approach)."""

    def __init__(self, model_name: str, device: str = "cpu"):
        from TTS.api import TTS  # type: ignore
        self.tts = TTS(model_name).to(device)
        self.sample_rate: int = 22050
        logger.info("Coqui TTS backend ready (model=%s, device=%s)", model_name, device)

    def synthesize(self, text: str, rate: float = 1.0,
                   speaker_wav: Optional[str] = None) -> bytes:
        buf = io.BytesIO()
        self.tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            speed=rate,
            file_path=buf,
        )
        return buf.getvalue()

    def available_voices(self) -> list[str]:
        if hasattr(self.tts, "speakers") and self.tts.speakers:
            return self.tts.speakers
        return ["default"]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class TTSSynthesizer:
    """
    Unified TTS synthesizer used by the FastAPI microservice.

    Backend selection (env var TTS_BACKEND):
      pocket  — PocketTTS  (default, fastest)
      piper   — Piper ONNX
      coqui   — Coqui TTS

    All backends expose:
      .synthesize(text, voice, speed)  → WAV bytes
      .available_voices()              → list[str]
      .sample_rate                     → int
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        device: str = "cpu",
        # PocketTTS
        pocket_voice: str = "alba",
        # Piper
        piper_model: Optional[str] = None,
        piper_config: Optional[str] = None,
        # Coqui
        coqui_model: Optional[str] = None,
    ):
        backend = (backend or os.getenv("TTS_BACKEND", "pocket")).lower()
        self._backend_name = backend
        self._backend = None

        if backend == "pocket":
            self._backend = _PocketBackend(voice=pocket_voice)

        elif backend == "piper":
            model = piper_model or os.getenv(
                "PIPER_MODEL", "en_US-lessac-medium.onnx"
            )
            self._backend = _PiperBackend(model_path=model, config_path=piper_config)

        elif backend == "coqui":
            model = coqui_model or os.getenv(
                "TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC"
            )
            self._backend = _CoquiBackend(model_name=model, device=device)

        else:
            raise ValueError(
                f"Unknown TTS_BACKEND '{backend}'. Choose: pocket | piper | coqui"
            )

        logger.info("TTSSynthesizer ready (backend=%s)", backend)

    # ------------------------------------------------------------------
    @property
    def sample_rate(self) -> int:
        return getattr(self._backend, "sample_rate", 22050)

    # ------------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> bytes:
        """
        Synthesize text → WAV bytes.

        Args:
            text:  Input text.
            voice: Voice name / speaker_wav path (backend-dependent).
            speed: Speech rate multiplier (1.0 = normal).

        Returns:
            Raw WAV bytes (16-bit, mono).
        """
        if not text or not text.strip():
            return _silence_wav(0.1, self.sample_rate)

        try:
            if self._backend_name == "pocket":
                # PocketTTS ignores voice per-call (voice set at init)
                return self._backend.synthesize(text, rate=speed)

            elif self._backend_name == "piper":
                return self._backend.synthesize(text, rate=speed)

            elif self._backend_name == "coqui":
                return self._backend.synthesize(
                    text, rate=speed, speaker_wav=voice
                )

        except Exception as exc:
            logger.error("Synthesis failed for text=%r: %s", text[:60], exc)
            return _silence_wav(max(0.5, len(text) * 0.05), self.sample_rate)

    # ------------------------------------------------------------------
    def synthesize_streaming(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        chunk_size_words: int = 5,
    ) -> Iterator[bytes]:
        """
        Parallel-pipeline streaming synthesis.

        How it works:
        ─────────────
        1. Text is split into word-group segments (same as before).
        2. A background producer thread synthesizes segments and puts
           finished WAV bytes into a queue.
        3. This generator (the consumer) yields each WAV the instant it
           arrives from the queue — while the producer is ALREADY working
           on the next segment in parallel.

        Result: chunk N is playing while chunk N+1 is being synthesized,
        eliminating the dead-silence gap between chunks.

        PocketTTS native path:
        ─────────────────────
        PocketTTS has its own internal streaming generator (model chunks
        arrive incrementally). For that backend we flush model chunks
        every ~100ms of audio so the very first audio arrives in <50ms,
        then the parallel-pipeline takes over for subsequent segments.
        """
        import queue as _queue
        from concurrent.futures import ThreadPoolExecutor

        # ── PocketTTS native streaming (intra-chunk parallelism) ──────────
        if self._backend_name == "pocket" and hasattr(
            self._backend.model, "generate_audio_stream"
        ):
            # Collect raw model chunks, flush every 100ms → yield each as WAV.
            # The model generates while the caller consumes, giving true overlap.
            chunks: list[np.ndarray] = []

            # Queue bridges the model-generator thread and this generator.
            wav_q: _queue.Queue = _queue.Queue()
            sr = self._backend.sample_rate

            def _pocket_producer():
                buf: list[np.ndarray] = []
                try:
                    for raw in self._backend.model.generate_audio_stream(
                        self._backend.voice_state, text
                    ):
                        audio = raw.numpy()
                        if speed != 1.0:
                            idx = np.arange(0, len(audio), speed)
                            idx = idx[idx < len(audio)].astype(int)
                            audio = audio[idx]
                        buf.append(audio)
                        # Flush every ~100ms of audio for low latency
                        if sum(len(c) for c in buf) >= sr * 0.1:
                            wav_q.put(_ndarray_to_wav_bytes(np.concatenate(buf), sr))
                            buf = []
                    if buf:
                        wav_q.put(_ndarray_to_wav_bytes(np.concatenate(buf), sr))
                except Exception as exc:
                    logger.error("PocketTTS producer error: %s", exc)
                finally:
                    wav_q.put(None)  # sentinel

            t = threading.Thread(target=_pocket_producer, daemon=True,
                                 name="pocket-producer")
            t.start()

            while True:
                item = wav_q.get()
                if item is None:
                    break
                yield item
            return

        # ── Parallel-pipeline for Piper / Coqui (and PocketTTS fallback) ──
        #
        # Segments: ["word1 word2 word3", "word4 word5 word6", ...]
        words = text.split()
        if not words:
            return

        segments: list[str] = []
        for i in range(0, len(words), chunk_size_words):
            seg = " ".join(words[i : i + chunk_size_words]).strip()
            if seg:
                segments.append(seg)

        if not segments:
            return

        # Use a bounded queue so the producer stays at most 1 chunk ahead.
        # maxsize=2 means: producer synthesizes chunk[i+1] while consumer
        # yields chunk[i] — perfect 1-ahead prefetch with no memory bloat.
        wav_q: _queue.Queue = _queue.Queue(maxsize=2)
        _SENTINEL = object()

        def _parallel_producer():
            for seg in segments:
                try:
                    wav = self.synthesize(seg, voice=voice, speed=speed)
                    wav_q.put(wav)           # blocks if consumer is slow (backpressure)
                except Exception as exc:
                    logger.error("Parallel producer error for %r: %s", seg[:40], exc)
                    wav_q.put(
                        _silence_wav(max(0.3, len(seg) * 0.05), self.sample_rate)
                    )
            wav_q.put(_SENTINEL)

        producer_thread = threading.Thread(
            target=_parallel_producer,
            daemon=True,
            name="synth-parallel-producer",
        )
        producer_thread.start()

        # Yield chunks as soon as they are ready — producer is already
        # working on the next one while this chunk is being played.
        while True:
            item = wav_q.get()
            if item is _SENTINEL:
                break
            yield item

    # ------------------------------------------------------------------
    def available_voices(self) -> list[str]:
        try:
            return self._backend.available_voices()
        except Exception:
            return ["default"]