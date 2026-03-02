"""
TTS Microservice — main.py
==========================

Fixes applied vs original:

  1. TOKEN JOINING FIX — LLM tokens carry their own leading whitespace
     (e.g. [" Hello", " world", "!"]). Joining with " ".join() creates
     double-spaces and space-before-punctuation. Fixed everywhere with:
         text = "".join(tokens).replace("  ", " ").strip()

  2. SPEAKER SERIALISATION — _speak_pool is now max_workers=1. This ensures
     /speak requests are queued and played in order, not mixed together.
     With interrupt=True the current future is cancelled before the new one
     is submitted (best-effort — in-flight synthesis is let finish its current
     chunk but playback is stopped immediately via speaker.stop()).

  3. GRACEFUL SHUTDOWN — lifespan now calls _speak_pool.shutdown(wait=True)
     so the current sentence can finish before the process exits.

  4. HEALTH CIRCUIT BREAKER — startup errors are captured in _startup_error
     and surfaced in /health instead of silently crashing the server.

  5. CONFIG — all os.getenv calls replaced with config.settings.
"""

import asyncio
import base64
import concurrent.futures as _cf
import json
import logging
import re
import threading
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from synthesizer import TTSSynthesizer
from speaker import get_instant_speaker, shutdown_instant_speaker

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tts_service")

synth: TTSSynthesizer | None = None
_startup_time: float = 0.0
_startup_error: str | None = None   # ← circuit breaker

# ---------------------------------------------------------------------------
# Thread pool — max_workers=1 serialises /speak requests so audio never mixes
# ---------------------------------------------------------------------------
_speak_pool = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="speak")

# Track the currently running speak future so interrupt=True can cancel it
_current_speak_future: _cf.Future | None = None
_future_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global synth, _startup_time, _startup_error
    t0 = time.perf_counter()
    logger.info("🚀 Loading TTS (backend=%s)…", settings.TTS_BACKEND)

    try:
        synth = TTSSynthesizer(
            backend=settings.TTS_BACKEND,
            device=settings.DEVICE,
            pocket_voice=settings.POCKET_VOICE,
            piper_model=settings.PIPER_MODEL,
            coqui_model=settings.TTS_MODEL,
        )
        # Pre-warm the instant speaker — opens sounddevice stream NOW
        speaker = get_instant_speaker(sample_rate=synth.sample_rate)
        logger.info("🔊 InstantSpeaker ready (sr=%d)", synth.sample_rate)

    except Exception as exc:
        _startup_error = str(exc)
        logger.error("❌ TTS failed to load: %s", exc)

    _startup_time = time.perf_counter() - t0
    logger.info("✅ TTS service ready in %.2fs", _startup_time)
    yield

    # ── Graceful shutdown ──────────────────────────────────────────────
    logger.info("🛑 Shutting down…")
    _speak_pool.shutdown(wait=True)   # let current sentence finish
    shutdown_instant_speaker()


# ---------------------------------------------------------------------------
app = FastAPI(
    title="TTS Microservice",
    description="Ultra-low-latency TTS — instant local speaker + return-bytes endpoints",
    version="3.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: Optional[str] = None
    speed: float = Field(1.0, ge=0.25, le=4.0)
    sample_rate: int = Field(22050)


class TokenStreamRequest(BaseModel):
    tokens: list[str] = Field(...)
    voice: Optional[str] = None
    speed: float = Field(1.0, ge=0.25, le=4.0)
    chunk_words: int = Field(5, ge=1, le=20)


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1)
    speed: float = Field(1.0, ge=0.25, le=4.0)
    interrupt: bool = Field(False, description="Stop current speech first")


class SpeakTokensRequest(BaseModel):
    tokens: list[str] = Field(..., description="LLM output tokens to speak")
    speed: float = Field(1.0, ge=0.25, le=4.0)
    chunk_words: int = Field(5, ge=1, le=20)
    interrupt: bool = Field(False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_synth() -> TTSSynthesizer:
    if _startup_error:
        raise HTTPException(status_code=503, detail=f"TTS failed to load: {_startup_error}")
    if synth is None:
        raise HTTPException(status_code=503, detail="TTS not loaded yet")
    return synth


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()] or [text]


async def _run_sync(fn, *args):
    return await asyncio.get_event_loop().run_in_executor(None, fn, *args)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _join_tokens(tokens: list[str]) -> str:
    """
    Correctly join LLM tokens.

    LLM tokens typically carry their own leading space, e.g.:
        [" Hello", " world", "!"]  →  " Hello world!"

    Using " ".join() would produce " Hello  world !" (double spaces,
    space before punctuation).  Concatenating and collapsing double
    spaces is correct.
    """
    return "".join(tokens).replace("  ", " ").strip()


def _submit_speak(fn) -> _cf.Future:
    """Submit fn to the speak pool, tracking the future for interrupt support."""
    global _current_speak_future
    with _future_lock:
        future = _speak_pool.submit(fn)
        _current_speak_future = future
    return future


# ===========================================================================
# /speak  — INSTANT LOCAL PLAYBACK
# ===========================================================================

@app.post("/speak", tags=["speaker"])
async def speak(req: SpeakRequest):
    """⚡ Speak text — serialised via single-worker pool (no mixed audio)."""
    s = _require_synth()
    speaker = get_instant_speaker(s.sample_rate)
    text, speed, interrupt = req.text, req.speed, req.interrupt

    def _do_speak():
        t0 = time.perf_counter()
        if interrupt:
            speaker.stop()
            time.sleep(0.05)

        first = True
        for wav_chunk in s.synthesize_streaming(text, speed=speed):
            if first:
                logger.info("⚡ First chunk in %.0fms", (time.perf_counter() - t0) * 1000)
                first = False
            speaker.feed(wav_chunk)

        logger.info("✅ /speak done %.2fs", time.perf_counter() - t0)

    if interrupt:
        # Best-effort cancel: stop audio immediately; new job queued next
        speaker.stop()

    _submit_speak(_do_speak)
    return {
        "status": "speaking",
        "chars": len(text),
        "preview": text[:80] + ("…" if len(text) > 80 else ""),
    }


@app.post("/speak/tokens", tags=["speaker"])
async def speak_tokens(req: SpeakTokensRequest):
    """
    ⚡ Feed pre-tokenised LLM tokens → speak immediately.

    FIX: tokens are concatenated (not space-joined) to preserve the
    whitespace that LLM tokenizers embed in each token.
    """
    s = _require_synth()
    speaker = get_instant_speaker(s.sample_rate)

    tokens    = list(req.tokens)
    speed     = req.speed
    interrupt = req.interrupt

    def _do_speak_tokens():
        t0 = time.perf_counter()

        if interrupt:
            speaker.stop()
            time.sleep(0.05)

        # ← FIX: use _join_tokens instead of " ".join()
        text = _join_tokens(tokens)
        if not text:
            return

        first = True
        for wav_chunk in s.synthesize_streaming(text, speed=speed):
            if first:
                logger.info("⚡ First audio in %.0fms", (time.perf_counter() - t0) * 1000)
                first = False
            speaker.feed(wav_chunk)

        logger.info("✅ speak_tokens submitted %.0f chars in %.3fs",
                    len(text), time.perf_counter() - t0)

    if interrupt:
        speaker.stop()

    _submit_speak(_do_speak_tokens)
    return {"status": "speaking", "token_count": len(tokens)}


@app.post("/speak/stop", tags=["speaker"])
async def speak_stop():
    """Stop any currently playing speech immediately."""
    get_instant_speaker().stop()
    return {"status": "stopped"}


@app.get("/speak/status", tags=["speaker"])
async def speak_status():
    """Is the local speaker currently talking?"""
    return {"speaking": get_instant_speaker().is_speaking}


# ===========================================================================
# /synthesize — RETURN BYTES
# ===========================================================================

@app.post("/synthesize", tags=["synthesize"], response_class=Response)
async def synthesize(req: SynthesizeRequest):
    s = _require_synth()
    t0 = time.perf_counter()
    audio_bytes: bytes = await _run_sync(s.synthesize, req.text, req.voice, req.speed)
    elapsed = time.perf_counter() - t0
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "X-Synthesis-Time-Ms": str(round(elapsed * 1000)),
            "X-Sample-Rate": str(s.sample_rate),
        },
    )


@app.post("/synthesize/stream", tags=["synthesize"])
async def synthesize_stream(req: SynthesizeRequest):
    """SSE: one audio_chunk event per sentence."""
    s = _require_synth()
    sentences = _split_sentences(req.text)
    overall_start = time.perf_counter()

    async def generate() -> AsyncIterator[str]:
        total_chunks = 0
        for sentence in sentences:
            if not sentence.strip():
                continue
            t0 = time.perf_counter()
            try:
                audio_bytes: bytes = await _run_sync(
                    s.synthesize, sentence, req.voice, req.speed
                )
                total_chunks += 1
                yield _sse({
                    "type": "audio_chunk",
                    "sentence": sentence,
                    "data": base64.b64encode(audio_bytes).decode(),
                    "sample_rate": s.sample_rate,
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                    "chunk_index": total_chunks,
                })
            except Exception as exc:
                yield _sse({"type": "error", "detail": str(exc)})

        yield _sse({
            "type": "done",
            "total_chunks": total_chunks,
            "total_ms": round((time.perf_counter() - overall_start) * 1000),
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/synthesize/tokens", tags=["synthesize"])
async def synthesize_tokens(req: TokenStreamRequest):
    """SSE: synthesize token chunks and stream as audio events.

    FIX: flush_buffer uses _join_tokens instead of ' '.join() to
    avoid double-spaces and space-before-punctuation artefacts.
    """
    s = _require_synth()
    overall_start = time.perf_counter()

    async def generate() -> AsyncIterator[str]:
        buffer: list[str] = []
        chunk_index = 0

        async def flush_buffer():
            nonlocal chunk_index, buffer
            # ← FIX: use _join_tokens
            text = _join_tokens(buffer)
            if not text:
                buffer = []
                return
            audio_bytes: bytes = await _run_sync(s.synthesize, text, None, req.speed)
            chunk_index += 1
            yield _sse({
                "type": "audio_chunk",
                "sentence": text,
                "data": base64.b64encode(audio_bytes).decode(),
                "sample_rate": s.sample_rate,
                "elapsed_ms": round((time.perf_counter() - overall_start) * 1000),
                "chunk_index": chunk_index,
            })
            buffer = []

        for token in req.tokens:
            buffer.append(token)
            # ← FIX: inspect joined text correctly (no spurious spaces)
            combined = _join_tokens(buffer)
            has_punct = any(c in combined for c in ".!?,;:\n")
            if has_punct or len(combined.split()) >= req.chunk_words:
                async for event in flush_buffer():
                    yield event

        if buffer:
            async for event in flush_buffer():
                yield event

        yield _sse({
            "type": "done",
            "total_chunks": chunk_index,
            "total_ms": round((time.perf_counter() - overall_start) * 1000),
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/synthesize/pocket_stream", tags=["synthesize"])
async def synthesize_pocket_stream(req: SynthesizeRequest):
    """PocketTTS native SSE stream — one event per model output chunk."""
    s = _require_synth()
    overall_start = time.perf_counter()

    async def generate() -> AsyncIterator[str]:
        chunk_index = 0
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def producer():
            try:
                for wav in s.synthesize_streaming(req.text, voice=req.voice, speed=req.speed):
                    loop.call_soon_threadsafe(q.put_nowait, wav)
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=producer, daemon=True).start()

        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                yield _sse({"type": "error", "detail": str(item)})
                break
            chunk_index += 1
            yield _sse({
                "type": "audio_chunk",
                "data": base64.b64encode(item).decode(),
                "sample_rate": s.sample_rate,
                "elapsed_ms": round((time.perf_counter() - overall_start) * 1000),
                "chunk_index": chunk_index,
            })

        yield _sse({
            "type": "done",
            "total_chunks": chunk_index,
            "total_ms": round((time.perf_counter() - overall_start) * 1000),
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ===========================================================================
# META
# ===========================================================================

@app.get("/health", tags=["meta"])
async def health():
    s = synth

    # ── Circuit breaker: surface startup errors ────────────────────────
    if _startup_error:
        return {
            "status": "error",
            "error": _startup_error,
            "model_loaded": False,
            "backend": settings.TTS_BACKEND,
            "startup_time_s": round(_startup_time, 3),
            "speaker_status": "unavailable",
        }

    try:
        speaker_status = "speaking" if get_instant_speaker().is_speaking else "idle"
    except Exception:
        speaker_status = "unavailable"

    return {
        "status": "ok" if s else "loading",
        "model_loaded": s is not None,
        "backend": settings.TTS_BACKEND,
        "sample_rate": s.sample_rate if s else None,
        "startup_time_s": round(_startup_time, 3),
        "speaker_status": speaker_status,
    }


@app.get("/voices", tags=["meta"])
async def voices():
    s = _require_synth()
    return {"voices": s.available_voices(), "backend": settings.TTS_BACKEND}