"""
TTS Microservice — main.py  (v3 — instant local playback)
==========================================================
/speak endpoint now plays audio INSTANTLY as each chunk is synthesized.
First audio heard in ~50-100ms instead of 3-4 seconds.

How it works:
  synthesizer.synthesize_streaming()  →  yields WAV chunks as generated
  InstantSpeaker.feed(chunk)          →  queued & played the instant it arrives
  sounddevice OutputStream            →  persistent open stream, zero open overhead

Endpoints:
  POST /speak                   → speak text NOW, instant first-audio
  POST /speak/tokens            → feed LLM tokens, speak in real-time chunks
  POST /speak/stop              → interrupt immediately
  GET  /speak/status            → is speaker talking?

  POST /synthesize              → return full WAV bytes
  POST /synthesize/stream       → SSE WAV chunks per sentence
  POST /synthesize/tokens       → SSE WAV chunks from token list
  POST /synthesize/pocket_stream→ PocketTTS native SSE stream

  GET  /voices
  GET  /health
"""

import asyncio
import base64
import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from synthesizer import TTSSynthesizer
from speaker import get_instant_speaker, shutdown_instant_speaker

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tts_service")

synth: TTSSynthesizer | None = None
_startup_time: float = 0.0


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global synth, _startup_time
    t0 = time.perf_counter()

    backend = os.getenv("TTS_BACKEND", "pocket")
    logger.info("🚀 Loading TTS (backend=%s)…", backend)

    synth = TTSSynthesizer(
        backend=backend,
        device=os.getenv("DEVICE", "cpu"),
        pocket_voice=os.getenv("POCKET_VOICE", "alba"),
        piper_model=os.getenv("PIPER_MODEL"),
        coqui_model=os.getenv("TTS_MODEL"),
    )

    # Pre-warm the instant speaker — opens sounddevice stream NOW, not on first request
    speaker = get_instant_speaker(sample_rate=synth.sample_rate)
    logger.info("🔊 InstantSpeaker ready (sr=%d)", synth.sample_rate)

    _startup_time = time.perf_counter() - t0
    logger.info("✅ TTS service ready in %.2fs", _startup_time)
    yield

    logger.info("🛑 Shutting down…")
    shutdown_instant_speaker()


# ---------------------------------------------------------------------------
app = FastAPI(
    title="TTS Microservice",
    description="Ultra-low-latency TTS — instant local speaker + return-bytes endpoints",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
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
    if synth is None:
        raise HTTPException(status_code=503, detail="TTS not loaded")
    return synth


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()] or [text]


async def _run_sync(fn, *args):
    return await asyncio.get_event_loop().run_in_executor(None, fn, *args)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# Thread pool for all local playback — avoids silent drops from background_tasks
import concurrent.futures as _cf
_speak_pool = _cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="speak")

# ===========================================================================
# /speak  — INSTANT LOCAL PLAYBACK
# ===========================================================================

@app.post("/speak", tags=["speaker"])
async def speak(req: SpeakRequest):
    """⚡ Speak text — uses thread pool for reliable execution (no silent drops)."""
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
                logger.info("⚡ First chunk in %.0fms", (time.perf_counter()-t0)*1000)
                first = False
            speaker.feed(wav_chunk)
        logger.info("✅ /speak done %.2fs", time.perf_counter()-t0)

    _speak_pool.submit(_do_speak)
    return {"status": "speaking", "chars": len(text),
            "preview": text[:80] + ("…" if len(text) > 80 else "")}


@app.post("/speak/tokens", tags=["speaker"])
async def speak_tokens(req: SpeakTokensRequest):
    """
    ⚡ Feed pre-tokenized words → speak IMMEDIATELY without re-buffering.

    FIXED: tokens are joined and synthesized as-is.
    The voice_agent already chunked them (every 3 words / punctuation).
    Re-buffering here caused silent drops on small batches.

    Uses thread pool (not background_tasks) to avoid task queue delays.
    Returns 202 immediately. Playback runs in pool thread.
    """
    s = _require_synth()
    speaker = get_instant_speaker(s.sample_rate)

    # Snapshot values for closure (req may be GC'd)
    tokens    = list(req.tokens)
    speed     = req.speed
    interrupt = req.interrupt

    def _do_speak_tokens():
        t0 = time.perf_counter()

        if interrupt:
            speaker.stop()
            time.sleep(0.05)

        # Join all tokens into one text — agent already chunked appropriately
        text = " ".join(tokens).strip()
        if not text:
            return

        first = True
        for wav_chunk in s.synthesize_streaming(text, speed=speed):
            if first:
                logger.info("⚡ First audio in %.0fms", (time.perf_counter() - t0) * 1000)
                first = False
            speaker.feed(wav_chunk)

        # Do NOT call speaker.flush() here — it blocks and prevents the next
        # batch from starting. The speaker queue drains on its own.
        logger.info("✅ speak_tokens submitted %.0f chars in %.3fs",
                    len(text), time.perf_counter() - t0)

    _speak_pool.submit(_do_speak_tokens)
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
# /synthesize — RETURN BYTES  (for remote clients, unchanged)
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
    """SSE: synthesize token chunks and stream as audio events."""
    s = _require_synth()
    overall_start = time.perf_counter()

    async def generate() -> AsyncIterator[str]:
        buffer: list[str] = []
        chunk_index = 0

        async def flush_buffer():
            nonlocal chunk_index, buffer
            text = " ".join(buffer).strip()
            if not text:
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
            combined = " ".join(buffer)
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
    try:
        speaker_status = "speaking" if get_instant_speaker().is_speaking else "idle"
    except Exception:
        speaker_status = "unavailable"

    return {
        "status": "ok" if s else "loading",
        "model_loaded": s is not None,
        "backend": os.getenv("TTS_BACKEND", "pocket"),
        "sample_rate": s.sample_rate if s else None,
        "startup_time_s": round(_startup_time, 3),
        "speaker_status": speaker_status,
    }


@app.get("/voices", tags=["meta"])
async def voices():
    s = _require_synth()
    return {"voices": s.available_voices(), "backend": os.getenv("TTS_BACKEND", "pocket")}