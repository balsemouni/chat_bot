"""
main.py — STT Microservice (Full GPU, Zero-Latency Edition)
============================================================

NEW vs previous version:
  - WebSocket endpoint /ws/transcribe — true real-time bidirectional streaming
    (eliminates HTTP overhead for ultra-low latency path)
  - /transcribe/stream  — SSE (legacy, kept for compatibility)
  - /transcribe/partial — 800ms partial results (SSE)
  - /transcribe         — blocking (legacy)
  - Whisper warmed up at startup (CUDA kernel pre-compilation)
  - GPU memory pre-allocated at lifespan startup
  - CORS enabled for cross-origin WebSocket clients

WebSocket protocol (JSON messages):
  Client → Server:
    {"audio_b64": "<base64 float32>", "sample_rate": 16000, "ai_is_speaking": false}
    {"cmd": "end"}   ← signal end of stream
  
  Server → Client:
    {"type": "word",        "word": "hello"}
    {"type": "done"}
    {"type": "silence"}
    {"type": "ai_filtered"}
    {"type": "error",       "msg": "..."}
"""

import base64
import json
import os
import logging

import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from asr import StreamingSpeechRecognizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("main")


class TranscribeRequest(BaseModel):
    audio_b64:      str
    sample_rate:    int  = 16000
    ai_is_speaking: bool = False


# ── Global state ─────────────────────────────────────────────────────────────
asr: StreamingSpeechRecognizer = None
_whisper_model: str = "base.en"


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr, _whisper_model

    device         = os.getenv("DEVICE", "cuda")
    ai_model       = os.getenv("AI_DETECTOR_MODEL_PATH")
    _whisper_model = os.getenv("WHISPER_MODEL", "base.en")
    compute_type   = os.getenv("WHISPER_COMPUTE_TYPE")     # None = auto
    num_workers    = int(os.getenv("WHISPER_WORKERS", "4"))

    logger.info("🚀 Loading Whisper (%s) on %s…", _whisper_model, device)
    asr = StreamingSpeechRecognizer(
        model_size=_whisper_model,
        device=device,
        ai_detector_model_path=ai_model,
        enable_ai_filtering=bool(ai_model),
        compute_type=compute_type,
        num_workers=num_workers,
    )
    logger.info("✅ STT service ready")
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="STT Service — Full GPU Zero Latency", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Helpers
# =============================================================================

def _decode_audio(audio_b64: str) -> np.ndarray:
    """Decode base64 float32 PCM audio."""
    raw = base64.b64decode(audio_b64)
    remainder = len(raw) % 4
    if remainder:
        raw = raw[:-remainder]
    return np.frombuffer(raw, dtype=np.float32).copy()


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


# =============================================================================
# WebSocket — TRUE real-time zero-latency endpoint
# =============================================================================

@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    """
    WebSocket streaming endpoint.

    Eliminates HTTP connection overhead per request.
    Ideal for continuous real-time transcription.

    Protocol:
      Client sends JSON:  {"audio_b64": "...", "sample_rate": 16000, "ai_is_speaking": false}
      Server streams JSON: {"type": "word", "word": "..."} for each word
                           {"type": "done"} at end of segment
                           {"type": "silence"} if no speech detected
                           {"type": "ai_filtered"} if AI voice detected
    """
    await websocket.accept()
    logger.info("WebSocket connected")

    try:
        while True:
            raw_msg = await websocket.receive_text()
            msg = json.loads(raw_msg)

            # Control command
            if msg.get("cmd") == "end":
                await websocket.send_text(json.dumps({"type": "done"}))
                break

            # Barge-in fast path
            if msg.get("ai_is_speaking", False):
                await websocket.send_text(json.dumps({"type": "ai_filtered"}))
                await websocket.send_text(json.dumps({"type": "done"}))
                continue

            # Decode audio
            audio = _decode_audio(msg["audio_b64"])
            sample_rate = msg.get("sample_rate", 16000)

            if len(audio) == 0:
                await websocket.send_text(json.dumps({"type": "silence"}))
                await websocket.send_text(json.dumps({"type": "done"}))
                continue

            # Stream words directly over WebSocket — zero buffering
            word_count = 0
            for word in asr.transcribe_streaming(audio, sample_rate):
                word = word.strip()
                if word:
                    word_count += 1
                    await websocket.send_text(
                        json.dumps({"type": "word", "word": word})
                    )

            event = "silence" if word_count == 0 else "done"
            await websocket.send_text(json.dumps({"type": event}))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await websocket.send_text(json.dumps({"type": "error", "msg": str(e)}))
        except Exception:
            pass


# =============================================================================
# /transcribe/stream — SSE word-by-word (legacy, partial + full audio)
# =============================================================================

@app.post("/transcribe/stream")
async def transcribe_stream(req: TranscribeRequest):
    async def generate():
        if req.ai_is_speaking:
            yield _sse({"type": "ai_filtered"})
            yield _sse({"type": "done"})
            return

        audio = _decode_audio(req.audio_b64)

        if len(audio) == 0:
            yield _sse({"type": "silence"})
            yield _sse({"type": "done"})
            return

        word_count = 0
        for word in asr.transcribe_streaming(audio, req.sample_rate):
            word = word.strip()
            if word:
                word_count += 1
                yield _sse({"type": "word", "word": word})

        yield _sse({"type": "silence" if word_count == 0 else "done"})

    return StreamingResponse(generate(), media_type="text/event-stream")


# =============================================================================
# /transcribe/partial — 800ms mid-utterance chunks (SSE)
# =============================================================================

@app.post("/transcribe/partial")
async def transcribe_partial(req: TranscribeRequest):
    """Called every ~800ms while user is still speaking."""
    return await transcribe_stream(req)


# =============================================================================
# /transcribe — blocking JSON (legacy)
# =============================================================================

@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    if req.ai_is_speaking:
        return {"text": "", "reason": "ai_speaking"}

    audio = _decode_audio(req.audio_b64)
    text  = asr.transcribe(audio, req.sample_rate)
    return {"text": text}


# =============================================================================
# /health
# =============================================================================

@app.get("/health")
async def health():
    import torch
    return {
        "status":       "ok" if asr is not None else "loading",
        "asr":          asr is not None,
        "asr_model":    _whisper_model,
        "sample_rate":  16000,
        "mode":         "full-gpu-zero-latency",
        "cuda":         torch.cuda.is_available(),
        "gpu":          torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "endpoints": {
            "realtime":  "ws://host/ws/transcribe  (WebSocket, recommended)",
            "streaming": "POST /transcribe/stream  (SSE)",
            "partial":   "POST /transcribe/partial (SSE, 800ms chunks)",
            "blocking":  "POST /transcribe         (JSON)",
        },
    }