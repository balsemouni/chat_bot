"""
main.py — STT Microservice  (Full GPU · Zero-Latency Edition)
=============================================================

Endpoints
---------
  WS   /ws/transcribe          real-time bidirectional streaming  ← recommended
  POST /transcribe/stream      SSE word-by-word                   ← legacy
  POST /transcribe/partial     SSE 800 ms partial results         ← legacy
  POST /transcribe             blocking JSON                      ← legacy
  GET  /health                 status + GPU info

WebSocket protocol (JSON)
--------------------------
Client → Server:
  {"audio_b64": "<base64 float32 PCM>", "sample_rate": 16000, "ai_is_speaking": false}
  {"cmd": "end"}

Server → Client:
  {"type": "word",        "word": "hello"}
  {"type": "done"}
  {"type": "silence"}
  {"type": "ai_filtered"}
  {"type": "error",       "msg": "..."}

Environment variables
---------------------
  DEVICE                  cuda | cpu                  default: cuda
  WHISPER_MODEL           tiny.en|base.en|small.en…   default: base.en
  WHISPER_COMPUTE_TYPE    float16 | int8 | auto        default: auto
  WHISPER_WORKERS         integer                      default: 4
  AI_DETECTOR_MODEL_PATH  path to .pth checkpoint      default: (none)
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

from asr import StreamingSpeechRecognizer   # ← local module

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger("stt.main")


# ── Request schema ────────────────────────────────────────────────────────────

class TranscribeRequest(BaseModel):
    audio_b64:      str
    sample_rate:    int  = 16000
    ai_is_speaking: bool = False


# ── Global state ──────────────────────────────────────────────────────────────

asr: StreamingSpeechRecognizer = None
_whisper_model: str = "base.en"


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr, _whisper_model

    device          = os.getenv("DEVICE", "cuda")
    ai_model        = os.getenv("AI_DETECTOR_MODEL_PATH")          # None if unset
    _whisper_model  = os.getenv("WHISPER_MODEL", "base.en")
    compute_type    = os.getenv("WHISPER_COMPUTE_TYPE")            # None → auto
    num_workers     = int(os.getenv("WHISPER_WORKERS", "4"))

    logger.info("🚀 Starting STT service — model=%s  device=%s", _whisper_model, device)

    asr = StreamingSpeechRecognizer(
        model_size             = _whisper_model,
        device                 = device,
        ai_detector_model_path = ai_model,
        enable_ai_filtering    = bool(ai_model),
        compute_type           = compute_type,   # ← now accepted by asr.py
        num_workers            = num_workers,    # ← now accepted by asr.py
    )

    logger.info("✅ STT service ready")
    yield
    # (nothing to clean up — Whisper has no explicit close)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="STT Service — Full GPU Zero Latency",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_audio(audio_b64: str) -> np.ndarray:
    """Decode a base64-encoded float32 PCM payload."""
    raw = base64.b64decode(audio_b64)
    # Trim stray bytes so frombuffer never raises
    remainder = len(raw) % 4
    if remainder:
        raw = raw[:-remainder]
    return np.frombuffer(raw, dtype=np.float32).copy()


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


# ── WebSocket — zero-latency real-time endpoint ───────────────────────────────

@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    """
    Persistent WebSocket for continuous real-time transcription.
    Eliminates per-request HTTP overhead.
    """
    await websocket.accept()
    logger.info("WebSocket connected  [%s]", websocket.client)

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            # ── control command ──────────────────────────────────────────
            if msg.get("cmd") == "end":
                await websocket.send_text(json.dumps({"type": "done"}))
                break

            # ── fast-path: AI is speaking → ignore mic input ─────────────
            if msg.get("ai_is_speaking", False):
                await websocket.send_text(json.dumps({"type": "ai_filtered"}))
                await websocket.send_text(json.dumps({"type": "done"}))
                continue

            # ── decode audio ─────────────────────────────────────────────
            audio = _decode_audio(msg["audio_b64"])
            sample_rate = msg.get("sample_rate", 16000)

            if len(audio) == 0:
                await websocket.send_text(json.dumps({"type": "silence"}))
                await websocket.send_text(json.dumps({"type": "done"}))
                continue

            # ── stream words ─────────────────────────────────────────────
            word_count = 0
            for word in asr.transcribe_streaming(audio, sample_rate):
                if word:
                    word_count += 1
                    await websocket.send_text(
                        json.dumps({"type": "word", "word": word})
                    )

            final = "done" if word_count > 0 else "silence"
            await websocket.send_text(json.dumps({"type": final}))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected  [%s]", websocket.client)
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        try:
            await websocket.send_text(json.dumps({"type": "error", "msg": str(exc)}))
        except Exception:
            pass


# ── POST /transcribe/stream — SSE word-by-word ────────────────────────────────

@app.post("/transcribe/stream")
async def transcribe_stream(req: TranscribeRequest):
    """SSE endpoint — streams one JSON event per word."""

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
            if word:
                word_count += 1
                yield _sse({"type": "word", "word": word})

        yield _sse({"type": "done" if word_count > 0 else "silence"})

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── POST /transcribe/partial — 800 ms mid-utterance SSE ──────────────────────

@app.post("/transcribe/partial")
async def transcribe_partial(req: TranscribeRequest):
    """Alias of /transcribe/stream — called every ~800 ms while user speaks."""
    return await transcribe_stream(req)


# ── POST /transcribe — blocking JSON (legacy) ─────────────────────────────────

@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    if req.ai_is_speaking:
        return {"text": "", "reason": "ai_speaking"}

    audio = _decode_audio(req.audio_b64)
    text  = asr.transcribe(audio, req.sample_rate)
    return {"text": text}


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    import torch
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "status":      "ok" if asr is not None else "loading",
        "asr":         asr is not None,
        "asr_model":   _whisper_model,
        "sample_rate": 16000,
        "cuda":        torch.cuda.is_available(),
        "gpu":         gpu_name,
        "endpoints": {
            "realtime":  "ws://<host>/ws/transcribe   (WebSocket — recommended)",
            "streaming": "POST /transcribe/stream     (SSE)",
            "partial":   "POST /transcribe/partial    (SSE, 800 ms chunks)",
            "blocking":  "POST /transcribe            (JSON)",
        },
    }