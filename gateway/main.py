"""
Gateway Service — WebSocket hub, pipeline orchestrator, session manager
=======================================================================

Mirrors the monolith's UltraLowLatencyVoiceAgentWithCAG structure:
  - audio_capture_thread  → client sends audio_chunk frames
  - vad_processor_thread  → client sends is_voice flag per chunk (VAD runs client-side)
                            OR client sends utterance_end when silence detected
  - _process_utterance    → router.handle_audio / router.handle_utterance_end
  - _get_cag_response     → router._pipeline (STT → LLM → TTS)
  - HubSpot logging       → hubspot_client (same as UIHandlerWithHubSpot)

WebSocket protocol (client → server):
  { "type": "audio_chunk",    "data": "<base64 float32 PCM>",
    "is_voice": true|false,   "sample_count": 512 }
  { "type": "utterance_end" }        ← VAD silence timeout on client
  { "type": "barge_in" }             ← user started talking while AI speaking
  { "type": "end_session" }

WebSocket protocol (server → client):
  { "type": "word",        "data": "hello" }          ← STT word stream
  { "type": "ai_token",    "data": "Sure" }           ← LLM token stream
  { "type": "audio_chunk", "data": "<base64>",
    "sample_rate": 22050 }                             ← TTS audio
  { "type": "status",      "state": "listening"|"thinking"|"speaking" }
  { "type": "interrupted" }                            ← barge-in confirmed
  { "type": "session_saved" }                          ← HubSpot save done

All HubSpot logging lives here. The router receives hubspot so it can
log utterances in real-time (same as UIHandlerWithHubSpot.add_utterance).
"""

import json
import logging
import os

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from session_manager import SessionManager
from router import PipelineRouter
from hubspot_client import HubSpotClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s"
)
logger = logging.getLogger("gateway")

# ── Service URLs ──────────────────────────────────────────────────────────────
STT_URL = os.getenv("STT_SERVICE_URL", "http://stt-service:8001")
LLM_URL = os.getenv("LLM_SERVICE_URL", "http://llm-service:8002")
TTS_URL = os.getenv("TTS_SERVICE_URL", "http://tts-service:8003")

# ── Singletons ────────────────────────────────────────────────────────────────
session_manager = SessionManager()
hubspot = HubSpotClient()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Voice Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint — one connection = one voice session
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Main voice session handler.

    Lifecycle:
      1. Client connects → session created, HubSpot session started
      2. Client streams VAD-gated audio chunks
      3. Gateway routes each chunk through STT → LLM → TTS pipeline
      4. Client signals utterance_end → pipeline flushes remaining audio
      5. Client signals end_session → transcript saved to HubSpot
      6. Disconnect → auto-save transcript to HubSpot
    """
    await websocket.accept()
    logger.info("Session connected: %s", session_id)

    session = session_manager.create(session_id)
    router = PipelineRouter(
        session, STT_URL, LLM_URL, TTS_URL,
        websocket, hubspot=hubspot          # wire HubSpot into router for real-time logging
    )
    hubspot.start_session(session_id)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Malformed frame from %s", session_id)
                continue

            ftype = frame.get("type")

            # ── Voice audio chunk ────────────────────────────────────────────
            if ftype == "audio_chunk":
                audio_b64    = frame.get("data", "")
                is_voice     = frame.get("is_voice", True)   # default True for legacy clients
                sample_count = frame.get("sample_count", 512)

                if audio_b64:
                    await router.handle_audio(
                        audio_b64,
                        sample_count=sample_count,
                        is_voice=is_voice,
                    )

            # ── Utterance end (VAD silence timeout on client) ────────────────
            elif ftype == "utterance_end":
                # Flush any remaining buffered audio and fire final pipeline.
                # Mirrors: silence_duration > silence_threshold_ms branch.
                await router.handle_utterance_end()

            # ── Explicit barge-in signal from client ─────────────────────────
            elif ftype == "barge_in":
                # Client detected user started talking while AI was speaking.
                # router.handle_audio will also barge-in automatically,
                # but this explicit signal cancels even before audio arrives.
                if router._ai_speaking:
                    await router._interrupt()

            # ── End session — save to HubSpot ────────────────────────────────
            elif ftype == "end_session":
                transcript = session.get_full_transcript()
                hubspot.end_session(session_id, transcript)
                await websocket.send_json({"type": "session_saved"})
                logger.info("Session ended cleanly: %s", session_id)
                break

            else:
                logger.debug("Unknown frame type %r from %s", ftype, session_id)

    except WebSocketDisconnect:
        logger.info("Session disconnected: %s", session_id)
        transcript = session.get_full_transcript()
        hubspot.end_session(session_id, transcript)

    except Exception as e:
        logger.error("Session error %s: %s", session_id, e)
        transcript = session.get_full_transcript()
        hubspot.end_session(session_id, transcript)

    finally:
        session_manager.remove(session_id)
        await router.cancel_all()


# ─────────────────────────────────────────────────────────────────────────────
# Health + debug endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":   "ok",
        "sessions": len(session_manager._sessions),
        "services": {
            "stt": STT_URL,
            "llm": LLM_URL,
            "tts": TTS_URL,
        },
    }


@app.get("/sessions")
async def list_sessions():
    """Debug: list all active sessions with turn counts."""
    return session_manager.list_all()


@app.get("/sessions/{session_id}/transcript")
async def get_transcript(session_id: str):
    """Debug: get full transcript for a session."""
    session = session_manager.get(session_id)
    if not session:
        return {"error": "session not found"}
    return {
        "session_id": session_id,
        "transcript": session.get_full_transcript(),
        "turns":      len(session.history),
    }