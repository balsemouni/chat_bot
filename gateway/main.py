"""
Gateway Service — Unified WebSocket + REST Voice Pipeline
=========================================================

Integrates:
  1. WebSocket /ws/{session_id}    — real-time per-session voice pipeline
                                     (STT → LLM → TTS with barge-in, HubSpot logging)
  2. REST /pipeline/audio          — blocking audio → transcript → reply audio
  3. REST /pipeline/audio/stream   — SSE streaming full pipeline
  4. REST /pipeline/text           — text → LLM → optional TTS (SSE)
  5. REST /stt, /llm, /tts         — individual service endpoints
  6. GET  /session/{id}            — conversation history
  7. DEL  /session/{id}            — clear session
  8. GET  /health                  — gateway + downstream health

FIXES APPLIED:
  ✓ Increased TTS chunk size (min_chars=25, max_chars=200) for natural speech
  ✓ Removed commas from punctuation list to prevent mid-sentence breaks
  ✓ Added proper Int16 audio format handling
  ✓ Improved phrase buffering for better prosody

WebSocket protocol (client → server):
  { "type": "audio_chunk",    "data": "<base64 float32 PCM>",
    "is_voice": true|false,   "sample_count": 512 }
  { "type": "utterance_end" }            ← VAD silence timeout on client
  { "type": "barge_in" }                 ← user started talking while AI speaking
  { "type": "end_session" }

WebSocket protocol (server → client):
  { "type": "word",        "data": "hello" }
  { "type": "ai_token",    "data": "Sure" }
  { "type": "audio_chunk", "data": "<base64>", "sample_rate": 22050 }
  { "type": "status",      "state": "listening"|"thinking"|"speaking" }
  { "type": "interrupted" }
  { "type": "session_saved" }

ENV VARS:
  STT_SERVICE_URL  = http://stt-service:8001
  LLM_SERVICE_URL  = http://llm-service:8002
  TTS_SERVICE_URL  = http://tts-service:8003
  HUBSPOT_API_KEY  = <your key>
  SILENCE_MS       = 600
  FAST_SILENCE_MS  = 200
  BARGE_RMS_STRICT = 0.040
  BARGE_MIN_MS     = 400
  TTS_MIN_CHARS    = 25     # Increased for natural flow
  TTS_MAX_CHARS    = 200    # Allow full sentences

Run:
  pip install fastapi uvicorn httpx requests numpy
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from session_manager import SessionManager
from router import PipelineRouter
from hubspot_client import HubSpotClient
from gpu_stats_server import router as gpu_router, GPUStats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s"
)
logger = logging.getLogger("gateway")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    stt_url: str            = "http://stt-service:8001"
    llm_url: str            = "http://llm-service:8002"
    tts_url: str            = "http://tts-service:8003"

    silence_ms: int         = 600
    fast_silence_ms: int    = 200
    barge_rms_min: float    = 0.015
    barge_rms_strict: float = 0.040
    barge_min_ms: int       = 400

    sample_rate: int        = 16000
    tts_sample_rate: int    = 22050
    blocksize: int          = 512

    # FIX: Increased TTS chunk sizes for natural prosody
    tts_min_chars: int      = 25      # Increased from 3 - prevents 1-word chunks
    tts_max_chars: int      = 200     # Increased from 50 - allow full sentences
    tts_punctuation: str    = ".!?;"  # Removed comma - don't break on commas


def _load_config() -> Config:
    return Config(
        stt_url=os.getenv("STT_SERVICE_URL", "http://stt-service:8001"),
        llm_url=os.getenv("LLM_SERVICE_URL", "http://llm-service:8002"),
        tts_url=os.getenv("TTS_SERVICE_URL", "http://tts-service:8003"),
        silence_ms=int(os.getenv("SILENCE_MS", "600")),
        fast_silence_ms=int(os.getenv("FAST_SILENCE_MS", "200")),
        barge_rms_strict=float(os.getenv("BARGE_RMS_STRICT", "0.040")),
        barge_min_ms=int(os.getenv("BARGE_MIN_MS", "400")),
        tts_min_chars=int(os.getenv("TTS_MIN_CHARS", "25")),      # Updated default
        tts_max_chars=int(os.getenv("TTS_MAX_CHARS", "200")),     # Updated default
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class AudioRequest(BaseModel):
    audio_b64: str
    sample_rate: int             = 16000
    session_id: str              = "default"
    system_prompt: Optional[str] = None


class TextRequest(BaseModel):
    text: str
    session_id: str              = "default"
    system_prompt: Optional[str] = None
    tts: bool                    = True


class STTRequest(BaseModel):
    audio_b64: str
    sample_rate: int    = 16000
    ai_is_speaking: bool = False


class LLMRequest(BaseModel):
    text: str
    session_id: str              = "default"
    system_prompt: Optional[str] = None
    stream: bool                 = True


class TTSRequest(BaseModel):
    text: str
    speed: float = 1.0
    stream: bool = True


class PipelineResponse(BaseModel):
    transcript: str
    reply_text: str
    audio_b64: Optional[str] = None
    session_id: str
    stt_ms: float
    llm_ms: float
    tts_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# In-memory REST session store  (separate from WebSocket sessions)
# ─────────────────────────────────────────────────────────────────────────────

class RestSessionStore:
    def __init__(self):
        self._sessions: Dict[str, List[dict]] = {}

    def get_history(self, session_id: str) -> List[dict]:
        return self._sessions.get(session_id, [])

    def append(self, session_id: str, role: str, content: str):
        self._sessions.setdefault(session_id, [])
        self._sessions[session_id].append({"role": role, "content": content})

    def clear(self, session_id: str):
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Phrase buffer (smart TTS chunking) - FIXED for natural speech
# ─────────────────────────────────────────────────────────────────────────────

class PhraseBuffer:
    """
    Accumulates tokens and flushes them as complete phrases.
    
    Rules:
      - Never flush a phrase shorter than min_chars
      - Prefer flushing on sentence-ending punctuation (.!?)
      - If no punctuation, flush at max_chars
      - Never break on commas (removed from punctuation set)
    """
    def __init__(self, min_chars: int = 25, max_chars: int = 200,
                 punctuation: str = ".!?;"):  # No commas
        self.min_chars   = min_chars
        self.max_chars   = max_chars
        self.punctuation = set(punctuation)
        self.buffer      = ""

    def add_token(self, token: str) -> List[str]:
        """Add a token to buffer, return list of complete phrases."""
        if not token:
            return []
        self.buffer += token
        phrases = []

        # If buffer exceeds max length, force a flush
        while len(self.buffer) >= self.max_chars:
            cut = self.max_chars
            # Look backward for sentence-ending punctuation
            for i in range(self.max_chars - 1, self.min_chars - 1, -1):
                if self.buffer[i] in self.punctuation:
                    cut = i + 1
                    break
            phrase = self.buffer[:cut].strip()
            if phrase and len(phrase) >= self.min_chars:
                phrases.append(phrase)
            self.buffer = self.buffer[cut:]

        # Flush on sentence-ending punctuation
        if self.buffer and self.buffer[-1] in self.punctuation:
            if len(self.buffer) >= self.min_chars:
                phrases.append(self.buffer.strip())
                self.buffer = ""

        return phrases

    def flush(self) -> Optional[str]:
        """Flush any remaining buffer content."""
        if self.buffer and len(self.buffer) >= self.min_chars:
            phrase = self.buffer.strip()
            self.buffer = ""
            return phrase
        return None

    def clear(self):
        """Clear the buffer."""
        self.buffer = ""


# ─────────────────────────────────────────────────────────────────────────────
# SSE helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sse(event_type: str, data: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"

def _sse_done() -> str:
    return "data: [DONE]\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# Service clients
# ─────────────────────────────────────────────────────────────────────────────

class STTClient:
    def __init__(self, url: str):
        self.url = url

    async def transcribe(self, audio_b64: str, sample_rate: int,
                         http: httpx.AsyncClient,
                         ai_is_speaking: bool = False) -> tuple[str, float]:
        start = time.perf_counter()
        try:
            resp = await http.post(
                f"{self.url}/transcribe",
                json={"audio_b64": audio_b64, "sample_rate": sample_rate,
                      "ai_is_speaking": ai_is_speaking},
                timeout=15.0,
            )
            data = resp.json()
            text = data.get("text", "").strip()
        except Exception as e:
            logger.error("STT error: %s", e)
            text = ""
        return text, (time.perf_counter() - start) * 1000

    async def transcribe_stream(self, audio_b64: str, sample_rate: int,
                                http: httpx.AsyncClient) -> AsyncGenerator[str, None]:
        try:
            async with http.stream(
                "POST", f"{self.url}/transcribe/stream",
                json={"audio_b64": audio_b64, "sample_rate": sample_rate},
                timeout=15.0,
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        data = json.loads(line[5:])
                        if data.get("type") == "word":
                            yield data.get("word", "")
                        elif data.get("type") in ("done", "end"):
                            break
                    except Exception:
                        continue
        except Exception as e:
            logger.error("STT stream error: %s", e)


class LLMClient:
    def __init__(self, url: str):
        self.url = url

    async def generate_stream(self, text: str, history: List[dict],
                              system_prompt: Optional[str],
                              http: httpx.AsyncClient) -> AsyncGenerator[str, None]:
        payload = {
            "query":      text,
            "session_id": "stream",
            "history":    history,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt

        try:
            async with http.stream(
                "POST", f"{self.url}/generate/stream",
                json=payload,
                timeout=60.0,
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    token = self._extract_token(line)
                    if token:
                        yield token
                    if "data: [DONE]" in line:
                        break
        except Exception as e:
            logger.error("LLM error: %s", e)

    def _extract_token(self, line: str) -> Optional[str]:
        if line.startswith("data: "):
            line = line[6:]
        try:
            data = json.loads(line)
            if "choices" in data:
                return data["choices"][0].get("delta", {}).get("content")
            for key in ("token", "text", "content"):
                if key in data:
                    return data[key]
        except Exception:
            pass
        if line and not line.startswith("{"):
            return line
        return None


class TTSClient:
    def __init__(self, url: str):
        self.url = url

    async def synthesize(self, text: str, http: httpx.AsyncClient,
                         speed: float = 1.0) -> Optional[bytes]:
        try:
            resp = await http.post(
                f"{self.url}/synthesize",
                json={"text": text, "speed": speed},
                timeout=30.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                if "audio_b64" in data:
                    return base64.b64decode(data["audio_b64"])
                return resp.content
        except Exception as e:
            logger.error("TTS error: %s", e)
        return None

    async def synthesize_stream(self, text: str, http: httpx.AsyncClient,
                                speed: float = 1.0) -> AsyncGenerator[bytes, None]:
        try:
            async with http.stream(
                "POST", f"{self.url}/synthesize/stream_pcm",
                json={"text": text, "speed": speed},
                timeout=30.0,
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        data = json.loads(line[5:])
                        if data.get("type") == "audio_chunk" and data.get("data"):
                            yield base64.b64decode(data["data"])
                        elif data.get("type") == "done":
                            break
                    except Exception:
                        continue
        except Exception as e:
            logger.error("TTS stream error: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# REST Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.stt    = STTClient(config.stt_url)
        self.llm    = LLMClient(config.llm_url)
        self.tts    = TTSClient(config.tts_url)

    async def speech_to_text(self, audio_b64: str, sample_rate: int,
                             http: httpx.AsyncClient,
                             ai_is_speaking: bool = False) -> tuple[str, float]:
        return await self.stt.transcribe(audio_b64, sample_rate, http, ai_is_speaking)

    async def generate_text(self, text: str, session_id: str,
                            system_prompt: Optional[str],
                            http: httpx.AsyncClient,
                            rest_sessions: RestSessionStore) -> tuple[str, float]:
        start   = time.perf_counter()
        history = rest_sessions.get_history(session_id)
        tokens  = []
        async for token in self.llm.generate_stream(text, history, system_prompt, http):
            tokens.append(token)
        reply   = "".join(tokens).strip()
        elapsed = (time.perf_counter() - start) * 1000
        rest_sessions.append(session_id, "user", text)
        rest_sessions.append(session_id, "assistant", reply)
        return reply, elapsed

    async def generate_text_stream(self, text: str, session_id: str,
                                   system_prompt: Optional[str],
                                   http: httpx.AsyncClient,
                                   rest_sessions: RestSessionStore) -> AsyncGenerator[str, None]:
        history = rest_sessions.get_history(session_id)
        tokens  = []
        async for token in self.llm.generate_stream(text, history, system_prompt, http):
            tokens.append(token)
            yield token
        reply = "".join(tokens).strip()
        rest_sessions.append(session_id, "user", text)
        rest_sessions.append(session_id, "assistant", reply)

    async def text_to_speech(self, text: str,
                             http: httpx.AsyncClient) -> tuple[Optional[bytes], float]:
        start = time.perf_counter()
        audio = await self.tts.synthesize(text, http)
        return audio, (time.perf_counter() - start) * 1000

    async def run_pipeline(self, audio_b64: str, sample_rate: int,
                           session_id: str, system_prompt: Optional[str],
                           http: httpx.AsyncClient,
                           rest_sessions: RestSessionStore) -> PipelineResponse:
        transcript, stt_ms = await self.speech_to_text(audio_b64, sample_rate, http)
        if not transcript:
            raise HTTPException(422, "Could not transcribe audio")

        reply_text, llm_ms = await self.generate_text(
            transcript, session_id, system_prompt, http, rest_sessions
        )

        audio_bytes, tts_ms = await self.text_to_speech(reply_text, http)
        audio_b64_out = base64.b64encode(audio_bytes).decode() if audio_bytes else None

        return PipelineResponse(
            transcript=transcript,
            reply_text=reply_text,
            audio_b64=audio_b64_out,
            session_id=session_id,
            stt_ms=stt_ms,
            llm_ms=llm_ms,
            tts_ms=tts_ms,
        )

    async def run_pipeline_stream(self, audio_b64: str, sample_rate: int,
                                  session_id: str, system_prompt: Optional[str],
                                  http: httpx.AsyncClient,
                                  rest_sessions: RestSessionStore) -> AsyncGenerator[str, None]:
        t0 = time.perf_counter()

        transcript, stt_ms = await self.speech_to_text(audio_b64, sample_rate, http)
        if not transcript:
            yield _sse("error", {"message": "Could not transcribe audio"})
            return
        yield _sse("transcript", {"text": transcript})

        phrase_buf = PhraseBuffer(self.config.tts_min_chars, self.config.tts_max_chars,
                                  self.config.tts_punctuation)
        llm_start  = time.perf_counter()
        tts_tasks: List[asyncio.Task] = []

        async for token in self.generate_text_stream(
            transcript, session_id, system_prompt, http, rest_sessions
        ):
            yield _sse("token", {"text": token})
            for phrase in phrase_buf.add_token(token):
                logger.debug(f"TTS phrase queued: {phrase[:50]}...")
                tts_tasks.append(asyncio.create_task(self.tts.synthesize(phrase, http)))

        final_phrase = phrase_buf.flush()
        if final_phrase:
            logger.debug(f"TTS final phrase: {final_phrase[:50]}...")
            tts_tasks.append(asyncio.create_task(self.tts.synthesize(final_phrase, http)))

        llm_ms    = (time.perf_counter() - llm_start) * 1000
        tts_start = time.perf_counter()

        for task in tts_tasks:
            audio_bytes = await task
            if audio_bytes:
                yield _sse("audio_chunk", {"data": base64.b64encode(audio_bytes).decode()})

        tts_ms = (time.perf_counter() - tts_start) * 1000

        yield _sse("done", {
            "stt_ms":   round(stt_ms, 1),
            "llm_ms":   round(llm_ms, 1),
            "tts_ms":   round(tts_ms, 1),
            "total_ms": round((time.perf_counter() - t0) * 1000, 1),
        })

    def should_barge_in(self, audio_b64: str, duration_ms: float) -> bool:
        try:
            raw   = base64.b64decode(audio_b64)
            audio = np.frombuffer(raw, dtype=np.float32)
            rms   = float(np.sqrt(np.mean(audio ** 2) + 1e-10))
            return rms >= self.config.barge_rms_strict and duration_ms >= self.config.barge_min_ms
        except Exception:
            return False

    def is_voice(self, audio_b64: str) -> bool:
        try:
            raw   = base64.b64decode(audio_b64)
            audio = np.frombuffer(raw, dtype=np.float32)
            rms   = float(np.sqrt(np.mean(audio ** 2) + 1e-10))
            return rms >= self.config.barge_rms_min
        except Exception:
            return False

    def detect_utterance_end(self, partial_text: str, silence_elapsed_ms: float) -> bool:
        threshold = self.config.silence_ms
        if partial_text and re.search(r"[.!?]\s*$", partial_text.strip()):
            threshold = self.config.fast_silence_ms
        return silence_elapsed_ms >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# App + lifespan
# ─────────────────────────────────────────────────────────────────────────────

cfg: Config                 = None   # type: ignore
orchestrator: Orchestrator  = None   # type: ignore
_http: httpx.AsyncClient    = None   # type: ignore
_rest_sessions: RestSessionStore = None   # type: ignore
_session_manager: SessionManager = None   # type: ignore
_hubspot: HubSpotClient     = None   # type: ignore
_gpu_stats: GPUStats        = None   # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cfg, orchestrator, _http, _rest_sessions, _session_manager, _hubspot

    cfg             = _load_config()
    orchestrator    = Orchestrator(cfg)
    _http           = httpx.AsyncClient(timeout=60.0)
    _rest_sessions  = RestSessionStore()
    _session_manager = SessionManager()
    _hubspot        = HubSpotClient()
    _gpu_stats      = GPUStats()

    logger.info("=" * 60)
    logger.info("🚀 Gateway starting with config:")
    logger.info(f"  🎤 STT: {cfg.stt_url}")
    logger.info(f"  🧠 LLM: {cfg.llm_url}")
    logger.info(f"  🔊 TTS: {cfg.tts_url}")
    logger.info(f"  📝 TTS chunking: min={cfg.tts_min_chars} chars, max={cfg.tts_max_chars} chars")
    logger.info(f"  🖥️  GPU monitoring: {'enabled' if _gpu_stats._nvml_ok else 'no GPU detected'}")
    logger.info("=" * 60)
    yield

    await _http.aclose()
    logger.info("👋 Gateway shut down")


app = FastAPI(
    title="Voice Agent Gateway",
    version="3.0.0",
    description="Unified REST + WebSocket voice pipeline gateway.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── GPU monitoring endpoints (served under /gpu/*)
app.include_router(gpu_router, prefix="/gpu")


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    results = {
        "gateway":  "ok",
        "ws_sessions": len(_session_manager._sessions),
        "rest_sessions": len(_rest_sessions.list_sessions()),
        "services": {},
    }
    for name, url in [("stt", cfg.stt_url), ("llm", cfg.llm_url), ("tts", cfg.tts_url)]:
        try:
            r = await _http.get(f"{url}/health", timeout=3.0)
            results["services"][name] = r.json() if r.status_code == 200 else f"HTTP {r.status_code}"
        except Exception as e:
            results["services"][name] = f"unreachable: {e}"
    return results


# ─────────────────────────────────────────────────────────────────────────────
# REST — individual service endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/stt")
async def stt_endpoint(req: STTRequest):
    text, ms = await orchestrator.speech_to_text(
        req.audio_b64, req.sample_rate, _http, req.ai_is_speaking
    )
    return {"text": text, "duration_ms": round(ms, 1)}


@app.post("/stt/stream")
async def stt_stream_endpoint(req: STTRequest):
    async def _gen():
        async for word in orchestrator.stt.transcribe_stream(
            req.audio_b64, req.sample_rate, _http
        ):
            yield _sse("word", {"word": word})
        yield _sse("done", {})
    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.post("/llm")
async def llm_endpoint(req: LLMRequest):
    if not req.stream:
        reply, ms = await orchestrator.generate_text(
            req.text, req.session_id, req.system_prompt, _http, _rest_sessions
        )
        return {"text": reply, "session_id": req.session_id, "ms": round(ms, 1)}

    async def _gen():
        async for token in orchestrator.generate_text_stream(
            req.text, req.session_id, req.system_prompt, _http, _rest_sessions
        ):
            yield _sse("token", {"text": token})
        yield _sse_done()
    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.post("/tts")
async def tts_endpoint(req: TTSRequest):
    if not req.stream:
        audio_bytes, ms = await orchestrator.text_to_speech(req.text, _http)
        if not audio_bytes:
            raise HTTPException(502, "TTS service failed")
        return {"audio_b64": base64.b64encode(audio_bytes).decode(), "ms": round(ms, 1)}

    async def _gen():
        async for chunk in orchestrator.tts.synthesize_stream(req.text, _http, req.speed):
            yield _sse("audio_chunk", {"data": base64.b64encode(chunk).decode()})
        yield _sse("done", {})
    return StreamingResponse(_gen(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────────────────────
# REST — full pipeline
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/pipeline/audio", response_model=PipelineResponse)
async def pipeline_audio(req: AudioRequest):
    return await orchestrator.run_pipeline(
        req.audio_b64, req.sample_rate, req.session_id, req.system_prompt,
        _http, _rest_sessions
    )


@app.post("/pipeline/audio/stream")
async def pipeline_audio_stream(req: AudioRequest):
    async def _gen():
        async for event in orchestrator.run_pipeline_stream(
            req.audio_b64, req.sample_rate, req.session_id, req.system_prompt,
            _http, _rest_sessions
        ):
            yield event
    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.post("/pipeline/text")
async def pipeline_text(req: TextRequest):
    async def _gen():
        phrase_buf = PhraseBuffer(cfg.tts_min_chars, cfg.tts_max_chars, cfg.tts_punctuation)
        tts_tasks: List[asyncio.Task] = []

        async for token in orchestrator.generate_text_stream(
            req.text, req.session_id, req.system_prompt, _http, _rest_sessions
        ):
            yield _sse("token", {"text": token})
            if req.tts:
                for phrase in phrase_buf.add_token(token):
                    tts_tasks.append(
                        asyncio.create_task(orchestrator.tts.synthesize(phrase, _http))
                    )

        if req.tts:
            final = phrase_buf.flush()
            if final:
                tts_tasks.append(
                    asyncio.create_task(orchestrator.tts.synthesize(final, _http))
                )
            for task in tts_tasks:
                audio_bytes = await task
                if audio_bytes:
                    yield _sse("audio_chunk", {"data": base64.b64encode(audio_bytes).decode()})

        yield _sse("done", {})
    return StreamingResponse(_gen(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────────────────────
# REST — session management
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    return {
        "session_id": session_id,
        "history":    _rest_sessions.get_history(session_id),
    }


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    _rest_sessions.clear(session_id)
    return {"cleared": session_id}


@app.get("/sessions")
async def list_sessions():
    return {
        "rest_sessions": _rest_sessions.list_sessions(),
        "ws_sessions":   _session_manager.list_all(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — real-time per-session voice pipeline (with HubSpot logging)
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Real-time voice session. One connection = one voice session.

    Lifecycle:
      1. Client connects → session created, HubSpot session started
      2. Client streams VAD-gated audio chunks (is_voice=true only)
      3. Gateway routes each chunk through STT → LLM → TTS pipeline
      4. Client signals utterance_end → pipeline flushes remaining audio
      5. Client signals end_session   → transcript saved to HubSpot
      6. Disconnect → auto-save transcript to HubSpot
    """
    await websocket.accept()
    logger.info("WS session connected: %s", session_id)

    session = _session_manager.create(session_id)
    router  = PipelineRouter(
        session, cfg.stt_url, cfg.llm_url, cfg.tts_url,
        websocket, hubspot=_hubspot,
    )
    _hubspot.start_session(session_id)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Malformed frame from %s", session_id)
                continue

            ftype = frame.get("type")

            if ftype == "audio_chunk":
                audio_b64    = frame.get("data", "")
                is_voice     = frame.get("is_voice", True)
                sample_count = frame.get("sample_count", 512)
                if audio_b64:
                    await router.handle_audio(
                        audio_b64,
                        sample_count=sample_count,
                        is_voice=True,  # Force true - let router decide
                    )

            elif ftype == "utterance_end":
                await router.handle_utterance_end()

            elif ftype == "barge_in":
                if router._ai_speaking:
                    await router._interrupt()

            elif ftype == "end_session":
                transcript = session.get_full_transcript()
                _hubspot.end_session(session_id, transcript)
                await websocket.send_json({"type": "session_saved"})
                logger.info("Session ended cleanly: %s", session_id)
                break

            # Legacy support: single audio_b64 field (gateway.py style)
            elif ftype == "audio":
                audio_b64    = frame.get("audio_b64", "")
                sample_count = frame.get("sample_count", 512)
                if audio_b64:
                    await router.handle_audio(audio_b64, sample_count=sample_count, is_voice=True)

            elif ftype == "ping":
                await websocket.send_json({"type": "pong"})

            elif ftype == "close":
                break

            else:
                logger.debug("Unknown frame type %r from %s", ftype, session_id)

    except WebSocketDisconnect:
        logger.info("WS session disconnected: %s", session_id)
        transcript = session.get_full_transcript()
        _hubspot.end_session(session_id, transcript)

    except Exception as e:
        logger.error("WS session error %s: %s", session_id, e)
        transcript = session.get_full_transcript()
        _hubspot.end_session(session_id, transcript)

    finally:
        _session_manager.remove(session_id)
        await router.cancel_all()


# ─────────────────────────────────────────────────────────────────────────────
# Debug — WebSocket transcript endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/transcript")
async def get_ws_transcript(session_id: str):
    session = _session_manager.get(session_id)
    if not session:
        return {"error": "session not found"}
    return {
        "session_id": session_id,
        "transcript": session.get_full_transcript(),
        "turns":      len(session.history),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dev entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)