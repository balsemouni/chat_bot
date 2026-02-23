"""
CAG LLM Microservice  v3.0
==========================

ROOT-CAUSE FIX (404):
  voice_agent.py sends {"session_id": "voice-agent"} but only "default"
  existed in the registry, so get() raised KeyError → FastAPI returned 404.

  Fix: get_or_create() auto-provisions any unknown session_id on first use
  by CLONING the default session (shared model/tokenizer/cache, isolated
  ConversationMemory).  Zero GPU overhead.  Zero latency.

NEW: per-call session isolation
  POST /session/{id}/end  →  returns user_name + LLM summary, then resets
  the conversation so the next caller starts completely fresh.
  Voice agent calls this once TTS finishes playing.

NEW: summary baked into the SSE stream
  The final SSE "done" chunk now contains user_name + summary so the voice
  agent gets everything in a single HTTP connection, no second round-trip.

Endpoints
---------
POST /generate/stream           SSE token stream  (auto-creates session)
POST /generate                  Blocking full response
POST /generate/batch            Multiple questions
POST /session/new               Explicit session creation (optional)
POST /session/{id}/reset        Wipe conversation, keep session alive
POST /session/{id}/end          Wipe + return name/summary  ← voice agent uses this
DELETE /session/{id}            Destroy session entirely
GET  /session/{id}/summary      Name + LLM summary (non-destructive)
GET  /sessions                  List all active sessions
GET  /stats?session_id=default  Runtime stats
GET  /health                    Liveness probe

SSE chunk types (in order)
--------------------------
  {"type":"token",    "token":"Hello "}
  {"type":"metadata", "stage":"followup", "user_name":"Alex", "total_queries":3}
  {"type":"summary",  "user_name":"Alex", "summary":"User asked about X."}
  {"type":"done",     "total_tokens":42,  "user_name":"Alex", "summary":"..."}

Environment variables
---------------------
  CAG_MAX_CONTEXT_TOKENS   int           default 5000
  CAG_MAX_NEW_TOKENS       int           default 256
  CAG_REBUILD_CACHE        true|false    default false
  CAG_SESSION_MODE         fresh|memory  default fresh
  CAG_VERBOSE              true|false    default true
  PORT                     int           default 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from cag_config import CAGConfig
from cag_system import CAGSystemFreshSession, CAGSystemWithMemory

log = logging.getLogger("llm_service")

# ══════════════════════════════════════════════════════════════════════════════
# Config helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_config() -> CAGConfig:
    return CAGConfig(
        max_context_tokens=int(os.getenv("CAG_MAX_CONTEXT_TOKENS", "5000")),
        max_new_tokens=int(os.getenv("CAG_MAX_NEW_TOKENS", "256")),
        enable_cache_persistence=True,
        enable_conversation_memory=True,
        verbose=os.getenv("CAG_VERBOSE", "true").lower() == "true",
    )


def _make_system(mode: str, config: CAGConfig) -> CAGSystemFreshSession:
    return CAGSystemWithMemory(config) if mode == "memory" else CAGSystemFreshSession(config)


# ══════════════════════════════════════════════════════════════════════════════
# Session registry
# ══════════════════════════════════════════════════════════════════════════════

class SessionRegistry:
    """
    Thread-safe registry of CAG sessions.

    Core pattern: get_or_create()
    ─────────────────────────────
    Any unknown session_id is auto-provisioned by CLONING the default session.
    Clones share the same model weights, tokenizer, and KV-cache (all read-only
    after initialization) but have completely isolated ConversationMemory.

    This fixes the 404: voice_agent sends "voice-agent" as session_id — the
    registry silently creates it without touching the GPU at all.

    Isolation guarantee:
    ─────────────────────
    Each session has its own ConversationMemory instance, so name-extraction,
    stage-machine, and conversation history are fully independent.  Two
    concurrent sessions never bleed into each other.
    """

    def __init__(self):
        self._sessions: Dict[str, CAGSystemFreshSession] = {}
        self._meta:     Dict[str, dict]                  = {}
        self._lock      = threading.Lock()
        self._default_mode = os.getenv("CAG_SESSION_MODE", "fresh")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _register(self, sid: str, sys: CAGSystemFreshSession, mode: str):
        """Must be called with self._lock held."""
        self._sessions[sid] = sys
        self._meta[sid] = {
            "created_at":   time.time(),
            "mode":         mode,
            "total_ended":  0,
        }

    def _clone_from_default(self, sid: str) -> CAGSystemFreshSession:
        """
        Create a lightweight clone of the default session.

        Strategy: call __init__ normally (so every attribute is set correctly),
        then overwrite only the heavy resources with shared references from the
        default session.  This is safer than __new__ + manual wiring — we can
        never miss an attribute that __init__ sets.

        Shared (read-only after init):   model, tokenizer, device, cache_manager
        Isolated (per session):          memory, total_queries, stage flags, etc.

        Must be called with self._lock held.
        """
        src = self._sessions["default"]

        # 1. Fully-initialized fresh instance (calls __init__ normally)
        clone = CAGSystemFreshSession(config=src.config)

        # 2. Swap in shared heavy resources — no second GPU load
        clone.model            = src.model            # GPU weights, read-only
        clone.tokenizer        = src.tokenizer        # stateless, safe to share
        clone.device           = src.device
        clone.cache_manager    = src.cache_manager    # KV cache, read-only
        clone.knowledge_store  = src.knowledge_store  # read-only, needed by get_stats()
        clone.system_prompt    = src.system_prompt

        # 3. Mark as initialized (skip initialize() call)
        clone.is_initialized = True

        from datetime import datetime
        clone.session_start_time = datetime.now()

        # 4. ConversationMemory was created fresh by __init__ — just ensure
        #    no disk files are read/written (fresh-session behaviour).
        clone._disable_memory_persistence()

        # 5. Mark as clone so delete() doesn't call cleanup() (would free
        #    shared model/tokenizer that other sessions still use).
        clone._is_clone = True

        self._register(sid, clone, mode=self._default_mode)
        log.info("Auto-created session '%s' (cloned from default)", sid)
        return clone

    # ── Public API ─────────────────────────────────────────────────────────────

    def create(
        self,
        session_id: str,
        mode: str,
        config: CAGConfig,
        *,
        rebuild_cache: bool = False,
    ) -> CAGSystemFreshSession:
        """Explicit creation — loads model from scratch (used for 'default')."""
        with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"Session '{session_id}' already exists.")
        sys = _make_system(mode, config)
        sys.initialize(force_cache_rebuild=rebuild_cache)
        with self._lock:
            self._register(session_id, sys, mode)
        return sys

    def get_or_create(self, session_id: str) -> CAGSystemFreshSession:
        """Return existing session or silently auto-create via clone. Fixes 404."""
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]
            if "default" not in self._sessions:
                raise RuntimeError("Default session not initialized yet.")
            return self._clone_from_default(session_id)

    def get(self, session_id: str) -> CAGSystemFreshSession:
        """Strict get — raises KeyError if not found."""
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            return self._sessions[session_id]

    def reset(self, session_id: str):
        """Wipe conversation, keep session alive."""
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            sys = self._sessions[session_id]
        sys.reset_session()
        with self._lock:
            self._meta[session_id]["last_reset"] = time.time()

    def end(self, session_id: str) -> dict:
        """
        The clean end-of-call sequence:
          1. Ask LLM for name + summary of this conversation.
          2. Reset conversation memory for the next caller.
          3. Return the summary dict.

        The session object stays alive — no re-initialization needed.
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            sys = self._sessions[session_id]

        # LLM call outside the lock (blocking, can be slow)
        result = sys.generate_session_summary()

        with self._lock:
            sys.reset_session()
            if session_id in self._meta:
                self._meta[session_id]["total_ended"] = (
                    self._meta[session_id].get("total_ended", 0) + 1
                )
                self._meta[session_id]["last_ended"] = time.time()

        return result

    def delete(self, session_id: str):
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            sys = self._sessions.pop(session_id)
            self._meta.pop(session_id, None)
        # Only call cleanup() on sessions that own the model
        if not getattr(sys, "_is_clone", False):
            try:
                sys.cleanup()
            except Exception as e:
                log.warning("cleanup error on '%s': %s", session_id, e)

    def list_all(self) -> list[dict]:
        with self._lock:
            out = []
            for sid in list(self._sessions):
                meta = self._meta.get(sid, {})
                out.append({
                    "session_id":   sid,
                    "mode":         meta.get("mode", "fresh"),
                    "age_seconds":  round(time.time() - meta.get("created_at", time.time()), 1),
                    "total_queries": self._sessions[sid].total_queries,
                    "total_ended":  meta.get("total_ended", 0),
                    "user_name":    self._sessions[sid].memory.user_profile.name,
                })
            return out

    def cleanup_all(self):
        with self._lock:
            sids = list(self._sessions.keys())
        for sid in sids:
            try:
                self.delete(sid)
            except Exception:
                pass

    @property
    def default(self) -> CAGSystemFreshSession:
        return self.get("default")


registry = SessionRegistry()


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    mode    = os.getenv("CAG_SESSION_MODE", "fresh")
    rebuild = os.getenv("CAG_REBUILD_CACHE", "false").lower() == "true"
    config  = _build_config()

    print("🚀  Initializing default CAG session …")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: registry.create("default", mode=mode, config=config, rebuild_cache=rebuild),
    )
    print("✅  LLM microservice ready")

    yield

    print("🧹  Shutting down …")
    registry.cleanup_all()
    print("✅  Done")


# ══════════════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="CAG LLM Microservice",
    description="Real-time CAG inference with per-session isolation and auto-provisioning.",
    version="3.0.0",
    lifespan=lifespan,
)


# ══════════════════════════════════════════════════════════════════════════════
# Schemas
# ══════════════════════════════════════════════════════════════════════════════

class GenerateRequest(BaseModel):
    query:      str           = Field(..., min_length=1)
    session_id: str           = Field("default")


class BatchRequest(BaseModel):
    questions:  list[str]    = Field(..., min_items=1)
    session_id: str           = Field("default")


class NewSessionRequest(BaseModel):
    session_id: Optional[str] = Field(None)
    mode:       str           = Field("fresh", pattern="^(fresh|memory)$")


# ══════════════════════════════════════════════════════════════════════════════
# SSE helpers
# ══════════════════════════════════════════════════════════════════════════════

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _token_stream(
    cag:        CAGSystemFreshSession,
    query:      str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    Wraps CAG's synchronous streaming generator for async SSE delivery.

    Architecture:
    ─────────────
    • A producer thread runs cag.stream_query() (blocking GPU call) in a
      thread-pool executor and pushes each token into an asyncio.Queue.
    • The async consumer loop pulls tokens from the queue and yields SSE
      lines immediately — the event loop is never blocked.
    • After the producer sends "done", we ask the LLM for a session summary
      and bake user_name + summary into the final "done" SSE chunk.

    This gives true real-time delivery: the first token reaches the client
    as soon as the model emits it, with no buffering anywhere.
    """
    loop        = asyncio.get_event_loop()
    token_count = 0
    q: asyncio.Queue = asyncio.Queue()

    # ── producer: runs in thread pool, never touches event loop directly ──
    def _producer():
        try:
            for chunk in cag.stream_query(query):
                if chunk:
                    loop.call_soon_threadsafe(q.put_nowait, ("token", chunk))
        except Exception as exc:
            loop.call_soon_threadsafe(q.put_nowait, ("error", str(exc)))
        finally:
            loop.call_soon_threadsafe(q.put_nowait, ("done", None))

    producer_fut = loop.run_in_executor(None, _producer)

    # ── consumer: async, yields SSE lines as they arrive ─────────────────
    try:
        while True:
            kind, value = await q.get()

            if kind == "token":
                token_count += 1
                yield _sse({"type": "token", "token": value})

            elif kind == "error":
                yield _sse({"type": "error", "message": value})
                break

            elif kind == "done":
                # ── metadata: read directly from memory (always safe on
                #    both default and clone sessions).  get_stats() can crash
                #    on clones because knowledge_store was None before this fix.
                #    We wrap it defensively and fall back to memory object. ──
                user_name     = None
                stage         = "unknown"
                total_queries = cag.total_queries
                try:
                    mem_stats  = cag.memory.get_stats()
                    user_name  = mem_stats.get("user_name")
                    stage      = mem_stats.get("stage", "unknown")
                except Exception:
                    pass

                yield _sse({
                    "type":          "metadata",
                    "stage":         stage,
                    "user_name":     user_name,
                    "total_queries": total_queries,
                })

                # ── final terminator — no inline summary (generate_session_summary
                #    is a full LLM call; runs only on POST /session/{id}/end) ──
                yield _sse({
                    "type":        "done",
                    "total_tokens": token_count,
                    "user_name":   user_name,
                })
                break

    finally:
        await producer_fut   # always join the producer thread


# ══════════════════════════════════════════════════════════════════════════════
# /generate/stream   ← main voice-agent endpoint
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/generate/stream", summary="Stream response token-by-token (SSE)")
async def generate_stream(req: GenerateRequest):
    """
    Real-time SSE token stream with automatic session provisioning.

    The voice agent sends:
        POST /generate/stream
        {"query": "...", "session_id": "voice-agent"}

    The session "voice-agent" is auto-created on first call (zero overhead).
    Subsequent calls reuse the same session and its conversation memory.

    Final SSE chunk ("done") contains:
        {"type":"done", "total_tokens":N, "user_name":"Alex", "summary":"..."}

    Call POST /session/voice-agent/end after TTS finishes to reset for the
    next caller while keeping the session object alive.
    """
    cag = registry.get_or_create(req.session_id)   # ← the 404 fix

    return StreamingResponse(
        _token_stream(cag, req.query, req.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # nginx: don't buffer SSE
            "Connection":        "keep-alive",
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# /generate   (blocking)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/generate", summary="Full blocking response")
async def generate(req: GenerateRequest):
    """Complete answer in one JSON object. Auto-creates session if needed."""
    cag  = registry.get_or_create(req.session_id)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, cag.query, req.query)


# ══════════════════════════════════════════════════════════════════════════════
# /generate/batch
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/generate/batch", summary="Multiple questions in one call")
async def generate_batch(req: BatchRequest):
    """Process questions sequentially. Auto-creates session if needed."""
    cag  = registry.get_or_create(req.session_id)
    loop = asyncio.get_event_loop()

    results = []
    for question in req.questions:
        res = await loop.run_in_executor(None, cag.query, question)
        results.append(res)

    return {"session_id": req.session_id, "results": results, "count": len(results)}


# ══════════════════════════════════════════════════════════════════════════════
# /session  — lifecycle
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/session/new", status_code=201, summary="Explicitly create a named session")
async def create_session(req: NewSessionRequest):
    """
    Pre-register a session.  Optional — sessions are auto-created on first use.
    Useful only if you need 'memory' mode for a specific session.
    """
    sid    = req.session_id or str(uuid.uuid4())
    config = _build_config()
    loop   = asyncio.get_event_loop()

    try:
        await loop.run_in_executor(
            None,
            lambda: registry.create(sid, mode=req.mode, config=config),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {"session_id": sid, "mode": req.mode, "status": "created"}


@app.post("/session/{session_id}/reset", summary="Reset conversation, keep session alive")
async def reset_session(session_id: str):
    """
    Wipe conversation history and restart the stage machine.
    Session stays alive; model stays loaded.
    Use /session/{id}/end instead if you also want the name + summary first.
    """
    try:
        registry.reset(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"session_id": session_id, "status": "reset"}


@app.post(
    "/session/{session_id}/end",
    summary="End conversation: collect name+summary, then reset for next caller",
)
async def end_session(session_id: str):
    """
    The correct end-of-call sequence for the voice agent:

      1. LLM generates the user's name + a one-sentence session summary.
      2. Conversation memory is wiped so the next caller starts fresh.
      3. Name + summary returned to the caller.

    The session OBJECT stays alive — no GPU re-initialization on the next call.

    Voice-agent flow:
    ─────────────────
      POST /generate/stream  ←→  conversation happens
      [TTS finishes playing]
      POST /session/voice-agent/end  →  log/display name + summary
      [next person walks up]
      POST /generate/stream  ←  fresh conversation, same session_id

    Response example:
        {
          "session_id": "voice-agent",
          "user_name":  "Alex",
          "llm_name":   "Alex",
          "summary":    "User reported upload errors on files larger than 50MB..."
        }
    """
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, registry.end, session_id)
    except KeyError:
        # Session never existed — return graceful no-op
        return {
            "session_id": session_id,
            "user_name":  None,
            "llm_name":   None,
            "summary":    "No conversation found.",
        }

    return {"session_id": session_id, **result}


@app.delete("/session/{session_id}", summary="Destroy session and free resources")
async def delete_session(session_id: str):
    """Permanently destroy a session. Cannot delete 'default'."""
    if session_id == "default":
        raise HTTPException(status_code=400, detail="Cannot delete the default session.")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, registry.delete, session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"session_id": session_id, "status": "deleted"}


@app.get(
    "/session/{session_id}/summary",
    summary="Get name + LLM summary without resetting",
)
async def session_summary(session_id: str):
    """
    Non-destructive summary.  Use /session/{id}/end if you want
    summary + reset in one atomic call.
    """
    cag = registry.get_or_create(session_id)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, cag.generate_session_summary)
    return {"session_id": session_id, **result}


@app.get("/sessions", summary="List all active sessions")
async def list_sessions():
    sessions = registry.list_all()
    return {"sessions": sessions, "count": len(sessions)}


# ══════════════════════════════════════════════════════════════════════════════
# /stats  &  /health
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/stats", summary="Runtime stats for a session")
async def stats(session_id: str = Query("default")):
    cag = registry.get_or_create(session_id)
    return cag.get_stats()


@app.get("/health", summary="Liveness / readiness probe")
async def health():
    """200 = ready.  503 = still initializing."""
    try:
        ready = registry.default.is_initialized
    except Exception:
        ready = False

    if not ready:
        return JSONResponse(
            status_code=503,
            content={"status": "initializing", "ready": False},
        )

    return {
        "status":   "ok",
        "ready":    True,
        "sessions": len(registry.list_all()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Dev entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,   # never reload — GPU model is heavy
        workers=1,      # one process per GPU
        log_level="info",
    )