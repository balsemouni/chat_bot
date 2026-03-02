"""
router.py — Zero-Latency Parallel Pipeline with CNN Barge-in (v3 — BUGS FIXED)
================================================================================

FIXES vs v2:
  ✓ Barge-in double-firing fixed — _barge_in_handled flag prevents CNN from
    firing after frontend already triggered barge-in (and vice versa)
  ✓ main.py barge-in now calls _interrupt_tts_only() NOT _interrupt()
    (full cancel was wrong — only TTS should stop on barge-in)
  ✓ CNN buffer now resets on BOTH human AND echo detection
    (was only resetting on human — echo frames were polluting the buffer)
  ✓ _stage_tts finally block deduplication preserved and working correctly

PIPELINE FLOW:
  Frontend streams audio chunks → handle_audio() accumulates them
  Frontend sends utterance_end  → _launch_pipeline() fires
  STT streams words  → sent to frontend live (type: "word")
  LLM streams tokens → sent to frontend live (type: "ai_token")
  TTS streams audio  → sent to frontend live (type: "audio_chunk") — zero latency

BARGE-IN FLOW (dual path, now deduped):
  Path 1 — Frontend fast path:
    Frontend VAD fires consecutive voice frames while AI speaks
    → sends { type: "barge_in" } WebSocket message
    → main.py calls router.handle_barge_in()   ← NEW: correct entry point
    → sets _barge_in_handled = True
    → calls _interrupt_tts_only()

  Path 2 — Backend CNN path:
    Audio chunks arrive while AI speaks
    CNN classifier: human speech or AI echo?
      Human detected → checks _barge_in_handled first
        → if already handled: skip (no double fire)
        → if not handled: sets flag, calls _interrupt_tts_only()
      AI echo → discard, RESET buffer   ← FIXED
      Uncertain → buffer more

WebSocket protocol (client → server):
  { "type": "audio_chunk",  "data": "<base64 float32 PCM>",
    "is_voice": true|false, "sample_count": 512 }
  { "type": "utterance_end" }
  { "type": "barge_in" }
  { "type": "end_session" }

WebSocket protocol (server → client):
  { "type": "word",        "data": "hello" }
  { "type": "ai_token",    "data": "Sure" }
  { "type": "audio_chunk", "data": "<base64>", "sample_rate": 22050 }
  { "type": "transcript",  "text": "full user transcript" }
  { "type": "ai_response", "text": "full AI response" }
  { "type": "status",      "state": "listening"|"thinking"|"speaking" }
  { "type": "interrupted" }
  { "type": "error",       "message": "..." }
"""

import asyncio
import base64
import json
import re
import time
import logging
import os
import sys
from typing import Optional, List

import httpx
import numpy as np

logger = logging.getLogger("router")

# ── Optional torch import — falls back gracefully if not installed ─────────────
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("⚠️  torch not installed — CNN barge-in disabled, using RMS fallback")


# ─────────────────────────────────────────────────────────────────────────────
# CNN Classifier for Smart Barge-in
# ─────────────────────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:
    class BargeInCNN(nn.Module):
        """
        CNN model architecture for audio classification.
        Must match the architecture used during training.
        """
        def __init__(self):
            super(BargeInCNN, self).__init__()
            self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
            self.pool  = nn.MaxPool1d(2)
            self.fc1   = nn.Linear(32 * 4000, 64)
            self.fc2   = nn.Linear(64, 2)  # 2 classes: AI_Echo=0, Human=1

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
else:
    class BargeInCNN:  # type: ignore
        pass


class BargeInClassifier:
    """
    CNN-based classifier to distinguish between human speech and AI echo.

    Falls back gracefully to RMS-gating when torch is unavailable.

    Returns:
      True  → Human speech detected → trigger barge-in
      False → AI echo / silence → ignore
      None  → Not enough audio yet (CNN path only)

    FIX v3: Buffer now resets on BOTH True AND False results.
    Previously only reset on True — echo frames were accumulating
    and polluting the next inference with stale data.
    """

    RMS_FALLBACK_THRESHOLD = 0.038

    def __init__(self, model_path="best_cnn_model.pth"):
        self.device = None
        self.model  = None
        self.min_samples_for_inference = 8000  # ~500ms at 16kHz
        self._inference_buffer: list  = []
        self._buffer_samples: int     = 0

        if not _TORCH_AVAILABLE:
            logger.warning("⚠️  torch unavailable — BargeInClassifier using RMS fallback")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            if os.path.exists(model_path):
                try:
                    self.model = torch.load(model_path, map_location=self.device)
                    logger.info(f"✅ Barge-in CNN loaded (full model) on {self.device}")
                except Exception:
                    self.model = BargeInCNN()
                    self.model.load_state_dict(
                        torch.load(model_path, map_location=self.device)
                    )
                    logger.info(f"✅ Barge-in CNN loaded (state dict) on {self.device}")

                self.model.eval()
                self.model.to(self.device)
            else:
                logger.warning(f"⚠️  CNN model not found at {model_path} — using RMS fallback")
                self.model = None

        except Exception as e:
            logger.error(f"❌ Failed to load CNN model: {e}")
            logger.warning("⚠️  Falling back to RMS-based barge-in")
            self.model = None

    def predict_is_human(self, audio_bytes: bytes, force_inference: bool = False) -> Optional[bool]:
        """
        Returns True if human speech, False if AI echo/silence, None if not enough audio yet.

        FIX: Buffer resets after EVERY completed inference (True or False),
        not only on True. Echo frames no longer accumulate between inferences.
        """
        # ── RMS fallback ──────────────────────────────────────────────────────
        if self.model is None:
            audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
            rms = float(np.sqrt(np.mean(audio_np ** 2) + 1e-10))
            logger.debug(f"[RMS barge-in] rms={rms:.4f} threshold={self.RMS_FALLBACK_THRESHOLD}")
            return rms >= self.RMS_FALLBACK_THRESHOLD

        # ── CNN inference ─────────────────────────────────────────────────────
        try:
            audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
            self._inference_buffer.append(audio_np)
            self._buffer_samples += len(audio_np)

            if self._buffer_samples < self.min_samples_for_inference and not force_inference:
                return None  # Not enough audio yet — keep buffering

            # Run inference
            full_audio = np.concatenate(self._inference_buffer)

            expected_samples = 16000
            if len(full_audio) > expected_samples:
                full_audio = full_audio[:expected_samples]
            elif len(full_audio) < expected_samples:
                padding = np.zeros(expected_samples - len(full_audio), dtype=np.float32)
                full_audio = np.concatenate([full_audio, padding])

            tensor = torch.from_numpy(full_audio).float().to(self.device)
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]

            with torch.no_grad():
                output = self.model(tensor)

            # FIX: Always reset buffer after inference — not just on human detection
            self.reset_buffer()

            if output.shape[1] == 2:
                probabilities = torch.softmax(output, dim=1)
                human_prob = probabilities[0][1].item()
                ai_prob    = probabilities[0][0].item()
                logger.debug(f"CNN Analysis — Human: {human_prob:.2%}, AI: {ai_prob:.2%}")
                return human_prob > 0.70

            elif output.shape[1] == 1:
                prob = torch.sigmoid(output).item()
                logger.debug(f"CNN Analysis — Human probability: {prob:.2%}")
                return prob > 0.70

            return None

        except Exception as e:
            logger.error(f"CNN inference error: {e}")
            self.reset_buffer()
            return True  # Safe fallback — assume human

    def reset_buffer(self):
        """Clear the inference buffer."""
        self._inference_buffer = []
        self._buffer_samples   = 0


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitter
# ─────────────────────────────────────────────────────────────────────────────

_SENTENCE_END        = re.compile(r'[.!?\n]')
_MIN_WORDS_FOR_FLUSH = 8


def _should_flush(buf: list[str]) -> bool:
    text = " ".join(buf)
    return bool(_SENTENCE_END.search(text)) or len(buf) >= _MIN_WORDS_FOR_FLUSH


# ─────────────────────────────────────────────────────────────────────────────
# Audio buffer helpers
# ─────────────────────────────────────────────────────────────────────────────

def _concat_audio_b64(chunks: list[str]) -> str:
    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0]
    raw = b"".join(base64.b64decode(c) for c in chunks)
    return base64.b64encode(raw).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Service URL resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_service_url(service_name: str, default_port: str) -> str:
    env_var = f"{service_name.upper()}_SERVICE_URL"
    url = os.getenv(env_var)

    if sys.platform == "win32":
        local_url = f"http://localhost:{default_port}"
        logger.info(f"🏠 Windows detected: Forcing {service_name} to {local_url}")
        return local_url

    if url:
        return url

    in_docker = os.getenv("RUNNING_IN_DOCKER", "").lower() == "true"
    if in_docker:
        return f"http://{service_name}:{default_port}"
    return f"http://localhost:{default_port}"


# ─────────────────────────────────────────────────────────────────────────────
# PipelineRouter
# ─────────────────────────────────────────────────────────────────────────────

class PipelineRouter:
    """
    Zero-latency STT → LLM → TTS pipeline with CNN-based smart barge-in.

    Barge-in design (v3 — deduped):
      - handle_barge_in(): called from main.py when frontend sends { type: "barge_in" }.
        Sets _barge_in_handled = True, then calls _interrupt_tts_only().
        FIX: Was calling _interrupt() (full cancel) — now correctly TTS-only.

      - _interrupt_tts_only(): stops TTS audio only. STT and LLM keep running.

      - _interrupt(): full pipeline cancel. ONLY used when launching a new pipeline
        over an existing one. NOT used for barge-in.

      - _barge_in_handled flag: prevents double-fire when both frontend fast path
        AND backend CNN detect the same barge-in event.
    """

    def __init__(self, session, stt_url: str, llm_url: str, tts_url: str,
                 ws, hubspot=None, cnn_model_path: str = "best_cnn_model.pth"):

        if sys.platform == "win32":
            stt_url = "http://localhost:8001"
            llm_url = "http://localhost:8002"
            tts_url = "http://localhost:8003"
            logger.info("🏠 Windows override: All services set to localhost")

        self.session  = session
        self.stt_url  = stt_url
        self.llm_url  = llm_url
        self.tts_url  = tts_url
        self.ws       = ws
        self.hubspot  = hubspot

        logger.info("=" * 60)
        logger.info("🔧 Router initialized with URLs:")
        logger.info(f"  🎤 STT: {stt_url}")
        logger.info(f"  🧠 LLM: {llm_url}")
        logger.info(f"  🔊 TTS: {tts_url}")
        logger.info(f"  🤖 CNN Model: {cnn_model_path}")

        self.classifier = BargeInClassifier(cnn_model_path)

        logger.info("=" * 60)

        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=5.0, pool=5.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        # Full pipeline cancel event
        self._cancel_event     = asyncio.Event()

        # TTS-only cancel event (barge-in)
        self._tts_cancel_event = asyncio.Event()

        self._pipeline_task: Optional[asyncio.Task] = None

        self._audio_buf: list[str] = []
        self._audio_samples: int   = 0

        self._ai_speaking = False

        # FIX: Dedup flag — prevents CNN + frontend both firing barge-in
        self._barge_in_handled = False

        self._MIN_AUDIO_SIZE_FOR_STT = 500

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry points
    # ─────────────────────────────────────────────────────────────────────────

    async def handle_audio(self, audio_b64: str, sample_count: int = 512,
                           is_voice: bool = True):
        """
        Called for every VAD-gated mic chunk (32ms / 512 samples).
        Audio is sent by frontend even while AI speaks — CNN uses it for barge-in.
        """
        if not is_voice:
            logger.debug("⚠️  Skipping non-voice audio chunk")
            return

        if len(self._audio_buf) == 0:
            await self._send({"type": "status", "state": "listening"})

        # ── CNN Barge-in detection (only while AI is speaking) ────────────────
        if self._ai_speaking:
            raw_audio = base64.b64decode(audio_b64)
            is_human  = self.classifier.predict_is_human(raw_audio)

            if is_human is True:
                # FIX: Check dedup flag before firing
                if not self._barge_in_handled:
                    logger.info("🔥 [CNN] Human detected during AI speech — BARGING IN")
                    self._barge_in_handled = True
                    await self._interrupt_tts_only()
                    self._audio_samples = 0
                    self._audio_buf     = []
                    self.classifier.reset_buffer()
                else:
                    logger.debug("🛡️  [CNN] Barge-in already handled by frontend — skipping")
                # Fall through — accumulate chunk as start of new utterance

            elif is_human is False:
                # FIX: Reset CNN buffer on echo detection too
                # Previously buffer kept growing with echo frames → bad next inference
                logger.debug("🛡️  [CNN] AI/Echo detected — Ignoring and resetting buffer")
                self.classifier.reset_buffer()
                return  # Don't add echo audio to STT buffer

            # is_human is None → not enough samples yet, keep buffering

        # ── Accumulate confirmed human audio ─────────────────────────────────
        self._audio_buf.append(audio_b64)
        self._audio_samples += sample_count
        logger.debug(f"🎤 Accumulated {self._audio_samples} samples ({len(self._audio_buf)} chunks)")

    async def handle_utterance_end(self):
        """
        Called when frontend signals end of utterance (VAD silence timeout).
        """
        if self._audio_samples > 0:
            logger.info(f"🎤 Utterance end — flushing {self._audio_samples} samples")
            combined = self._flush_audio_buf()

            if len(combined) > self._MIN_AUDIO_SIZE_FOR_STT:
                await self._launch_pipeline(combined)
            else:
                logger.warning(f"⚠️  Skipping STT — audio too small ({len(combined)} chars)")
                await self._send({"type": "status", "state": "listening"})
        else:
            logger.debug("Utterance end with no audio — staying listening")
            await self._send({"type": "status", "state": "listening"})

    async def handle_barge_in(self):
        """
        Called from main.py when frontend sends { type: "barge_in" }.

        FIX v3:
          - Was calling _interrupt() (full pipeline cancel) — WRONG.
            Full cancel kills STT + LLM + TTS, losing conversation context.
          - Now correctly calls _interrupt_tts_only() — stops TTS only.
          - Sets _barge_in_handled so CNN path doesn't double-fire.
        """
        if self._barge_in_handled:
            logger.debug("🛡️  Frontend barge-in received but already handled by CNN — skipping")
            return

        logger.info("🔥 [Frontend] Barge-in received — stopping TTS only")
        self._barge_in_handled = True
        await self._interrupt_tts_only()

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline launcher
    # ─────────────────────────────────────────────────────────────────────────

    async def _launch_pipeline(self, audio_b64: str):
        """Cancel any running pipeline and start a fresh one."""
        if self._pipeline_task and not self._pipeline_task.done():
            logger.info("🔄 Cancelling existing pipeline for new utterance")
            self._cancel_event.set()
            try:
                await asyncio.wait_for(self._pipeline_task, timeout=0.2)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Fresh events for the new pipeline
        self._cancel_event     = asyncio.Event()
        self._tts_cancel_event = asyncio.Event()

        # Reset barge-in dedup flag for new pipeline
        self._barge_in_handled = False

        # Reset CNN buffer so old audio doesn't bleed in
        self.classifier.reset_buffer()

        self._pipeline_task = asyncio.create_task(
            self._pipeline(audio_b64), name="pipeline"
        )

    async def _interrupt_tts_only(self):
        """
        Barge-in: stop TTS audio ONLY.
        STT and LLM stages keep running — conversation context is preserved.
        """
        logger.info("🛑 TTS-only interrupt — STT/LLM unaffected")
        self._tts_cancel_event.set()
        self._ai_speaking           = False
        self.session.ai_is_speaking = False
        self.session.mark_interrupted()
        await self._send({"type": "interrupted"})
        await self._send({"type": "status", "state": "listening"})

    async def _interrupt(self):
        """
        Full pipeline cancel.
        ONLY used when a new pipeline launches over an existing one.
        NOT called for barge-in — use handle_barge_in() for that.
        """
        logger.info("⚡ FULL CANCEL — cancelling entire pipeline")
        self._cancel_event.set()
        self._tts_cancel_event.set()
        self._ai_speaking           = False
        self.session.ai_is_speaking = False
        self.session.mark_interrupted()

        if self._pipeline_task and not self._pipeline_task.done():
            self._pipeline_task.cancel()
            try:
                await asyncio.wait_for(self._pipeline_task, timeout=0.15)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        await self._send({"type": "interrupted"})
        await self._send({"type": "status", "state": "listening"})

    # ─────────────────────────────────────────────────────────────────────────
    # Full parallel pipeline — STT → LLM → TTS
    # ─────────────────────────────────────────────────────────────────────────

    async def _pipeline(self, audio_b64: str):
        """
        STT → LLM → TTS, all streaming, all parallel via asyncio queues.

        Queue topology:
          audio_b64
            → [_stage_stt]                → word_q
            → [_stage_transcript_and_llm] → token_q
            → [_stage_sentence_buffer]    → tts_q
            → [_stage_tts]               → WebSocket
        """
        cancel     = self._cancel_event
        tts_cancel = self._tts_cancel_event

        word_q  = asyncio.Queue(maxsize=64)
        token_q = asyncio.Queue(maxsize=256)
        tts_q   = asyncio.Queue(maxsize=32)

        try:
            await self._send({"type": "status", "state": "thinking"})

            stt_task = asyncio.create_task(
                self._stage_stt(audio_b64, word_q, cancel), name="stt"
            )
            llm_task = asyncio.create_task(
                self._stage_transcript_and_llm(word_q, token_q, cancel), name="llm"
            )
            buf_task = asyncio.create_task(
                self._stage_sentence_buffer(token_q, tts_q, cancel), name="buf"
            )
            tts_task = asyncio.create_task(
                self._stage_tts(tts_q, cancel, tts_cancel), name="tts"
            )

            await asyncio.gather(stt_task, llm_task, buf_task, tts_task)

        except asyncio.CancelledError:
            cancel.set()
            raise
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            await self._send({"type": "error", "message": str(e)})
        finally:
            self._ai_speaking           = False
            self.session.ai_is_speaking = False
            cancel.set()
            self.classifier.reset_buffer()

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1 — STT
    # ─────────────────────────────────────────────────────────────────────────

    async def _stage_stt(self, audio_b64: str, word_q: asyncio.Queue,
                         cancel: asyncio.Event):
        t0 = time.perf_counter()
        first_word = True

        if not audio_b64:
            logger.warning("Empty audio buffer for STT")
            await word_q.put(None)
            return

        try:
            try:
                health = await self._http.get(f"{self.stt_url}/health", timeout=5.0)
                if health.status_code != 200:
                    logger.warning(f"STT health check returned {health.status_code}")
            except Exception as e:
                logger.error(f"❌ Cannot connect to STT at {self.stt_url}: {e}")
                await self._send({"type": "error", "message": "STT service unavailable"})
                await word_q.put(None)
                return

            async with self._http.stream(
                "POST",
                f"{self.stt_url}/transcribe/stream",
                json={
                    "audio_b64":      audio_b64,
                    "sample_rate":    16000,
                    "ai_is_speaking": False,
                },
                timeout=30.0,
            ) as resp:
                if resp.status_code != 200:
                    logger.error(f"STT returned {resp.status_code}")
                    await word_q.put(None)
                    return

                word_count = 0
                async for line in resp.aiter_lines():
                    if cancel.is_set():
                        break
                    if not line or not line.startswith("data:"):
                        continue
                    try:
                        p = json.loads(line[5:])
                        t = p.get("type", "")

                        if t == "word":
                            word = p.get("word", "").strip()
                            if word:
                                word_count += 1
                                if first_word:
                                    elapsed = (time.perf_counter() - t0) * 1000
                                    logger.info(f"🎤 STT first word {elapsed:.0f}ms: '{word}'")
                                    first_word = False
                                await word_q.put(word)
                                await self._send({"type": "word", "data": word})

                        elif t in ("done", "silence"):
                            logger.info(f"STT complete — {word_count} words")
                            break
                        elif t == "error":
                            logger.error(f"STT error: {p.get('message')}")
                            break

                    except Exception as e:
                        logger.error(f"Error parsing STT response: {e}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"❌ STT stage error: {e}")
            await self._send({"type": "error", "message": f"STT error: {str(e)}"})
        finally:
            await word_q.put(None)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2 — Transcript + LLM
    # ─────────────────────────────────────────────────────────────────────────

    async def _stage_transcript_and_llm(self, word_q: asyncio.Queue,
                                         token_q: asyncio.Queue,
                                         cancel: asyncio.Event):
        words = []

        while not cancel.is_set():
            try:
                word = await asyncio.wait_for(word_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if word is None:
                break
            words.append(word)

        if cancel.is_set() or not words:
            logger.warning("No words received from STT")
            await token_q.put(None)
            return

        transcript = " ".join(words).strip()
        if not transcript:
            logger.warning("Empty transcript from STT")
            await token_q.put(None)
            return

        logger.info(f"📝 Transcript ({len(words)} words): {transcript[:100]}")
        await self._send({"type": "transcript", "text": transcript})

        self.session.add_user_utterance(transcript)
        self.session.state          = "thinking"
        self.session.ai_is_thinking = True

        if self.hubspot:
            self.hubspot.add_utterance(self.session.id, "user", "User", transcript)

        await self._send({"type": "status", "state": "thinking"})

        llm_payload = {
            "query":      transcript,
            "session_id": self.session.id,
            "history":    self.session.get_messages_for_llm()[:-1],
        }

        try:
            try:
                health = await self._http.get(f"{self.llm_url}/health", timeout=5.0)
                if health.status_code != 200:
                    logger.warning(f"LLM health check returned {health.status_code}")
            except Exception as e:
                logger.error(f"❌ Cannot connect to LLM at {self.llm_url}: {e}")
                await self._send({"type": "error", "message": "LLM service unreachable"})
                await token_q.put(None)
                return

            first_token = True
            t_llm       = time.perf_counter()
            token_count = 0

            async with self._http.stream(
                "POST",
                f"{self.llm_url}/generate/stream",
                json=llm_payload,
                timeout=60.0,
            ) as resp:
                if resp.status_code != 200:
                    logger.error(f"LLM returned {resp.status_code}")
                    await token_q.put(None)
                    return

                async for line in resp.aiter_lines():
                    if cancel.is_set():
                        break
                    if not line or not line.startswith("data:"):
                        continue
                    try:
                        p = json.loads(line[5:])
                        t = p.get("type", "")

                        if t == "token":
                            token = p.get("token", "")
                            if token:
                                token_count += 1
                                if first_token:
                                    elapsed = (time.perf_counter() - t_llm) * 1000
                                    logger.info(f"🤖 LLM first token {elapsed:.0f}ms")
                                    first_token = False
                                await token_q.put(token)
                                await self._send({"type": "ai_token", "data": token})

                        elif t == "done":
                            logger.info(f"LLM complete — {token_count} tokens")
                            break
                        elif t == "error":
                            logger.error(f"LLM error: {p.get('message')}")
                            break

                    except Exception as e:
                        logger.error(f"Error parsing LLM response: {e}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"❌ LLM stage error: {e}")
            await self._send({"type": "error", "message": f"LLM error: {str(e)}"})
        finally:
            self.session.ai_is_thinking = False
            await token_q.put(None)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3 — Sentence buffer
    # ─────────────────────────────────────────────────────────────────────────

    async def _stage_sentence_buffer(self, token_q: asyncio.Queue,
                                      tts_q: asyncio.Queue,
                                      cancel: asyncio.Event):
        buf: list[str]         = []
        full_tokens: list[str] = []
        flush_count = 0

        async def flush():
            nonlocal flush_count
            if not buf:
                return
            text = " ".join(buf).strip()
            if text:
                flush_count += 1
                logger.debug(f"Flushing sentence {flush_count}: {text[:50]}...")
                await tts_q.put(text)
                full_tokens.extend(buf)
            buf.clear()

        try:
            while not cancel.is_set():
                try:
                    token = await asyncio.wait_for(token_q.get(), timeout=0.3)
                except asyncio.TimeoutError:
                    if buf:
                        await flush()
                    continue

                if token is None:
                    await flush()
                    break

                buf.append(token)
                if _should_flush(buf):
                    await flush()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"❌ Sentence buffer error: {e}")
        finally:
            if full_tokens:
                full_response = " ".join(full_tokens)
                interrupted   = cancel.is_set()
                self.session.add_ai_utterance(full_response, interrupted=interrupted)
                logger.info(f"✅ AI response ({len(full_tokens)} tokens, interrupted={interrupted})")
                await self._send({"type": "ai_response", "text": full_response})

                if self.hubspot:
                    self.hubspot.add_utterance(
                        self.session.id, "ai", "Assistant", full_response
                    )

            await tts_q.put(None)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 4 — TTS
    # ─────────────────────────────────────────────────────────────────────────

    async def _stage_tts(self, tts_q: asyncio.Queue, cancel: asyncio.Event,
                         tts_cancel: asyncio.Event):
        self._ai_speaking           = True
        self.session.ai_is_speaking = True
        self.session.state          = "ai_speaking"
        await self._send({"type": "status", "state": "speaking"})

        t_first: Optional[float]    = None
        pending: list[asyncio.Task] = []
        chunk_count = 0
        sem = asyncio.Semaphore(2)

        def _should_stop() -> bool:
            return cancel.is_set() or tts_cancel.is_set()

        try:
            try:
                health = await self._http.get(f"{self.tts_url}/health", timeout=5.0)
                if health.status_code != 200:
                    logger.warning(f"TTS health check returned {health.status_code}")
            except Exception as e:
                logger.error(f"❌ Cannot connect to TTS at {self.tts_url}: {e}")
                await self._send({"type": "error", "message": "TTS service unreachable"})
                return

            while not _should_stop():
                try:
                    chunk_text = await asyncio.wait_for(tts_q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if chunk_text is None or _should_stop():
                    break

                chunk_count += 1
                if t_first is None:
                    t_first = time.perf_counter()

                task = asyncio.create_task(
                    self._synthesize_and_send(chunk_text, sem, cancel, tts_cancel),
                    name=f"tts-chunk-{chunk_count}",
                )
                pending.append(task)

            for task in pending:
                if _should_stop():
                    task.cancel()
                else:
                    try:
                        await task
                    except (asyncio.CancelledError, Exception) as e:
                        logger.warning(f"TTS chunk error: {e}")

        except asyncio.CancelledError:
            for task in pending:
                task.cancel()
            raise
        except Exception as e:
            logger.error(f"❌ TTS stage error: {e}")
        finally:
            self._ai_speaking           = False
            self.session.ai_is_speaking = False
            self.session.state          = "listening"

            # Only send listening status if barge-in did NOT occur.
            # _interrupt_tts_only() already sent "interrupted" + "status:listening"
            # Sending again would cause duplicate state update on frontend.
            if not tts_cancel.is_set():
                await self._send({"type": "status", "state": "listening"})

            if t_first:
                elapsed = (time.perf_counter() - t_first) * 1000
                logger.info(f"🔊 TTS first audio {elapsed:.0f}ms, {chunk_count} chunks total")

    async def _synthesize_and_send(self, text: str, sem: asyncio.Semaphore,
                                    cancel: asyncio.Event, tts_cancel: asyncio.Event):
        async with sem:
            if cancel.is_set() or tts_cancel.is_set():
                return
            try:
                async with self._http.stream(
                    "POST",
                    f"{self.tts_url}/synthesize/pocket_stream",
                    json={"text": text, "speed": 1.0},
                    timeout=30.0,
                ) as resp:
                    if resp.status_code != 200:
                        logger.error(f"TTS returned {resp.status_code}")
                        return

                    chunk_received = False
                    async for line in resp.aiter_lines():
                        if cancel.is_set() or tts_cancel.is_set():
                            return
                        if not line or not line.startswith("data:"):
                            continue
                        try:
                            p = json.loads(line[5:])
                            if p.get("type") == "audio_chunk":
                                chunk_received = True
                                await self._send({
                                    "type":        "audio_chunk",
                                    "data":        p["data"],
                                    "sample_rate": p.get("sample_rate", 22050),
                                })
                            elif p.get("type") in ("done", "error"):
                                if not chunk_received:
                                    logger.warning(f"TTS no audio for: '{text[:30]}'")
                                break
                        except Exception as e:
                            logger.error(f"Error parsing TTS response: {e}")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"❌ TTS synthesize error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _flush_audio_buf(self) -> str:
        if not self._audio_buf:
            return ""
        combined = _concat_audio_b64(self._audio_buf)
        self._audio_buf.clear()
        self._audio_samples = 0
        return combined

    async def _send(self, data: dict):
        try:
            await self.ws.send_json(data)
        except Exception as e:
            logger.debug(f"WS send error: {e}")

    async def cancel_all(self):
        """Graceful shutdown."""
        self._cancel_event.set()
        self._tts_cancel_event.set()
        if self._pipeline_task and not self._pipeline_task.done():
            self._pipeline_task.cancel()
            try:
                await asyncio.wait_for(self._pipeline_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        await self._http.aclose()
        self.classifier.reset_buffer()