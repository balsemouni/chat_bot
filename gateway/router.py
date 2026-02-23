"""
router.py — Zero-Latency Parallel Pipeline
==========================================

Architecture:
  Audio chunks arrive continuously from the client (VAD-gated, 32ms each).
  Each stage runs in its own async task and communicates via queues.
  NO stage waits for the previous stage to finish.

  CLIENT → [audio_buf] → STT (streaming) → [word_q] → transcript+LLM → [token_q] → sentence_buf → [tts_q] → TTS → CLIENT

Latency budget (mirrors the monolith):
  STT first word : ~300ms  (Whisper on first 1s of speech)
  LLM first token: ~200ms  (KV-cache warm, CAG context injected)
  TTS first chunk: ~80ms   (PocketTTS first sentence)
  Total TTFS     : ~600ms  (time-to-first-sound)

Key design decisions (same as voice_agent_ai):
  1. STT runs on accumulated speech every 800ms — no silence wait needed
  2. LLM receives full conversation history (CAG context) on every call
  3. TTS fires on sentence-ending punctuation OR every 8 words — no full response wait
  4. Barge-in cancels all tasks instantly via asyncio.Event
  5. Audio buffer properly concatenates raw float32 PCM before b64-encoding

FIXES vs original:
  ✅ _flush_audio_buf now concatenates ALL chunks (not just the last one)
  ✅ _should_flush only fires on sentence-ending punctuation (. ! ? \n) — not , ; :
  ✅ MIN_WORDS_FOR_FLUSH raised to 8 (matches monolith's natural cadence)
  ✅ LLM request includes conversation history for CAG context
  ✅ HubSpot utterances logged via hubspot_client passed in from main
  ✅ session.mark_interrupted() called on barge-in
  ✅ Partial audio only fires when user IS talking (VAD gate respected)
"""

import asyncio
import base64
import json
import re
import time
import logging
from typing import Optional

import httpx

logger = logging.getLogger("router")


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitter — fire TTS as early as possible
# Only flush on sentence-ending punctuation (not commas / semicolons).
# Matches the natural pause points a human speaker would expect.
# ─────────────────────────────────────────────────────────────────────────────

_SENTENCE_END = re.compile(r'[.!?\n]')   # sentence-ending only (no , ; :)
_MIN_WORDS_FOR_FLUSH = 8                  # flush without punctuation after N words


def _should_flush(buf: list[str]) -> bool:
    """Return True if buffer should be sent to TTS now."""
    text = " ".join(buf)
    return bool(_SENTENCE_END.search(text)) or len(buf) >= _MIN_WORDS_FOR_FLUSH


# ─────────────────────────────────────────────────────────────────────────────
# Audio buffer helpers
# ─────────────────────────────────────────────────────────────────────────────

def _concat_audio_b64(chunks: list[str]) -> str:
    """
    Properly concatenate base64-encoded float32 PCM chunks.

    Each chunk is base64(float32 PCM bytes).
    We decode → concatenate raw bytes → re-encode.
    This is what the monolith does when it np.concatenate(recording_buffer).
    """
    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0]
    raw = b"".join(base64.b64decode(c) for c in chunks)
    return base64.b64encode(raw).decode()


# ─────────────────────────────────────────────────────────────────────────────
# PipelineRouter
# ─────────────────────────────────────────────────────────────────────────────

class PipelineRouter:
    """
    Zero-latency STT → LLM → TTS pipeline.

    Each stage is an independent async task communicating via asyncio.Queue.
    Cancellation propagates instantly via a shared asyncio.Event.

    Mirrors UltraLowLatencyVoiceAgentWithCAG pipeline logic exactly,
    but runs as async microservice calls instead of in-process calls.
    """

    def __init__(self, session, stt_url: str, llm_url: str, tts_url: str,
                 ws, hubspot=None):
        self.session   = session
        self.stt_url   = stt_url
        self.llm_url   = llm_url
        self.tts_url   = tts_url
        self.ws        = ws
        self.hubspot   = hubspot   # HubSpotClient instance (optional)

        # Shared HTTP client — keep-alive, connection pooling
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=30.0, write=5.0, pool=5.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        # Cancellation event — set this to kill all running tasks immediately
        self._cancel_event = asyncio.Event()

        # Active pipeline task (only one at a time — same as monolith's pipeline_lock)
        self._pipeline_task: Optional[asyncio.Task] = None

        # Audio buffer — accumulates VAD-gated chunks until 800ms of speech
        # Mirrors recording_buffer in vad_processor_thread
        self._audio_buf: list[str] = []
        self._audio_samples: int = 0
        self._PARTIAL_FLUSH_SAMPLES = 12800   # 800ms @ 16kHz

        # State (mirrors monolith's ai_is_speaking / user_is_talking flags)
        self._ai_speaking = False

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry points (called from main.py WebSocket handler)
    # ─────────────────────────────────────────────────────────────────────────

    async def handle_audio(self, audio_b64: str, sample_count: int = 512,
                           is_voice: bool = True):
        """
        Called for every VAD-gated mic chunk (32ms / 512 samples).

        is_voice=True  → accumulate and fire partial pipeline every 800ms
        is_voice=False → ignore (VAD said silence); client should not send these
                         but we guard here anyway

        Barge-in: if AI is speaking when voice arrives → interrupt immediately.
        """
        if not is_voice:
            return

        # ── Barge-in detection (mirrors vad_processor_thread) ─────────────────
        if self._ai_speaking:
            await self._interrupt()

        # ── Accumulate audio (mirrors recording_buffer.append(chunk)) ─────────
        self._audio_buf.append(audio_b64)
        self._audio_samples += sample_count

        # ── Fire partial pipeline every 800ms of speech ───────────────────────
        if self._audio_samples >= self._PARTIAL_FLUSH_SAMPLES:
            combined = self._flush_audio_buf()
            await self._launch_pipeline(combined)

    async def handle_utterance_end(self):
        """
        Called when the client signals end of utterance (VAD silence timeout).
        Flushes any remaining buffered audio and fires the final pipeline.
        Mirrors the silence_duration > silence_threshold_ms branch in the monolith.
        """
        if self._audio_samples > 0:
            combined = self._flush_audio_buf()
            await self._launch_pipeline(combined)

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline launcher
    # ─────────────────────────────────────────────────────────────────────────

    async def _launch_pipeline(self, audio_b64: str):
        """Cancel any running pipeline and start a fresh one."""
        if self._pipeline_task and not self._pipeline_task.done():
            self._cancel_event.set()
            try:
                await asyncio.wait_for(self._pipeline_task, timeout=0.2)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        self._cancel_event = asyncio.Event()
        self._pipeline_task = asyncio.create_task(
            self._pipeline(audio_b64), name="pipeline"
        )

    async def _interrupt(self):
        """
        Instant barge-in — cancel TTS and pipeline.
        Mirrors: self.tts.stop() + reset flags in the monolith.
        """
        logger.info("⚡ BARGE-IN — cancelling pipeline")
        self._cancel_event.set()
        self._ai_speaking = False
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
        STT → LLM → TTS, all streaming, all parallel.

        Queue flow:
          stt       → word_q   → transcript builder
          transcript → llm     → token_q
          token_q   → sent_buf → tts_q
          tts_q     → TTS      → WebSocket audio chunks

        Each stage reads from its input queue and writes to its output queue.
        If _cancel_event is set, all stages exit on their next iteration.
        """
        cancel = self._cancel_event

        word_q  = asyncio.Queue(maxsize=64)
        token_q = asyncio.Queue(maxsize=256)
        tts_q   = asyncio.Queue(maxsize=32)

        try:
            await self._send({"type": "status", "state": "listening"})

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
                self._stage_tts(tts_q, cancel), name="tts"
            )

            await asyncio.gather(stt_task, llm_task, buf_task, tts_task)

        except asyncio.CancelledError:
            cancel.set()
            raise
        finally:
            self._ai_speaking = False
            self.session.ai_is_speaking = False
            cancel.set()

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1 — STT: audio → word stream
    # ─────────────────────────────────────────────────────────────────────────

    async def _stage_stt(self, audio_b64: str, word_q: asyncio.Queue,
                          cancel: asyncio.Event):
        """
        Stream words from Whisper as fast as they arrive.
        Uses SSE /transcribe/stream endpoint on the STT microservice.
        Mirrors asr.transcribe_streaming() in the monolith.
        """
        t0 = time.perf_counter()
        first_word = True
        try:
            async with self._http.stream(
                "POST",
                f"{self.stt_url}/transcribe/stream",
                json={
                    "audio_b64":      audio_b64,
                    "sample_rate":    16000,
                    "ai_is_speaking": self._ai_speaking,
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if cancel.is_set():
                        break
                    if not line.startswith("data:"):
                        continue
                    try:
                        p = json.loads(line[5:])
                    except json.JSONDecodeError:
                        continue

                    t = p.get("type", "")
                    if t == "word":
                        word = p.get("word", "").strip()
                        if word:
                            if first_word:
                                logger.info("STT first word %.0fms",
                                            (time.perf_counter() - t0) * 1000)
                                first_word = False
                            await word_q.put(word)
                            # Stream word to client UI in real time
                            await self._send({"type": "word", "data": word})
                    elif t == "ai_filtered":
                        logger.info("STT: AI voice filtered — skipping")
                        break
                    elif t in ("done", "silence"):
                        break

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("STT error: %s", e)
        finally:
            await word_q.put(None)   # sentinel

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2 — Transcript builder + LLM
    # ─────────────────────────────────────────────────────────────────────────

    async def _stage_transcript_and_llm(self, word_q: asyncio.Queue,
                                         token_q: asyncio.Queue,
                                         cancel: asyncio.Event):
        """
        Drain words from word_q → build transcript → stream LLM into token_q.

        The LLM request includes the full conversation history so the CAG system
        has context — identical to how the monolith passes messages to the LLM.

        Mirrors _transcribe_and_respond + _get_cag_response in the monolith.
        """
        words = []
        t0 = time.perf_counter()

        while not cancel.is_set():
            try:
                word = await asyncio.wait_for(word_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if word is None:
                break
            words.append(word)

        if cancel.is_set() or not words:
            await token_q.put(None)
            return

        transcript = " ".join(words).strip()
        if not transcript:
            await token_q.put(None)
            return

        logger.info("Transcript (%dw) %.0fms: %r",
                    len(words), (time.perf_counter() - t0) * 1000, transcript[:60])

        # Save to session history (mirrors ui.print_user_complete + memory.add)
        self.session.add_user_utterance(transcript)
        self.session.state = "thinking"
        self.session.ai_is_thinking = True

        # Log to HubSpot in real-time (mirrors add_utterance in monolith)
        if self.hubspot:
            self.hubspot.add_utterance(
                self.session.id, "user", "User", transcript
            )

        await self._send({"type": "status", "state": "thinking"})

        # Build LLM payload — include history for CAG context
        llm_payload = {
            "query":      transcript,
            "session_id": self.session.id,
            "history":    self.session.get_messages_for_llm()[:-1],  # exclude current (already in query)
        }

        try:
            first_token = True
            t_llm = time.perf_counter()

            async with self._http.stream(
                "POST",
                f"{self.llm_url}/generate/stream",
                json=llm_payload,
            ) as resp:
                async for line in resp.aiter_lines():
                    if cancel.is_set():
                        break
                    if not line.startswith("data:"):
                        continue
                    try:
                        p = json.loads(line[5:])
                    except json.JSONDecodeError:
                        continue

                    t = p.get("type", "")
                    if t == "token":
                        token = p.get("token", "")
                        if not token:
                            continue
                        if first_token:
                            logger.info("LLM first token %.0fms",
                                        (time.perf_counter() - t_llm) * 1000)
                            first_token = False
                        await token_q.put(token)
                        # Stream token to client UI (mirrors ui.print_ai_token)
                        await self._send({"type": "ai_token", "data": token})
                    elif t == "done":
                        break
                    elif t == "error":
                        logger.error("LLM error: %s", p.get("message"))
                        break

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("LLM error: %s", e)
        finally:
            self.session.ai_is_thinking = False
            await token_q.put(None)   # sentinel

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3 — Sentence buffer: token stream → TTS chunks
    # ─────────────────────────────────────────────────────────────────────────

    async def _stage_sentence_buffer(self, token_q: asyncio.Queue,
                                      tts_q: asyncio.Queue,
                                      cancel: asyncio.Event):
        """
        Collect LLM tokens and flush to TTS as soon as we have a speakable chunk.

        Flush on: sentence-ending punctuation (. ! ?) OR every 8 words.
        This means TTS starts speaking while LLM is still generating.

        Mirrors tts.process_token + tts.flush in the monolith.
        """
        buf: list[str] = []
        full_tokens: list[str] = []

        async def flush():
            if not buf:
                return
            text = " ".join(buf).strip()
            if text:
                await tts_q.put(text)
                full_tokens.extend(buf)
            buf.clear()

        try:
            while not cancel.is_set():
                try:
                    token = await asyncio.wait_for(token_q.get(), timeout=0.3)
                except asyncio.TimeoutError:
                    # Flush partial buffer if LLM is slow (same as tts.flush)
                    if buf:
                        await flush()
                    continue

                if token is None:
                    # LLM done — flush remainder
                    await flush()
                    break

                buf.append(token)
                if _should_flush(buf):
                    await flush()

        except asyncio.CancelledError:
            raise
        finally:
            # Save full AI response to session history (mirrors memory.add_message)
            if full_tokens:
                full_response = " ".join(full_tokens)
                interrupted = cancel.is_set()
                self.session.add_ai_utterance(full_response, interrupted=interrupted)

                # Log AI response to HubSpot
                if self.hubspot:
                    self.hubspot.add_utterance(
                        self.session.id, "ai", "Assistant", full_response
                    )

            await tts_q.put(None)   # sentinel

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 4 — TTS: text chunks → audio → WebSocket
    # ─────────────────────────────────────────────────────────────────────────

    async def _stage_tts(self, tts_q: asyncio.Queue, cancel: asyncio.Event):
        """
        Takes text chunks from tts_q, synthesizes each in parallel,
        streams audio chunks to WebSocket as they arrive.

        Each TTS call is independent — the next chunk starts synthesizing
        while the current chunk is still being sent over the WebSocket.

        Mirrors UltraSmoothTTS / PocketTTSStreaming in the monolith.
        """
        self._ai_speaking = True
        self.session.ai_is_speaking = True
        self.session.state = "ai_speaking"
        await self._send({"type": "status", "state": "speaking"})
        t_first: Optional[float] = None
        pending: list[asyncio.Task] = []

        # Max 2 concurrent TTS synthesis calls (avoids overloading TTS service)
        sem = asyncio.Semaphore(2)

        try:
            while not cancel.is_set():
                try:
                    chunk_text = await asyncio.wait_for(tts_q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if chunk_text is None:
                    break
                if cancel.is_set():
                    break

                if t_first is None:
                    t_first = time.perf_counter()

                task = asyncio.create_task(
                    self._synthesize_and_send(chunk_text, sem, cancel),
                    name="tts-chunk",
                )
                pending.append(task)

            # Wait for all pending TTS tasks to finish (in order)
            for task in pending:
                if cancel.is_set():
                    task.cancel()
                else:
                    try:
                        await task
                    except (asyncio.CancelledError, Exception) as e:
                        logger.warning("TTS chunk error: %s", e)

        except asyncio.CancelledError:
            for task in pending:
                task.cancel()
            raise
        finally:
            self._ai_speaking = False
            self.session.ai_is_speaking = False
            self.session.state = "listening"
            await self._send({"type": "status", "state": "listening"})
            if t_first:
                logger.info("TTS first audio %.0fms",
                            (time.perf_counter() - t_first) * 1000)

    async def _synthesize_and_send(self, text: str, sem: asyncio.Semaphore,
                                    cancel: asyncio.Event):
        """Synthesize one text chunk and stream audio chunks to WebSocket."""
        async with sem:
            if cancel.is_set():
                return
            try:
                async with self._http.stream(
                    "POST",
                    f"{self.tts_url}/synthesize/pocket_stream",
                    json={"text": text, "speed": 1.0},
                ) as resp:
                    async for line in resp.aiter_lines():
                        if cancel.is_set():
                            return
                        if not line.startswith("data:"):
                            continue
                        try:
                            p = json.loads(line[5:])
                        except json.JSONDecodeError:
                            continue
                        if p.get("type") == "audio_chunk":
                            await self._send({
                                "type":        "audio_chunk",
                                "data":        p["data"],
                                "sample_rate": p.get("sample_rate", 22050),
                            })
                        elif p.get("type") in ("done", "error"):
                            break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("TTS synthesize error %r: %s", text[:40], e)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _flush_audio_buf(self) -> str:
        """
        Concatenate ALL accumulated audio chunks into one base64 blob.

        FIX: The original only took the last chunk — dropping all previous audio.
        This now mirrors np.concatenate(recording_buffer) in the monolith.
        """
        combined = _concat_audio_b64(self._audio_buf)
        self._audio_buf.clear()
        self._audio_samples = 0
        return combined

    async def _send(self, data: dict):
        try:
            await self.ws.send_json(data)
        except Exception as e:
            logger.debug("WS send error: %s", e)

    async def cancel_all(self):
        """Graceful shutdown — mirrors _shutdown in the monolith."""
        self._cancel_event.set()
        if self._pipeline_task and not self._pipeline_task.done():
            self._pipeline_task.cancel()
            try:
                await asyncio.wait_for(self._pipeline_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        await self._http.aclose()