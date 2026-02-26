""" test_client.py — Zero-Latency Voice Agent
==========================================

PIPELINE:
  MIC → DeepFilter → VAD → STT → LLM → TTS
  Every stage streams. Nothing waits for the previous to complete.

TTS LATENCY STRATEGY — IMMEDIATE PER-TOKEN FLUSH:
  Every LLM token is sent to TTS IMMEDIATELY as it arrives.
  No buffering. No word-boundary waiting. No delay.

  Each token fires a TTS HTTP request the instant it's received.
  Synthesis tasks run in parallel, playback happens IN ORDER via async queue.

  LATENCY BREAKDOWN (local network):
    VAD end-of-speech  :  60–200ms  (adaptive energy drop)
    STT finalize       :  80–200ms  (audio pre-buffered, just HTTP RTT)
    LLM first token    :  50–300ms  (model dependent)
    TTS first audio    :  80–150ms  (fires on FIRST token — instant)
    ─────────────────────────────────
    Total best case    : ~270ms     (perceived as instant)

KEY CHANGES vs PREVIOUS VERSION:
  1. TTSClient flushes on EVERY token — zero buffering delay
  2. Each token fires TTS request immediately as it arrives from LLM
  3. All other optimizations preserved (parallel TTS, ordered playback, etc.)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import queue
import signal
import sys
import threading
import time
import wave

import httpx
import numpy as np
import sounddevice as sd

sys.path.insert(0, "stt-service")

# ── silence noisy libraries ───────────────────────────────────────────────────
logging.disable(logging.CRITICAL)
for _n in ("httpx", "httpcore", "asyncio", "urllib3", "anyio", "sounddevice"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

# ── ANSI colours ──────────────────────────────────────────────────────────────
G, R, Y, B, M, C = "\033[92m", "\033[91m", "\033[93m", "\033[94m", "\033[95m", "\033[96m"
DIM, BLD, RST    = "\033[2m",  "\033[1m",  "\033[0m"

# ── gate thresholds (overridden by CLI) ───────────────────────────────────────
BARGE_MIN_MS     = 200
BARGE_RMS_MIN    = 0.015
BARGE_RMS_STRICT = 0.035
DEBUG            = False


# =============================================================================
# DeepFilter Noise Suppressor
# =============================================================================
class DeepFilterSuppressor:
    """Removes background noise, keeps voice. Graceful fallback if not installed."""

    def __init__(self, sr: int = 16000):
        self._sr     = sr
        self._model  = None
        self._state  = None
        self.enabled = False
        self._lock   = threading.Lock()
        try:
            from df import init_df, enhance
            self._enhance             = enhance
            self._model, self._state, _ = init_df()
            self.enabled              = True
        except Exception:
            pass

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return audio
        with self._lock:
            try:
                import torch
                t        = torch.from_numpy(audio).unsqueeze(0)
                enhanced = self._enhance(self._model, self._state, t)
                return enhanced.squeeze(0).numpy()
            except Exception:
                return audio


# =============================================================================
# WebRTC VAD  (with adaptive end-of-speech)
# =============================================================================
class WebRTCVAD:
    """Google WebRTC VAD + adaptive EOS detection. Falls back gracefully."""

    def __init__(self, sr: int = 16000, aggressiveness: int = 2,
                 silence_ms: int = 200, min_ms: int = 150):
        self._sr           = sr
        self._silence_ms   = silence_ms
        self._min_ms       = min_ms
        self._frame_ms     = 30
        self._frame_sz     = int(sr * self._frame_ms / 1000)
        self.enabled       = False
        self._speaking     = False
        self._last_voice   = 0.0
        self._buf: list[np.ndarray] = []
        self._leftover     = np.array([], dtype=np.float32)
        # adaptive EOS
        self._speak_rms    = 0.0
        self._quiet_frames = 0
        try:
            import webrtcvad
            self._vad    = webrtcvad.Vad(aggressiveness)
            self.enabled = True
        except Exception:
            pass

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def process(self, chunk: np.ndarray) -> np.ndarray | None:
        if not self.enabled:
            return None
        combined = np.concatenate([self._leftover, chunk])
        offset   = 0
        result   = None
        while offset + self._frame_sz <= len(combined):
            frame  = combined[offset:offset + self._frame_sz]
            offset += self._frame_sz
            pcm16  = (frame * 32767).astype(np.int16).tobytes()
            try:
                is_voice = self._vad.is_speech(pcm16, self._sr)
            except Exception:
                is_voice = False
            now       = time.monotonic()
            frame_rms = float(np.sqrt(np.mean(frame ** 2) + 1e-10))
            if is_voice:
                self._last_voice   = now
                self._quiet_frames = 0
                if not self._speaking:
                    self._speaking  = True
                    self._buf       = []
                    self._speak_rms = frame_rms
                else:
                    self._speak_rms = 0.9 * self._speak_rms + 0.1 * frame_rms
                self._buf.append(frame)
            elif self._speaking:
                self._buf.append(frame)
                self._quiet_frames += 1
                elapsed = (now - self._last_voice) * 1000
                # adaptive: energy dropped to <30% of speaking level + 60ms quiet
                adaptive = (
                    self._quiet_frames >= 2
                    and frame_rms < self._speak_rms * 0.3
                    and elapsed >= 60
                )
                if elapsed >= self._silence_ms or adaptive:
                    audio          = np.concatenate(self._buf) if self._buf else np.array([], np.float32)
                    self._speaking = False
                    self._buf      = []
                    self._quiet_frames = 0
                    self._speak_rms    = 0.0
                    min_samples = int(self._min_ms / 1000 * self._sr)
                    if len(audio) >= min_samples:
                        result = audio.astype(np.float32)
        self._leftover = combined[offset:]
        return result


# =============================================================================
# Energy VAD  (fallback)
# =============================================================================
class EnergyVAD:
    def __init__(self, sr=16000, threshold=0.008, silence_ms=200,
                 min_ms=150, max_s=20.0):
        self.sr          = sr
        self.threshold   = threshold
        self.silence_ms  = silence_ms
        self.min_samples = int(min_ms / 1000 * sr)
        self.max_samples = int(max_s * sr)
        self._buf: list[np.ndarray] = []
        self._speaking   = False
        self._last_voice = 0.0
        self._speak_rms  = 0.0

    @staticmethod
    def rms(chunk: np.ndarray) -> float:
        return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2) + 1e-10))

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def process(self, chunk: np.ndarray) -> np.ndarray | None:
        r   = self.rms(chunk)
        now = time.monotonic()
        if r >= self.threshold:
            self._last_voice = now
            if not self._speaking:
                self._speaking  = True
                self._buf       = []
                self._speak_rms = r
            else:
                self._speak_rms = 0.9 * self._speak_rms + 0.1 * r
            self._buf.append(chunk)
            if sum(len(c) for c in self._buf) >= self.max_samples:
                return self._emit()
        elif self._speaking:
            self._buf.append(chunk)
            elapsed  = (now - self._last_voice) * 1000
            adaptive = (r < self._speak_rms * 0.3 and elapsed >= 60)
            if elapsed >= self.silence_ms or adaptive:
                return self._emit()
        return None

    def _emit(self) -> np.ndarray | None:
        audio          = np.concatenate(self._buf) if self._buf else np.array([], np.float32)
        self._speaking = False
        self._buf      = []
        self._speak_rms = 0.0
        return audio.astype(np.float32) if len(audio) >= self.min_samples else None


# =============================================================================
# Audio helpers
# =============================================================================
def _decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    buf = io.BytesIO(data)
    with wave.open(buf, "rb") as wf:
        sr  = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, sr


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return audio
    idx = np.linspace(0, len(audio) - 1, int(len(audio) * dst / src))
    return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)


# =============================================================================
# Player  — dedicated thread, fed from async event loop
# =============================================================================
class Player:
    """
    Plays WAV bytes in a background thread.
    feed() is thread-safe and non-blocking.
    Notifies an asyncio.Event when playback finishes.
    """

    def __init__(self, sr: int = 22050):
        self._sr         = sr
        self._gen        = 0
        self._lock       = threading.Lock()
        self._q: queue.Queue[bytes | None] | None = None
        self._alive      = False
        self._done_event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._alive

    def setup(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def feed(self, wav: bytes):
        with self._lock:
            if not self._alive:
                self._start()
            if self._q:
                try:
                    self._q.put_nowait(wav)
                except queue.Full:
                    # drop oldest, keep newest
                    try:
                        self._q.get_nowait()
                        self._q.put_nowait(wav)
                    except Exception:
                        pass

    def stop(self):
        with self._lock:
            self._gen  += 1
            self._alive = False
            q, self._q  = self._q, None
        if q:
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break
            q.put_nowait(None)  # wake the thread
        self._notify_done()

    def new_done_event(self) -> asyncio.Event:
        """Call from async context before starting a new TTS response."""
        evt = asyncio.Event()
        self._done_event = evt
        return evt

    def _notify_done(self):
        if self._loop and not self._loop.is_closed() and self._done_event:
            try:
                self._loop.call_soon_threadsafe(self._done_event.set)
            except RuntimeError:
                pass

    def _start(self):
        gen       = self._gen
        q: queue.Queue[bytes | None] = queue.Queue(maxsize=512)
        self._q   = q
        self._alive = True
        threading.Thread(target=self._run, args=(gen, q),
                         daemon=True, name=f"player-{gen}").start()

    def _run(self, gen: int, q: queue.Queue):
        stream = None
        try:
            stream = sd.OutputStream(
                samplerate=self._sr, channels=1, dtype="float32",
                blocksize=256  # 256 vs 512 = half output buffer latency (~5ms)
            )
            stream.start()
            while True:
                with self._lock:
                    if self._gen != gen:
                        break
                try:
                    wav = q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if wav is None:
                    break
                try:
                    audio, sr = _decode_wav(wav)
                    if sr != self._sr:
                        audio = _resample(audio, sr, self._sr)
                    blk = 256
                    for i in range(0, len(audio), blk):
                        with self._lock:
                            if self._gen != gen:
                                break
                        if stream.active:
                            stream.write(audio[i:i + blk])
                except Exception:
                    pass
                finally:
                    try: q.task_done()
                    except Exception: pass
        except Exception:
            pass
        finally:
            with self._lock:
                if self._gen == gen:
                    self._alive = False
            try:
                if stream:
                    stream.stop()
                    stream.close()
            except Exception:
                pass
            self._notify_done()


# =============================================================================
# CNN Detector
# =============================================================================
class CNNDetector:
    def __init__(self, model_path: str, device: str, threshold: float):
        self._det       = None
        self._lock      = threading.Lock()
        self.threshold  = threshold
        self.enabled    = False
        if not model_path:
            return
        try:
            from ai_voice_detector import AIVoiceDetector
            self._det    = AIVoiceDetector(model_path, device, threshold)
            self.enabled = True
        except Exception as e:
            _p(f"{Y}⚠ CNN load failed: {e} → RMS-only mode{RST}")

    def classify_sync(self, audio: np.ndarray, sr: int = 16000) -> tuple[str, float]:
        if not self.enabled or self._det is None:
            return "human", 1.0
        if len(audio) < int(sr * 0.25):
            return "uncertain", 0.0
        with self._lock:
            try:
                is_ai, conf, _ = self._det.is_ai_voice(audio, sr)
                return ("ai" if is_ai else "human"), float(conf)
            except Exception:
                return "uncertain", 0.0


# =============================================================================
# TTSClient — fully async, IMMEDIATE per-token flush, parallel overlap
# =============================================================================
class TTSClient:
    """
    Receives LLM tokens one-by-one via push_token() / end_tokens().

    ZERO-LATENCY MODE: Every token is flushed to TTS IMMEDIATELY.
    No buffering. No word-boundary detection. No waiting.

    Each token fires its own TTS HTTP request the instant it arrives.
    Multiple TTS HTTP requests run in parallel but play in ORDER
    via a sequenced async queue — so audio is always coherent.

    Why immediate per-token:
      - First audio chunk fires on the very FIRST LLM token (~50ms after LLM starts)
      - Zero accumulation delay — each token goes to TTS the moment it exists
      - TTS requests pipeline in parallel — synthesis of token N+1 starts
        while token N audio is still being played
      - Ordered playback queue ensures correct audio sequence always
    """

    def __init__(self, url: str, speed: float = 1.0, sr: int = 22050):
        self._url    = url
        self._speed  = speed
        self._player = Player(sr)
        self._http: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Per-response state (reset each call to start_response)
        self._token_buf: list[str]               = []
        self._play_q:    asyncio.Queue[asyncio.Task | None] | None = None
        self._play_task: asyncio.Task | None     = None
        self._tts_task:  asyncio.Task | None     = None
        self._done_event: asyncio.Event          = asyncio.Event()
        self._done_event.set()

    @property
    def is_speaking(self) -> bool:
        tts_busy = self._tts_task is not None and not self._tts_task.done()
        return self._player.is_playing or tts_busy

    def setup(self, loop: asyncio.AbstractEventLoop, http: httpx.AsyncClient):
        self._loop = loop
        self._http = http
        self._player.setup(loop)

    # ── called from _llm_and_speak ────────────────────────────────────────────

    def start_response(self) -> asyncio.Event:
        """
        Call ONCE at the start of each LLM response.
        Returns a done-event you can await when TTS finishes.
        """
        # cancel previous if still running
        for t in (self._tts_task, self._play_task):
            if t and not t.done():
                t.cancel()
        self._player.stop()

        self._token_buf  = []
        self._done_event = asyncio.Event()
        # play_q serialises synthesis tasks so audio always plays in order
        self._play_q     = asyncio.Queue()

        self._play_task = self._loop.create_task(self._ordered_player())
        self._tts_task  = None   # set per flush

        return self._done_event

    def push_token(self, token: str):
        """
        ZERO-LATENCY: called for every LLM token.
        Flushes IMMEDIATELY on every token — no buffering, no waiting.
        Each token fires its own TTS synthesis request right away.
        """
        if not token:
            return
        self._token_buf.append(token)
        self._flush()  # flush on EVERY token immediately

    def end_tokens(self):
        """Call when LLM stream ends. Flushes any remaining buffer (should be empty)."""
        self._flush()
        # sentinel None → ordered_player knows to set done_event
        if self._play_q:
            self._loop.call_soon_threadsafe(
                self._play_q.put_nowait, None
            )

    def interrupt(self):
        """Barge-in: stop everything immediately."""
        for t in (self._tts_task, self._play_task):
            if t and not t.done():
                t.cancel()
        self._player.stop()
        self._token_buf = []
        self._done_event.set()

    def shutdown(self):
        self.interrupt()

    # ── internal ──────────────────────────────────────────────────────────────

    def _flush(self):
        """Fire a TTS synthesis task for current buffer contents immediately."""
        text = "".join(self._token_buf).strip()
        self._token_buf = []
        if not text or not self._play_q:
            return
        # create synthesis task — runs concurrently, result queued for ordered play
        task = self._loop.create_task(self._synthesize(text))
        # schedule it into the ordered play queue (thread-safe)
        self._loop.call_soon_threadsafe(self._play_q.put_nowait, task)

    async def _synthesize(self, text: str) -> list[bytes]:
        """
        Synthesize text → return list of WAV byte chunks.
        Runs concurrently with other synthesis tasks.
        """
        chunks: list[bytes] = []
        if not self._http:
            return chunks
        try:
            async with self._http.stream(
                "POST",
                f"{self._url}/synthesize/pocket_stream",
                json={"text": text, "speed": self._speed},
                timeout=15.0,
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        p = json.loads(line[5:])
                    except json.JSONDecodeError:
                        continue
                    t = p.get("type", "")
                    if t == "audio_chunk":
                        chunks.append(base64.b64decode(p["data"]))
                    elif t in ("done", "error"):
                        break
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        return chunks

    async def _ordered_player(self):
        """
        Drains the play_q in order.
        Each item is either:
          - an asyncio.Task[list[bytes]]  → await it, play result
          - None                          → all done, set done_event
        """
        try:
            while True:
                item = await self._play_q.get()
                if item is None:
                    break
                try:
                    wav_chunks: list[bytes] = await item
                    for wav in wav_chunks:
                        self._player.feed(wav)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass
        except asyncio.CancelledError:
            pass
        finally:
            self._done_event.set()


# =============================================================================
# UI
# =============================================================================
_print_lock = threading.Lock()


def _p(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


class UI:
    _status_line = False

    @staticmethod
    def _clear():
        if UI._status_line:
            print(f"\r{' ' * 72}\r", end="", flush=True)
            UI._status_line = False

    @staticmethod
    def header():
        _p(f"\n{BLD}{B}{'═' * 64}{RST}")
        _p(f"{BLD}{B} ⚡ ZERO-LATENCY VOICE AGENT — IMMEDIATE PER-TOKEN TTS{RST}")
        _p(f"{BLD}{B}{'═' * 64}{RST}")
        _p(f"{DIM} MIC → DeepFilter → VAD → STT → LLM → TTS{RST}")
        _p(f"{DIM} Ctrl-C to quit{RST}\n")

    @staticmethod
    def service(name: str, ok: bool, detail: str = ""):
        icon = f"{G}✅" if ok else f"{R}❌"
        _p(f"  {icon} {BLD}{name}{RST} {DIM}{detail}{RST}")

    @staticmethod
    def feature(name: str, ok: bool, detail: str = ""):
        icon = f"{G}●" if ok else f"{Y}○"
        _p(f"  {icon} {DIM}{name}: {detail}{RST}")

    @staticmethod
    def ready(cnn_on, df_on, wrtc_on, threshold):
        _p(f"\n{BLD}Pipeline:{RST}")
        UI.feature("DeepFilterNet",       df_on,   "active"   if df_on   else "pip install deepfilternet")
        UI.feature("WebRTC VAD",          wrtc_on, "active"   if wrtc_on else "pip install webrtcvad  (using energy VAD)")
        UI.feature("Adaptive EOS",        True,    "60ms energy-drop detection")
        UI.feature("Immediate TTS flush", True,    "every token fires TTS instantly — zero buffering")
        UI.feature("Parallel TTS+play",   True,    "synthesis overlaps with playback")
        UI.feature("HTTP keep-alive",     True,    "persistent connection pool")
        UI.feature("CNN detector",        cnn_on,  f"thr={threshold:.0%}" if cnn_on else "disabled")
        _p()

    @staticmethod
    def listening():
        with _print_lock:
            UI._clear()
            print(f"\r{G}🎤 Listening…{RST} ", end="", flush=True)
            UI._status_line = True

    @staticmethod
    def transcribing():
        with _print_lock:
            UI._clear()
            print(f"\r{C}📝 Transcribing…{RST} ", end="", flush=True)
            UI._status_line = True

    @staticmethod
    def thinking():
        with _print_lock:
            UI._clear()
            print(f"\r{DIM}⏳ Thinking…{RST} ", end="", flush=True)
            UI._status_line = True

    @staticmethod
    def voice_bar(rms: float):
        filled = min(24, int(rms / BARGE_RMS_MIN * 7))
        bar    = ("█" * filled).ljust(24, "░")
        with _print_lock:
            print(f"\r{Y}🎙 [{bar}] {rms:.4f}{RST}", end="", flush=True)
            UI._status_line = True

    @staticmethod
    def user(text: str, ms: float):
        with _print_lock:
            UI._clear()
            print(f"\n{C}{BLD}You:{RST} {C}{text}{RST} {DIM}({ms:.0f}ms){RST}")
            UI._status_line = False

    @staticmethod
    def ai_start():
        with _print_lock:
            UI._clear()
            print(f"\n{M}{BLD}AI:{RST} ", end="", flush=True)
            UI._status_line = False

    @staticmethod
    def ai_token(tok: str):
        print(tok, end="", flush=True)

    @staticmethod
    def ai_end():
        print()

    @staticmethod
    def barge_in(conf: float, rms: float, ms: float):
        with _print_lock:
            UI._clear()
            print(f"\n{Y}{BLD}⚡ Barge-in{RST} "
                  f"{DIM}human={conf:.0%} rms={rms:.4f} dur={ms:.0f}ms{RST}")
            UI._status_line = False

    @staticmethod
    def dropped(reason: str):
        if DEBUG:
            with _print_lock:
                UI._clear()
                print(f"{DIM} ↷ dropped: {reason}{RST}")
                UI._status_line = False

    @staticmethod
    def warn(msg: str):
        with _print_lock:
            UI._clear()
            print(f"\n{Y}⚠ {msg}{RST}")
            UI._status_line = False

    @staticmethod
    def err(msg: str):
        with _print_lock:
            UI._clear()
            print(f"\n{R}❌ {msg}{RST}")
            UI._status_line = False


# =============================================================================
# StreamingSTT — zero-copy chunk accumulation
# =============================================================================
class StreamingSTT:
    """
    Accumulates mic chunks as a list (zero-copy until finalize).
    finalize() does ONE np.concatenate then one HTTP POST.
    """

    def __init__(self, url: str, sr: int = 16000):
        self._url    = url
        self._sr     = sr
        self._chunks: list[np.ndarray] = []
        self._lock   = threading.Lock()

    def reset(self):
        with self._lock:
            self._chunks = []

    def feed_chunk(self, chunk: np.ndarray):
        with self._lock:
            self._chunks.append(chunk)

    async def finalize(self, http: httpx.AsyncClient,
                       ai_speaking: bool) -> tuple[str, float]:
        with self._lock:
            chunks       = self._chunks
            self._chunks = []
        if not chunks:
            return "", 0.0
        # single concatenation
        audio = np.concatenate(chunks).astype(np.float32)
        t0    = time.perf_counter()
        b64   = base64.b64encode(audio.tobytes()).decode()
        words: list[str] = []
        try:
            async with http.stream(
                "POST",
                f"{self._url}/transcribe/stream",
                json={
                    "audio_b64"      : b64,
                    "sample_rate"    : self._sr,
                    "ai_is_speaking" : ai_speaking,
                },
                timeout=15.0,
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        p = json.loads(line[5:])
                    except json.JSONDecodeError:
                        continue
                    t = p.get("type", "")
                    if t == "word":
                        words.append(p["word"])
                    elif t in ("done", "silence", "ai_filtered", "end"):
                        break
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        ms = (time.perf_counter() - t0) * 1000
        return " ".join(words).strip(), ms


# =============================================================================
# Voice Agent
# =============================================================================
class VoiceAgent:

    def __init__(
        self,
        stt_url   = "http://localhost:8001",
        llm_url   = "http://localhost:8002",
        tts_url   = "http://localhost:8003",
        sr        = 16000,
        vad_thr   = 0.008,
        silence_ms= 200,
        tts_speed = 1.0,
        ai_model  = None,
        ai_device = "cpu",
        ai_thr    = 0.70,
        wrtc_aggr = 2,
    ):
        self._stt_url   = stt_url
        self._llm_url   = llm_url
        self._tts_url   = tts_url
        self._sr        = sr
        self._tts_speed = tts_speed
        self._tts_sr    = 22050

        self._df        = DeepFilterSuppressor(sr)
        self._wrtc_vad  = WebRTCVAD(sr, wrtc_aggr, silence_ms, min_ms=150)
        self._evad      = EnergyVAD(sr=sr, threshold=vad_thr,
                                    silence_ms=silence_ms, min_ms=150)
        self._use_wrtc  = self._wrtc_vad.enabled
        self._stt       = StreamingSTT(stt_url, sr)
        self._cnn       = CNNDetector(ai_model or "", ai_device, ai_thr)
        self._tts: TTSClient | None = None

        # single persistent HTTP client — connection pool, keep-alive
        self._http: httpx.AsyncClient | None = None

        self._audio_q: queue.Queue[np.ndarray]              = queue.Queue(maxsize=1000)
        self._utterance_q: queue.Queue[tuple[bool, list]]   = queue.Queue(maxsize=10)

        self._running     = False
        self._ai_speaking = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pipeline: asyncio.Task | None          = None

        signal.signal(signal.SIGINT,  self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    # ── entry ─────────────────────────────────────────────────────────────────

    def run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        finally:
            self._cleanup()

    async def _main(self):
        self._http = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
                keepalive_expiry=30,
            ),
        )
        self._running = True
        UI.header()
        _p(f"{BLD}Services:{RST}")
        await self._health_check()

        self._tts = TTSClient(self._tts_url, self._tts_speed, self._tts_sr)
        self._tts.setup(self._loop, self._http)

        UI.ready(self._cnn.enabled, self._df.enabled,
                 self._wrtc_vad.enabled, self._cnn.threshold)

        threading.Thread(target=self._mic_thread, daemon=True, name="mic").start()
        threading.Thread(target=self._vad_thread, daemon=True, name="vad").start()
        asyncio.create_task(self._utterance_processor())

        UI.listening()
        while self._running:
            await asyncio.sleep(0.05)
        if self._tts:
            self._tts.shutdown()

    def _cleanup(self):
        loop = self._loop
        if not loop or loop.is_closed():
            return

        async def _close():
            if self._pipeline and not self._pipeline.done():
                self._pipeline.cancel()
                try:
                    await asyncio.wait_for(self._pipeline, timeout=0.3)
                except Exception:
                    pass
            if self._http:
                try:
                    await asyncio.wait_for(self._http.aclose(), timeout=1.0)
                except Exception:
                    pass
            tasks = [t for t in asyncio.all_tasks(loop)
                     if t is not asyncio.current_task(loop)]
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        try:
            loop.run_until_complete(_close())
        except Exception:
            pass
        finally:
            loop.close()

    # ── health check ──────────────────────────────────────────────────────────

    async def _health_check(self):
        for name, url in [("STT", self._stt_url),
                          ("LLM", self._llm_url),
                          ("TTS", self._tts_url)]:
            try:
                r  = await self._http.get(f"{url}/health", timeout=5)
                d  = r.json()
                ok = r.status_code == 200 and d.get("status") == "ok"
                UI.service(name, ok,
                           f"status={d.get('status','?')} "
                           f"backend={d.get('backend','')}")
                if name == "TTS" and ok and d.get("sample_rate"):
                    self._tts_sr = int(d["sample_rate"])
            except Exception as e:
                UI.service(name, False, str(e))
        _p()

    # ── mic thread ────────────────────────────────────────────────────────────

    def _mic_thread(self):
        def _cb(indata, frames, _t, _s):
            chunk    = indata[:, 0].copy()
            filtered = self._df.process(chunk)
            try:
                self._audio_q.put_nowait(filtered)
            except queue.Full:
                try:
                    self._audio_q.get_nowait()
                    self._audio_q.put_nowait(filtered)
                except Exception:
                    pass

        with sd.InputStream(samplerate=self._sr, channels=1,
                            dtype="float32", blocksize=512, callback=_cb):
            while self._running:
                time.sleep(0.05)

    # ── VAD thread ────────────────────────────────────────────────────────────

    def _vad_thread(self):
        vad            = self._wrtc_vad if self._use_wrtc else self._evad
        current_chunks: list[np.ndarray] = []

        while self._running:
            try:
                chunk = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if vad.is_speaking:
                UI.voice_bar(EnergyVAD.rms(chunk))
                current_chunks.append(chunk)

            utterance = vad.process(chunk)
            if utterance is None:
                if vad.is_speaking and not current_chunks:
                    current_chunks = []
                continue

            saved          = list(current_chunks)
            current_chunks = []

            # ── gate 1: duration ──────────────────────────────────────────────
            dur_ms = len(utterance) / self._sr * 1000
            if dur_ms < BARGE_MIN_MS:
                UI.dropped(f"dur={dur_ms:.0f}ms")
                self._stt.reset()
                continue

            # ── gate 2: RMS ───────────────────────────────────────────────────
            rms = EnergyVAD.rms(utterance)
            if rms < BARGE_RMS_MIN:
                UI.dropped(f"rms={rms:.4f}")
                self._stt.reset()
                continue

            # ── gate 3: CNN ───────────────────────────────────────────────────
            label, conf = self._cnn.classify_sync(utterance, self._sr)
            if label == "ai":
                UI.dropped(f"CNN=ai {conf:.0%}")
                self._stt.reset()
                continue
            if label == "uncertain" or (label == "human" and conf < self._cnn.threshold):
                if rms < BARGE_RMS_STRICT:
                    UI.dropped(f"CNN uncertain {conf:.0%}")
                    self._stt.reset()
                    continue

            is_barge = self._ai_speaking or (self._tts and self._tts.is_speaking)
            if is_barge:
                UI.barge_in(conf, rms, dur_ms)

            snapshot = saved if saved else [utterance]
            try:
                self._utterance_q.put_nowait((is_barge, snapshot))
            except queue.Full:
                try:
                    self._utterance_q.get_nowait()
                    self._utterance_q.put_nowait((is_barge, snapshot))
                except Exception:
                    pass
            self._stt.reset()

    # ── utterance processor ───────────────────────────────────────────────────

    async def _utterance_processor(self):
        while self._running:
            try:
                is_barge, chunks = self._utterance_q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            for c in chunks:
                self._stt.feed_chunk(c)

            await self._dispatch(is_barge)

            if self._pipeline:
                try:
                    await asyncio.wait_for(asyncio.shield(self._pipeline), timeout=30.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass

    # ── dispatch ──────────────────────────────────────────────────────────────

    async def _dispatch(self, is_barge_in: bool):
        if self._pipeline and not self._pipeline.done():
            self._pipeline.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self._pipeline), timeout=0.1)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        if is_barge_in and self._tts:
            self._tts.interrupt()
            self._ai_speaking = False
        self._pipeline = asyncio.create_task(self._run_pipeline())

    # ── pipeline ──────────────────────────────────────────────────────────────

    async def _run_pipeline(self):
        try:
            UI.transcribing()
            transcript, stt_ms = await self._stt.finalize(self._http, self._ai_speaking)
            if not transcript or not any(c.isalpha() for c in transcript):
                UI.listening()
                return
            UI.user(transcript, stt_ms)
            UI.thinking()
            await self._llm_and_speak(transcript)
            UI.listening()
        except asyncio.CancelledError:
            self._ai_speaking = False
        except Exception as e:
            self._ai_speaking = False
            UI.err(f"Pipeline: {e}")
            UI.listening()

    # ── LLM → TTS ─────────────────────────────────────────────────────────────

    async def _llm_and_speak(self, prompt: str):
        """
        Stream LLM tokens. Push EVERY token to TTS IMMEDIATELY.
        TTSClient fires a TTS request for each token the instant it arrives.
        Ordered playback queue ensures audio plays in correct sequence.
        Await done_event (set by asyncio.Event, no polling).
        """
        self._ai_speaking = True
        done_event        = self._tts.start_response()
        first_token       = True

        try:
            async with self._http.stream(
                "POST",
                f"{self._llm_url}/generate/stream",
                json={"query": prompt, "session_id": "voice-client"},
                timeout=30.0,
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        p = json.loads(line[5:])
                    except json.JSONDecodeError:
                        continue
                    t = p.get("type", "")
                    if t == "token":
                        tok = p.get("token", "")
                        if not tok:
                            continue
                        if first_token:
                            UI.ai_start()
                            first_token = False
                        UI.ai_token(tok)
                        # push to TTS — fires immediately, no buffering
                        self._tts.push_token(tok)
                    elif t == "done":
                        break
                    elif t == "error":
                        UI.warn(f"LLM: {p.get('message','?')}")
                        break

            # flush any remaining buffer + signal end
            self._tts.end_tokens()

            if not first_token:
                UI.ai_end()

            # wait for TTS to finish playing (event-driven, zero polling)
            await done_event.wait()

        except asyncio.CancelledError:
            self._tts.interrupt()
            self._ai_speaking = False
            raise
        except Exception as e:
            self._tts.interrupt()
            self._ai_speaking = False
            UI.err(f"LLM: {e}")
            return

        self._ai_speaking = False

    # ── signal handler ────────────────────────────────────────────────────────

    def _on_signal(self, sig, frame):
        _p(f"\n\n{Y}👋 Shutting down…{RST}")
        self._running = False
        sys.exit(0)


# =============================================================================
# CLI
# =============================================================================
def main():
    global BARGE_MIN_MS, BARGE_RMS_MIN, BARGE_RMS_STRICT, DEBUG

    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Zero-Latency Voice Agent — Immediate Per-Token TTS Pipeline",
    )
    ap.add_argument("--stt",           default="http://localhost:8001")
    ap.add_argument("--llm",           default="http://localhost:8002")
    ap.add_argument("--tts",           default="http://localhost:8003")
    ap.add_argument("--sample-rate",   default=16000,  type=int)
    ap.add_argument("--vad-thr",       default=0.008,  type=float)
    ap.add_argument("--silence-ms",    default=200,    type=int)
    ap.add_argument("--tts-speed",     default=1.0,    type=float)
    ap.add_argument("--ai-model",      default=None,   metavar="PATH")
    ap.add_argument("--ai-device",     default="cpu")
    ap.add_argument("--ai-threshold",  default=0.70,   type=float)
    ap.add_argument("--rms-min",       default=BARGE_RMS_MIN,    type=float)
    ap.add_argument("--rms-strict",    default=BARGE_RMS_STRICT, type=float)
    ap.add_argument("--dur-min-ms",    default=BARGE_MIN_MS,     type=int)
    ap.add_argument("--wrtc-aggr",     default=2,      type=int,
                    help="WebRTC VAD aggressiveness 0-3")
    ap.add_argument("--debug",         action="store_true")
    args = ap.parse_args()

    BARGE_MIN_MS     = args.dur_min_ms
    BARGE_RMS_MIN    = args.rms_min
    BARGE_RMS_STRICT = args.rms_strict
    DEBUG            = args.debug

    VoiceAgent(
        stt_url   = args.stt,
        llm_url   = args.llm,
        tts_url   = args.tts,
        sr        = args.sample_rate,
        vad_thr   = args.vad_thr,
        silence_ms= args.silence_ms,
        tts_speed = args.tts_speed,
        ai_model  = args.ai_model,
        ai_device = args.ai_device,
        ai_thr    = args.ai_threshold,
        wrtc_aggr = args.wrtc_aggr,
    ).run()


if __name__ == "__main__":
    main()