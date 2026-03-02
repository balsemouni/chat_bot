"""
vad.py — Voice Activity Detector (GPU, Zero-Latency)
=====================================================

PIPELINE PER CHUNK (all async, main thread < 0.5ms):
  1. AGC          — normalize volume (CPU, fast)
  2. RMS check    — hard noise floor, skip silent/noisy frames immediately
  3. AI detector  — rolling CNN check (background thread, cached result)
  4. DeepFilter   — noise reduction (background CUDA thread, cached result)
  5. Silero VAD   — voice probability (background CUDA stream, cached result)
  6. Decision     — hysteresis + consecutive-voice gate (CPU, microseconds)
  7. SpeechBuffer — accumulate frames and emit complete utterances

KEY IMPROVEMENTS vs previous version:
  - noise_floor: Hard RMS gate rejects fan hum / mic hiss before Silero sees it.
    Value doubles when AI is speaking to prevent the AI from hearing itself.
  - consecutive_voice >= MIN_VOICE_FRAMES: We require N consecutive frames of
    voice before starting a recording. A single loud click or pop no longer
    triggers transcription.
  - consecutive_silence gate while recording: Once we ARE recording we stay
    lenient — silence needs to persist for several frames before we stop.
    This prevents words from being clipped at the end.

Returns 6-tuple (unchanged API):
  segment, is_voice, prob, rms_val, show_bar, barge_in_allowed
"""

import threading
from queue import Queue, Empty
from collections import deque

import numpy as np
import torch

from agc import SimpleAGC
from buffer import SpeechBuffer

# ── Optional imports ──────────────────────────────────────────────────────────
_DEEPFILTER_OK = False
try:
    from deepfilter import DeepFilterNoiseReducer
    _DEEPFILTER_OK = True
except Exception:
    pass

_AIVoiceDetector = None
_AI_DETECTOR_AVAILABLE = False
try:
    from ai_voice_detector import AIVoiceDetector as _AIVoiceDetector
    _AI_DETECTOR_AVAILABLE = True
except ImportError:
    pass


# ── Tuning constants ──────────────────────────────────────────────────────────

# How many consecutive voice frames are required before we start recording.
# Prevents random noise spikes from being treated as the start of speech.
MIN_VOICE_FRAMES = 3

# How many consecutive silence frames we tolerate WHILE recording before we
# accept that the user has stopped speaking. Set conservatively so we don't
# clip word endings.
MAX_SILENCE_FRAMES_WHILE_RECORDING = 5


# =============================================================================
class VoiceActivityDetector:
    """
    Zero-latency VAD — all heavy work runs in background threads/CUDA streams.

    The main thread calls process_chunk() and reads CACHED results from each
    subsystem. Nothing blocks; latency is bounded by a few microseconds of
    Python overhead.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        idle_threshold: float = 0.25,       # Silero prob threshold (normal speech)
        barge_in_threshold: float = 0.55,   # Stricter threshold when AI is talking
        min_rms: float = 0.02,              # Noise floor — raise if you hear fan/hum
        enable_noise_reduction: bool = True,
        min_chunk_samples: int = 512,
        ai_detector_model_path: str = None,
        ai_detection_threshold: float = 0.70,
        enable_ai_filtering: bool = True,
        ai_window_sec: float = 0.8,
    ):
        self.sample_rate        = sample_rate
        self.min_chunk_samples  = min_chunk_samples
        self.idle_threshold     = idle_threshold
        self.barge_in_threshold = barge_in_threshold
        self.min_rms            = min_rms
        self.enable_ai_filtering = enable_ai_filtering and _AI_DETECTOR_AVAILABLE

        # Force CUDA if available
        if torch.cuda.is_available():
            device = "cuda"
        self.device = torch.device(device)
        print(f"🚀 VAD device: {self.device}")

        # ── CUDA streams (one per subsystem, no contention) ───────────────────
        if self.device.type == "cuda":
            self._silero_stream = torch.cuda.Stream()
            self._agc_stream    = torch.cuda.Stream()
            print("  ✅ Dedicated CUDA streams: silero / agc")
        else:
            self._silero_stream = None
            self._agc_stream    = None

        # ── Silero VAD ────────────────────────────────────────────────────────
        print(f"🎯 Loading Silero VAD on {str(self.device).upper()}…")
        self.vad_model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad",
            force_reload=False, verbose=False,
        )
        self.vad_model.to(self.device).eval()

        if self.device.type == "cuda":
            try:
                self.vad_model = torch.compile(
                    self.vad_model, mode="reduce-overhead", fullgraph=False
                )
                print("  ✅ Silero torch.compile(reduce-overhead)")
            except Exception:
                pass

            # Pre-allocate reusable pinned + GPU tensors (avoids malloc per chunk)
            self._pinned_buf = torch.zeros(min_chunk_samples, dtype=torch.float32).pin_memory()
            self._gpu_buf    = torch.zeros(min_chunk_samples, dtype=torch.float32, device=self.device)

            # Warmup — compile CUDA kernels now, not on first real audio
            with torch.cuda.stream(self._silero_stream):
                with torch.no_grad():
                    self.vad_model(self._gpu_buf, sample_rate)
            torch.cuda.synchronize()
            print("  ✅ Silero GPU warmed up")
        else:
            self._pinned_buf = None
            self._gpu_buf    = torch.zeros(min_chunk_samples, dtype=torch.float32)

        # Async queues for Silero (maxsize=1 → always latest frame, no queue buildup)
        self._vad_in  = Queue(maxsize=1)
        self._vad_out = Queue(maxsize=1)

        # State shared between main thread and Silero worker
        self.last_vad_prob       = 0.0
        self.consecutive_voice   = 0   # Consecutive frames classified as voice
        self.consecutive_silence = 0   # Consecutive frames classified as silence
        self.running = True

        threading.Thread(target=self._vad_worker, daemon=True, name="silero").start()
        print("✅ Silero VAD ready")

        # ── CNN AI Detector ───────────────────────────────────────────────────
        self.ai_detector    = None
        self._ai_result     = False     # Cached: True = AI voice detected
        self._ai_confidence = 0.0
        self._ai_lock       = threading.Lock()
        self._ai_busy       = False
        self._ai_window_samples = int(sample_rate * ai_window_sec)
        self._ai_rolling: deque = deque()
        self._ai_rolling_len    = 0
        self._ai_check_queue: Queue = Queue(maxsize=1)

        if self.enable_ai_filtering and ai_detector_model_path and _AIVoiceDetector:
            try:
                self.ai_detector = _AIVoiceDetector(
                    model_path=ai_detector_model_path,
                    device=str(self.device),
                    confidence_threshold=ai_detection_threshold,
                )
                threading.Thread(
                    target=self._ai_worker, daemon=True, name="ai-detect"
                ).start()
                print(f"✅ AI voice filter active (threshold={ai_detection_threshold:.0%})")
            except Exception as e:
                print(f"⚠️  AI detector failed: {e}")
        elif enable_ai_filtering and not _AI_DETECTOR_AVAILABLE:
            print("⚠️  AIVoiceDetector not available — all audio treated as Human")
        else:
            print("⚠️  No AI model path — barge-in always allowed")

        # ── AGC ───────────────────────────────────────────────────────────────
        self.agc = SimpleAGC(target_rms=0.015, sample_rate=sample_rate)

        # ── DeepFilter ────────────────────────────────────────────────────────
        self.denoiser = None
        if enable_noise_reduction:
            if _DEEPFILTER_OK:
                try:
                    self.denoiser = DeepFilterNoiseReducer(
                        sample_rate=sample_rate,
                        device=str(self.device),
                        passthrough_mode=False,
                    )
                    print("✅ DeepFilter async CUDA active")
                except Exception as e:
                    print(f"⚠️  DeepFilter failed: {e}")
            else:
                print("⚠️  deepfilternet not installed — skipping noise reduction")

        # ── Speech Buffer ─────────────────────────────────────────────────────
        # min_speech_ms defaults to 500ms inside SpeechBuffer — segments shorter
        # than that are rejected before reaching Whisper (hallucination prevention).
        self.buffer = SpeechBuffer(sample_rate)

        print(
            f"✅ VAD pipeline ready | "
            f"idle_threshold={idle_threshold} | barge_in_threshold={barge_in_threshold} | "
            f"noise_floor={min_rms} | min_voice_frames={MIN_VOICE_FRAMES}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Background workers
    # ──────────────────────────────────────────────────────────────────────────

    def _vad_worker(self):
        """Silero VAD inference on a dedicated CUDA stream. Never blocks main thread."""
        while self.running:
            try:
                chunk_np = self._vad_in.get(timeout=0.1)
            except Empty:
                continue
            try:
                n = self.min_chunk_samples
                # Pad or trim to the expected tensor size
                if len(chunk_np) < n:
                    chunk_np = np.pad(chunk_np.astype(np.float32), (0, n - len(chunk_np)))
                else:
                    chunk_np = chunk_np[:n].astype(np.float32)

                ctx = torch.cuda.stream(self._silero_stream) if self._silero_stream else _nullctx()
                with ctx:
                    if self._pinned_buf is not None:
                        # Pinned memory → faster CPU-to-GPU transfer
                        self._pinned_buf.copy_(torch.from_numpy(chunk_np))
                        self._gpu_buf.copy_(self._pinned_buf, non_blocking=True)
                    else:
                        self._gpu_buf.copy_(torch.from_numpy(chunk_np))

                    with torch.no_grad():
                        prob = self.vad_model(self._gpu_buf, self.sample_rate)
                        if isinstance(prob, torch.Tensor):
                            prob = prob.item()

                # Replace stale result with fresh one (drop, then put)
                try:
                    self._vad_out.get_nowait()
                except Empty:
                    pass
                self._vad_out.put_nowait(prob)

            except Exception as e:
                print(f"❌ Silero worker error: {e}")

    def _ai_worker(self):
        """CNN AI-voice detection. Runs in its own thread; result is cached."""
        while self.running:
            try:
                audio = self._ai_check_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                is_ai, conf, _ = self.ai_detector.is_ai_voice(audio, self.sample_rate)
                with self._ai_lock:
                    self._ai_result     = is_ai
                    self._ai_confidence = conf
                    self._ai_busy       = False
                label = "🤖 AI → BLOCKED" if is_ai else "🧑 Human → ALLOWED"
                print(f"  CNN result: {label} ({conf:.0%})")
            except Exception as e:
                print(f"⚠️  AI worker error: {e}")
                with self._ai_lock:
                    self._ai_busy = False

    def _feed_ai_rolling(self, chunk: np.ndarray):
        """
        Maintain a rolling window of the last `ai_window_sec` seconds and
        trigger a background CNN check when the window is full.
        """
        if self.ai_detector is None:
            return

        self._ai_rolling.append(chunk)
        self._ai_rolling_len += len(chunk)

        # Trim window to max size
        while self._ai_rolling_len > self._ai_window_samples and self._ai_rolling:
            old = self._ai_rolling.popleft()
            self._ai_rolling_len -= len(old)

        # Only dispatch a new check when we have a full window and no check is running
        if self._ai_rolling_len >= self._ai_window_samples:
            with self._ai_lock:
                if self._ai_busy:
                    return
                self._ai_busy = True

            snapshot = np.concatenate(list(self._ai_rolling))
            try:
                self._ai_check_queue.get_nowait()
            except Empty:
                pass
            try:
                self._ai_check_queue.put_nowait(snapshot)
            except Exception:
                with self._ai_lock:
                    self._ai_busy = False

    # ──────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_rms(audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2) + 1e-10))

    def process_chunk(self, audio_chunk: np.ndarray, ai_is_speaking: bool = False):
        """
        Process one microphone chunk. Blocking time < 0.5ms on GPU systems.

        Returns
        -------
        segment          : np.ndarray | None   — complete utterance, or None
        is_voice         : bool                — current frame is voice
        prob             : float               — Silero voice probability (0–1)
        rms_val          : float               — RMS energy of the frame
        show_bar         : bool                — display voice activity bar in UI
        barge_in_allowed : bool                — False when AI voice detected
        """
        if len(audio_chunk) == 0:
            return None, False, 0.0, 0.0, False, False

        # ── Step 1: AGC — normalize gain ──────────────────────────────────────
        audio = self.agc.process(audio_chunk.astype(np.float32))
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # ── Step 2: RMS noise floor ───────────────────────────────────────────
        # Frames below this threshold are pure silence/hum — skip immediately.
        # The floor is doubled when AI is speaking to block mic bleed-through.
        rms_val    = self._calc_rms(audio)
        rms_floor  = self.min_rms * 2.0 if ai_is_speaking else self.min_rms
        if rms_val < rms_floor:
            # Force consecutive-voice counter to zero so noise can't accumulate
            self.consecutive_voice = 0
            return self.buffer.push(audio, False), False, self.last_vad_prob, rms_val, False, True

        # ── Step 3: Feed CNN rolling window (non-blocking) ────────────────────
        self._feed_ai_rolling(audio)

        # ── Step 4: Read cached CNN result ────────────────────────────────────
        with self._ai_lock:
            ai_detected = self._ai_result

        # If AI voice detected, block barge-in entirely
        if ai_detected:
            return None, False, self.last_vad_prob, rms_val, False, False

        # ── Step 5: DeepFilter noise reduction (non-blocking) ─────────────────
        if self.denoiser is not None:
            audio = self.denoiser.process(audio)

        # ── Step 6: Submit to Silero (non-blocking) ───────────────────────────
        try:
            self._vad_in.get_nowait()  # Drop stale frame
        except Empty:
            pass
        self._vad_in.put_nowait(audio)

        # ── Step 7: Read cached Silero probability ────────────────────────────
        try:
            prob = self._vad_out.get_nowait()
            self.last_vad_prob = prob
        except Empty:
            prob = self.last_vad_prob  # Use last known value if no new result yet

        # ── Step 8: Voice/silence decision with hysteresis ────────────────────
        # Use a stricter threshold when the AI is speaking (barge-in guard).
        threshold = self.barge_in_threshold if ai_is_speaking else self.idle_threshold

        if prob > threshold + 0.05:
            self.consecutive_voice   += 1
            self.consecutive_silence  = max(0, self.consecutive_silence - 1)
            raw_voice = True
        elif prob < threshold - 0.05:
            self.consecutive_silence += 1
            self.consecutive_voice    = max(0, self.consecutive_voice - 1)
            raw_voice = False
        else:
            # In the ambiguous band — favour whichever streak is longer
            raw_voice = self.consecutive_voice > self.consecutive_silence

        # ── Step 9: Consecutive-voice gate ────────────────────────────────────
        # We require MIN_VOICE_FRAMES consecutive voice frames before we call
        # it "speech". This kills single-frame noise pops and clicks.
        if raw_voice:
            is_voice = self.consecutive_voice >= MIN_VOICE_FRAMES
        else:
            # Already recording? Stay lenient — only stop after several silence frames.
            if self.buffer.recording and self.consecutive_silence < MAX_SILENCE_FRAMES_WHILE_RECORDING:
                is_voice = True
            else:
                is_voice = False

        # ── Step 10: Push to speech buffer ────────────────────────────────────
        segment = self.buffer.push(audio, is_voice)

        return segment, is_voice, prob, rms_val, is_voice, True

    # ──────────────────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        with self._ai_lock:
            ai_det  = self._ai_result
            ai_conf = self._ai_confidence
        return {
            "vad_prob":          self.last_vad_prob,
            "consecutive_voice": self.consecutive_voice,
            "consecutive_silence": self.consecutive_silence,
            "ai_detected":       ai_det,
            "ai_confidence":     ai_conf,
            "barge_in_allowed":  not ai_det,
            "recording":         self.buffer.recording,
        }

    def reset(self):
        """Reset all state (e.g. on session restart)."""
        self.last_vad_prob       = 0.0
        self.consecutive_voice   = 0
        self.consecutive_silence = 0
        self._ai_rolling.clear()
        self._ai_rolling_len = 0
        with self._ai_lock:
            self._ai_result     = False
            self._ai_confidence = 0.0
            self._ai_busy       = False
        for q in (self._vad_in, self._vad_out, self._ai_check_queue):
            while True:
                try:
                    q.get_nowait()
                except Empty:
                    break

    def __del__(self):
        self.running = False


# ── Utility ───────────────────────────────────────────────────────────────────

class _nullctx:
    """No-op context manager used when CUDA streams are not available."""
    def __enter__(self): return self
    def __exit__(self, *a): pass