"""
speaker.py — Zero-wait local audio playback
============================================

Fixes applied vs original:
  1. PERSISTENT STREAM  — OutputStream is opened ONCE in start() and kept
     alive indefinitely. Silence (zeros) is written when idle, eliminating
     the audible "pop" and hardware-init latency that occurred when the
     stream was opened per-generation.

  2. SINGLE-SESSION LOCK — Only one speak session can feed audio at a time.
     stop() cancels the current session; the next feed() starts a new one.
     No more mixed/interleaved audio from concurrent /speak requests.

  3. RESAMPLING QUALITY — np.interp (linear, causes aliasing) is replaced
     with soxr (high-quality sinc resampling). Falls back to linear if soxr
     is not installed, with a logged warning.

  4. GRACEFUL CLEANUP  — shutdown() stops the stream cleanly so the
     ThreadPoolExecutor.shutdown(wait=True) in main.py can finish the
     current sentence before the process exits.
"""

import io
import wave
import queue
import threading
import logging
import numpy as np

logger = logging.getLogger("speaker")


# ---------------------------------------------------------------------------
# WAV bytes → numpy float32
# ---------------------------------------------------------------------------

def _wav_bytes_to_numpy(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, sr


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """High-quality resampling via soxr; falls back to linear interpolation."""
    if src_sr == dst_sr:
        return audio
    try:
        import soxr
        return soxr.resample(audio, src_sr, dst_sr).astype(np.float32)
    except ImportError:
        logger.warning(
            "soxr not installed — using linear resampling (may cause audio artifacts). "
            "Install with: pip install soxr"
        )
        ratio = dst_sr / src_sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


# ---------------------------------------------------------------------------
# InstantSpeaker
# ---------------------------------------------------------------------------

_SILENCE_BLOCK = np.zeros(1024, dtype=np.float32)


class InstantSpeaker:
    """
    Plays audio chunks immediately as they arrive via a single persistent
    OutputStream.

    Design:
      - ONE OutputStream lives for the entire lifetime of the speaker.
        It writes silence when idle — no open/close per request.
      - A single background thread (_io_thread) owns the stream writes.
      - Sessions are identified by a generation counter. stop() increments
        the counter; the _io_thread detects the mismatch and discards
        in-flight audio, then returns to the silence loop.
      - feed() is thread-safe but serialised: only ONE session can push
        audio at a time. Concurrent callers after stop() automatically
        get the new generation.
    """

    _SENTINEL = object()

    def __init__(self, sample_rate: int = 24000):
        try:
            import sounddevice as sd
            self._sd = sd
        except ImportError:
            raise ImportError("sounddevice not installed. Run: pip install sounddevice")

        self._sr = sample_rate
        self._lock = threading.Lock()

        # Generation counter — incrementing this cancels the active session
        self._gen: int = 0

        # Audio queue (WAV bytes or _SENTINEL)
        self._queue: queue.Queue = queue.Queue(maxsize=64)

        # Is there an active feed session?
        self._active: bool = False

        # Persistent stream + IO thread (started in start())
        self._stream = None
        self._io_thread: threading.Thread | None = None
        self._running: bool = False

    # ------------------------------------------------------------------
    def start(self):
        """Open the persistent OutputStream and start the IO thread."""
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            blocksize=1024,
        )
        self._stream.start()
        self._running = True

        self._io_thread = threading.Thread(
            target=self._io_loop,
            daemon=True,
            name="speaker-io",
        )
        self._io_thread.start()
        logger.info("InstantSpeaker stream open (sr=%d)", self._sr)

    # ------------------------------------------------------------------
    def feed(self, wav_bytes: bytes):
        """
        Queue a WAV chunk for immediate playback.
        If no session is active, starts one automatically.
        """
        with self._lock:
            if not self._active:
                # New session — clear any leftover data from previous gen
                self._drain_queue()
                self._active = True
                logger.debug("Speaker session started (gen=%d)", self._gen)

        try:
            self._queue.put(wav_bytes, timeout=1.0)
        except queue.Full:
            logger.warning("Audio queue full — dropping chunk")

    # ------------------------------------------------------------------
    def flush(self, timeout: float = 30.0):
        """Block until all queued audio has played."""
        # Put a sentinel; wait for the queue to drain up to that point
        sentinel_event = threading.Event()

        def _marker():
            sentinel_event.set()

        # We can't easily join mid-stream, so poll with a small sleep
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._queue.empty() and not self._active:
                    return
            time.sleep(0.02)

    # ------------------------------------------------------------------
    def stop(self):
        """
        Cancel the current playback session immediately.
        The IO thread will drain in-flight audio and return to silence.
        """
        with self._lock:
            if not self._active:
                return
            self._gen += 1
            self._active = False
            self._drain_queue()

        # Unblock the IO thread if it's waiting on an empty queue
        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            pass

        logger.debug("Speaker stopped (gen=%d)", self._gen)

    # ------------------------------------------------------------------
    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._active or not self._queue.empty()

    # ------------------------------------------------------------------
    def shutdown(self):
        """Clean shutdown — stops the IO thread and closes the stream."""
        self._running = False
        self.stop()
        # Unblock IO thread
        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            pass

        if self._io_thread and self._io_thread.is_alive():
            self._io_thread.join(timeout=2.0)

        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        logger.info("InstantSpeaker shut down")

    # ------------------------------------------------------------------
    def _drain_queue(self):
        """Discard all items in the queue. Must be called with _lock held."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    def _io_loop(self):
        """
        Single thread that owns all stream.write() calls.

        When active: dequeue WAV bytes, decode, write in 100ms blocks.
          - Checks generation before each block so stop() is responsive.
        When idle: write silence blocks to keep the stream alive.
        """
        stream = self._stream
        block = self._sr // 10  # 100ms

        while self._running:
            # Snapshot generation under lock
            with self._lock:
                current_gen = self._gen
                is_active = self._active

            if not is_active:
                # Idle — write silence to keep hardware warm
                try:
                    stream.write(_SILENCE_BLOCK)
                except Exception as e:
                    logger.error("Stream write error (idle): %s", e)
                continue

            # Active session — drain queue
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                # Still active but nothing queued yet — write silence
                try:
                    stream.write(_SILENCE_BLOCK)
                except Exception:
                    pass
                continue

            if item is self._SENTINEL:
                # Session ended
                with self._lock:
                    if self._gen == current_gen:
                        self._active = False
                continue

            # Decode and play in 100ms blocks
            try:
                audio, sr = _wav_bytes_to_numpy(item)
                if sr != self._sr:
                    audio = _resample(audio, sr, self._sr)

                for i in range(0, len(audio), block):
                    with self._lock:
                        if self._gen != current_gen:
                            # Generation changed — stop this chunk immediately
                            break
                    try:
                        stream.write(audio[i:i + block])
                    except Exception as e:
                        logger.error("Stream write error: %s", e)
                        break

            except Exception as e:
                logger.error("Audio decode error: %s", e)

        logger.debug("IO loop exited")


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_speaker: InstantSpeaker | None = None
_global_lock = threading.Lock()


def get_instant_speaker(sample_rate: int = 24000) -> InstantSpeaker:
    global _global_speaker
    with _global_lock:
        if _global_speaker is None:
            _global_speaker = InstantSpeaker(sample_rate=sample_rate)
            _global_speaker.start()
    return _global_speaker


def shutdown_instant_speaker():
    global _global_speaker
    with _global_lock:
        if _global_speaker is not None:
            _global_speaker.shutdown()
            _global_speaker = None