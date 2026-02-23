"""
speaker.py — Zero-wait local audio playback (fixed)
====================================================

Root cause of silence bug:
  stop() set _stop_event → killed the playback loop thread permanently.
  _rearm() only cleared the flag — but never restarted the dead thread.
  All subsequent audio fed into the queue was never played.

Fix:
  - Each speak session gets its OWN thread + stream (generation based).
  - stop() aborts the current generation and marks it cancelled.
  - The next feed() call starts a fresh generation automatically.
  - No shared persistent thread that can die silently.
"""

import io
import wave
import queue
import threading
import time
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
    if src_sr == dst_sr:
        return audio
    ratio = dst_sr / src_sr
    new_len = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


# ---------------------------------------------------------------------------
# InstantSpeaker  (generation-based, self-healing)
# ---------------------------------------------------------------------------

class InstantSpeaker:
    """
    Plays audio chunks immediately as they arrive.

    Design:
      - Each "generation" is one play session (one call to speak/feed sequence).
      - stop() increments the generation counter, which kills the current
        playback loop. The old thread cleans itself up.
      - The next feed() automatically starts a new generation thread.
      - No global persistent thread that can silently die.
    """

    def __init__(self, sample_rate: int = 24000):
        try:
            import sounddevice as sd
            self._sd = sd
        except ImportError:
            raise ImportError("sounddevice not installed. Run: pip install sounddevice")

        self._sr = sample_rate
        self._lock = threading.Lock()

        # Generation counter — incrementing this kills the active loop
        self._gen = 0

        # Per-generation queue and thread
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None

        # Is any generation currently active?
        self._active = False

    # ------------------------------------------------------------------
    def start(self):
        """Called at startup — no-op since we create threads on demand."""
        pass

    # ------------------------------------------------------------------
    def feed(self, wav_bytes: bytes):
        """
        Queue a WAV chunk for immediate playback.
        Starts a new playback thread if none is running.
        """
        with self._lock:
            if self._queue is None or not self._active:
                self._start_new_generation()
            try:
                self._queue.put(wav_bytes, timeout=1.0)
            except queue.Full:
                logger.warning("Audio queue full — dropping chunk")

    # ------------------------------------------------------------------
    def flush(self, timeout: float = 30.0):
        """Block until all queued audio has played."""
        q = None
        with self._lock:
            q = self._queue
        if q is not None:
            try:
                q.join()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def stop(self):
        """
        Interrupt playback immediately.
        Kills the current generation — the playback thread will notice
        and exit cleanly. The next feed() will start a fresh one.
        """
        with self._lock:
            self._gen += 1          # invalidate current generation
            self._active = False
            q = self._queue
            self._queue = None

        # Drain the old queue so task_done counts stay balanced
        if q is not None:
            while not q.empty():
                try:
                    q.get_nowait()
                    q.task_done()
                except queue.Empty:
                    break

        # Abort any blocking sounddevice.play() calls
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

        logger.debug("Speaker stopped (gen=%d)", self._gen)

    # ------------------------------------------------------------------
    @property
    def is_speaking(self) -> bool:
        with self._lock:
            if not self._active:
                return False
            q = self._queue
        return q is not None and (not q.empty() or self._active)

    # ------------------------------------------------------------------
    def shutdown(self):
        """Clean shutdown."""
        self.stop()

    # ------------------------------------------------------------------
    def _start_new_generation(self):
        """
        Must be called with self._lock held.
        Creates a fresh queue and starts a new playback thread.
        """
        gen = self._gen
        q: queue.Queue = queue.Queue(maxsize=64)
        self._queue = q
        self._active = True

        t = threading.Thread(
            target=self._playback_loop,
            args=(gen, q),
            daemon=True,
            name=f"speaker-gen-{gen}",
        )
        self._thread = t
        t.start()
        logger.debug("Started speaker generation %d", gen)

    # ------------------------------------------------------------------
    def _playback_loop(self, my_gen: int, q: queue.Queue):
        """
        Plays audio from q until:
          - sentinel None received, OR
          - self._gen != my_gen  (stop() was called), OR
          - exception
        """
        import sounddevice as sd

        stream = None
        try:
            stream = sd.OutputStream(
                samplerate=self._sr,
                channels=1,
                dtype="float32",
                blocksize=1024,
            )
            stream.start()
            logger.debug("Playback stream open (gen=%d, sr=%d)", my_gen, self._sr)

            while True:
                # Check if we've been superseded
                with self._lock:
                    if self._gen != my_gen:
                        break

                try:
                    item = q.get(timeout=0.1)
                except queue.Empty:
                    # Check again if generation changed
                    with self._lock:
                        if self._gen != my_gen:
                            break
                    continue

                if item is None:
                    q.task_done()
                    break

                try:
                    audio, sr = _wav_bytes_to_numpy(item)
                    if sr != self._sr:
                        audio = _resample(audio, sr, self._sr)

                    # Write in small blocks so stop() can interrupt promptly
                    block = self._sr // 10  # 100ms blocks
                    for i in range(0, len(audio), block):
                        with self._lock:
                            if self._gen != my_gen:
                                break
                        stream.write(audio[i:i + block])

                except Exception as e:
                    logger.error("Playback error (gen=%d): %s", my_gen, e)
                finally:
                    q.task_done()

        except Exception as e:
            logger.error("Stream open error (gen=%d): %s", my_gen, e)
        finally:
            # Mark inactive only if we're still the current generation
            with self._lock:
                if self._gen == my_gen:
                    self._active = False
            try:
                if stream:
                    stream.stop()
                    stream.close()
            except Exception:
                pass
            logger.debug("Playback loop exited (gen=%d)", my_gen)


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