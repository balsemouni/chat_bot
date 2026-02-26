"""
buffer.py — GPU-Aware Speech Buffer
=====================================
Audio frames stored as numpy (CPU) for downstream Whisper.
Pre-roll and segment assembly are vectorized with np.concatenate.
"""

import numpy as np
import time


class SpeechBuffer:
    """
    Speech buffer — NO minimum length, accepts ALL speech.
    Optimized with pre-allocated ring pre-buffer.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        pre_ms: int = 200,
        post_ms: int = 600,
        min_speech_ms: int = 0,
        max_speech_ms: int = 10000,
    ):
        self.sample_rate       = sample_rate
        self.pre_frames        = int(sample_rate * pre_ms  / 1000)
        self.post_frames       = int(sample_rate * post_ms / 1000)
        self.min_speech_frames = int(sample_rate * min_speech_ms / 1000)
        self.max_speech_frames = int(sample_rate * max_speech_ms / 1000)

        # Ring buffer for pre-roll (fixed-size list, O(1) append)
        self._pre: list[np.ndarray] = []
        self._pre_len: int = 0

        self._speech: list[np.ndarray] = []
        self._speech_len: int = 0
        self.post_counter: int = 0
        self.recording: bool = False
        self.speech_start_time: float = 0.0

        print(f"📦 SpeechBuffer ready — no min length, max={max_speech_ms}ms")

    # -------------------------------------------------------------------
    def push(self, frame: np.ndarray, is_voice: bool) -> np.ndarray | None:
        """Push one audio frame. Returns completed segment or None."""
        # Maintain pre-roll ring
        self._pre.append(frame)
        self._pre_len += len(frame)
        while self._pre_len > self.pre_frames and self._pre:
            removed = self._pre.pop(0)
            self._pre_len -= len(removed)

        if is_voice:
            if not self.recording:
                self.recording = True
                self._speech = list(self._pre)          # copy pre-roll
                self._speech_len = self._pre_len
                self.post_counter = 0
                self.speech_start_time = time.monotonic()

            self._speech.append(frame)
            self._speech_len += len(frame)
            self.post_counter = 0

            if self._speech_len > self.max_speech_frames:
                print("⏰ Max length reached — forcing segment")
                return self._finalize()

        elif self.recording:
            self._speech.append(frame)
            self._speech_len += len(frame)
            self.post_counter += len(frame)

            if self.post_counter >= self.post_frames:
                duration_ms = self._speech_len / self.sample_rate * 1000
                print(f"✅ Utterance: {duration_ms:.0f}ms")
                return self._finalize()

        return None

    # ------------------------------------------------------------------
    def _finalize(self) -> np.ndarray | None:
        if not self._speech:
            return None
        segment = np.concatenate(self._speech)
        duration = len(segment) / self.sample_rate
        print(f"📤 Segment: {duration:.3f}s ({len(segment)} samples)")
        self._reset()
        return segment

    def _reset(self):
        self._speech = []
        self._speech_len = 0
        self.post_counter = 0
        self.recording = False
        self.speech_start_time = 0.0

    def force_end(self) -> np.ndarray | None:
        if self.recording:
            print("⚡ Force-ending segment")
            return self._finalize()
        return None

    def get_state(self) -> dict:
        duration_ms = self._speech_len / self.sample_rate * 1000 if self._speech_len else 0
        return {
            "recording":   self.recording,
            "frames":      self._speech_len,
            "duration_ms": duration_ms,
            "post_counter": self.post_counter,
            "pre_len":     self._pre_len,
        }