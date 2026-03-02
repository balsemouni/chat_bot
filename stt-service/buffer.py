"""
Speech Segment Buffer
=====================
Accumulates raw audio chunks while the VAD reports voice activity,
then emits a single concatenated numpy array when silence begins.

Used by VoiceActivityDetector.process_chunk() — the return value
of push() is the segment to send to the STT microservice.
"""

import numpy as np
from typing import Optional


class SpeechBuffer:
    """
    State machine:  IDLE  →  SPEAKING  →  IDLE  → (emit segment)

    Parameters
    ----------
    sample_rate   : audio sample rate (default 16 000 Hz)
    min_speech_ms : minimum speech duration to emit a segment.
                    Shorter bursts are discarded as noise. (default 200 ms)
    """

    def __init__(self, sample_rate: int = 16000, min_speech_ms: int = 200):
        self.sample_rate = sample_rate
        self.min_samples = int(sample_rate * min_speech_ms / 1000)
        self._chunks: list = []
        self._in_speech: bool = False

    # ------------------------------------------------------------------
    def push(
        self, audio: np.ndarray, is_voice: bool
    ) -> Optional[np.ndarray]:
        """
        Feed one chunk into the buffer.

        Returns
        -------
        np.ndarray  – complete speech segment, ready for transcription
        None        – still accumulating, or segment was too short
        """
        if is_voice:
            self._chunks.append(audio)
            self._in_speech = True
            return None                          # still speaking

        if self._in_speech:
            # Silence just started → finalise segment
            segment = np.concatenate(self._chunks)
            self._chunks = []
            self._in_speech = False

            if len(segment) >= self.min_samples:
                return segment                   # ✅ valid utterance
            # Too short — discard silently

        return None

    # ------------------------------------------------------------------
    @property
    def is_speaking(self) -> bool:
        return self._in_speech

    @property
    def buffered_seconds(self) -> float:
        total = sum(len(c) for c in self._chunks)
        return total / self.sample_rate

    def reset(self) -> None:
        """Discard any partially buffered audio."""
        self._chunks = []
        self._in_speech = False