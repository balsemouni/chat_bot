"""
Automatic Gain Control (AGC)
============================
Smoothly normalises microphone volume so quiet speakers are
boosted and clipping is avoided. Runs synchronously inside
the VAD pipeline (no latency impact).
"""

import numpy as np


class SimpleAGC:
    """
    Single-pole IIR AGC.

    target_rms  – desired RMS level after processing  (0.015 ≈ -36 dBFS)
    smoothing   – how slowly the gain changes          (0 = instant, 0.99 = very slow)
    max_gain    – hard cap to avoid amplifying silence (10× = +20 dB)
    """

    def __init__(
        self,
        target_rms: float = 0.015,
        smoothing: float = 0.95,
        max_gain: float = 10.0,
    ):
        self.target_rms = target_rms
        self.smoothing = smoothing
        self.max_gain = max_gain
        self._gain = 1.0          # running gain estimate

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply AGC to a chunk of float32 audio.

        Args:
            audio: 1-D float32 numpy array (any length)

        Returns:
            Gain-adjusted audio, clipped to [-1, 1]
        """
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2) + 1e-10))

        if rms > 0:
            desired = min(self.target_rms / rms, self.max_gain)
            # Smooth gain change to avoid clicks
            self._gain = (
                self.smoothing * self._gain
                + (1.0 - self.smoothing) * desired
            )

        return np.clip(audio * self._gain, -1.0, 1.0).astype(np.float32)

    @property
    def current_gain(self) -> float:
        return self._gain