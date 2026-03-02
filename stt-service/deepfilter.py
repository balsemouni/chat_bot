"""
DeepFilter Noise Reducer
========================
Wraps the deepfilternet library when available.
Falls back to a transparent pass-through stub if the library
is not installed — so the rest of the pipeline never breaks.

Install real noise reduction:
    pip install deepfilternet

Leave it out for zero-latency mode (stub adds 0 ms overhead).
"""

import numpy as np


# ── Try to import the real DeepFilter library ────────────────────────────────
try:
    from df.enhance import enhance, init_df, load_audio, save_audio
    from df.io import resample
    _DEEPFILTER_AVAILABLE = True
except ImportError:
    _DEEPFILTER_AVAILABLE = False


class DeepFilterNoiseReducer:
    """
    Noise reducer backed by DeepFilterNet (if installed) or a stub.

    Parameters
    ----------
    sample_rate : target sample rate — must be 16 000 or 48 000 Hz
    device      : 'cuda' or 'cpu'  (only used by the real model)
    """

    def __init__(self, sample_rate: int = 16000, device: str = "cpu"):
        self.sample_rate = sample_rate
        self.device = device
        self._model = None
        self._df_state = None

        if _DEEPFILTER_AVAILABLE:
            try:
                self._model, self._df_state, _ = init_df()
                print("✅ DeepFilterNet loaded — real noise reduction active")
            except Exception as exc:
                print(f"⚠️  DeepFilterNet init failed ({exc}) — using stub")
        else:
            print(
                "⚠️  deepfilternet not installed — noise reduction disabled.\n"
                "    pip install deepfilternet  to enable."
            )

    # ------------------------------------------------------------------
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Denoise one chunk of float32 audio.

        Args:
            audio : 1-D float32 numpy array at self.sample_rate

        Returns:
            Denoised float32 array (same shape)
        """
        if self._model is None or self._df_state is None:
            return audio          # stub: pass-through

        try:
            # DeepFilter expects shape (1, samples) at 48 kHz
            chunk = audio.astype(np.float32)

            # Upsample to 48 kHz if needed
            if self.sample_rate != 48000:
                import torchaudio.functional as F
                import torch
                t = torch.from_numpy(chunk).unsqueeze(0)
                t = F.resample(t, self.sample_rate, 48000)
                chunk = t.squeeze(0).numpy()

            chunk_2d = chunk[np.newaxis, :]           # (1, N)
            enhanced = enhance(self._model, self._df_state, chunk_2d)

            # Downsample back
            if self.sample_rate != 48000:
                t = torch.from_numpy(enhanced.squeeze(0)).unsqueeze(0)
                t = F.resample(t, 48000, self.sample_rate)
                enhanced = t.squeeze(0).numpy()[np.newaxis, :]

            return enhanced.squeeze(0).astype(np.float32)

        except Exception as exc:
            # Never crash the VAD pipeline on noise-reduction failure
            print(f"⚠️  DeepFilter process error: {exc}")
            return audio

    @property
    def available(self) -> bool:
        return self._model is not None