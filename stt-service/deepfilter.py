"""
deepfilter.py  — v2  TRUE ASYNC CUDA, ZERO BLOCKING LATENCY
=============================================================

HOW IT WORKS:
  - process() returns the PREVIOUS chunk's denoised audio (or raw if first call).
  - Current chunk is submitted to background CUDA thread immediately.
  - Main thread NEVER waits → latency added = 0ms.
  - Quality: one chunk of lag (32ms at 512 samples / 16kHz) — imperceptible.

  passthrough_mode=True  → instant return of raw audio (bypass entirely)
  passthrough_mode=False → real async CUDA denoising (recommended)
"""

import threading
import numpy as np
import torch
from queue import Queue, Empty

try:
    from df.enhance import init_df, enhance
    from df.io import resample as df_resample
    _DF_AVAILABLE = True
except ImportError:
    _DF_AVAILABLE = False


class DeepFilterNoiseReducer:
    """
    Zero-blocking-latency DeepFilter wrapper.

    process(chunk) → returns denoised audio from PREVIOUS call.
    Current chunk sent to background CUDA thread.
    Result ready for next call.  Net blocking time: 0ms.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        chunk_size: int = 512,
        passthrough_mode: bool = True,
    ):
        self.input_sr       = sample_rate
        self.chunk_size     = chunk_size
        self.passthrough_mode = passthrough_mode
        self.model          = None
        self.running        = False

        if torch.cuda.is_available() and device == "cpu":
            device = "cuda"
        self.device = device

        if not _DF_AVAILABLE:
            print("⚠️  deepfilternet not installed — passthrough mode")
            self.passthrough_mode = True
            return

        if self.passthrough_mode:
            print("⚡ DeepFilter: PASSTHROUGH (zero latency, no denoising)")
            return

        try:
            print(f"🧠 Loading DeepFilterNet on {self.device.upper()}...")
            self.model, self.df_state, _ = init_df(post_filter=True, log_level="ERROR")
            self.model = self.model.to(self.device).eval()

            if self.device == "cuda":
                # JIT compile for max GPU throughput
                self.model = torch.jit.optimize_for_inference(
                    torch.jit.script(self.model)
                )
                # Warmup
                _w = torch.randn(1, 1, chunk_size, device=self.device)
                with torch.no_grad():
                    enhance(self.model, self.df_state, _w)
                torch.cuda.synchronize()
                print("  GPU warmed up ✅")

            self.df_sr           = self.df_state.sr()
            self.needs_resample  = (sample_rate != self.df_sr)

            # Double-buffered async queues
            # input_q  → pending chunk to denoise
            # output_q → most recent denoised result
            self._input_q  = Queue(maxsize=1)
            self._output_q = Queue(maxsize=1)
            self._last_raw = None    # returned when no denoised ready yet

            self.running = True
            threading.Thread(
                target=self._worker, daemon=True, name="deepfilter"
            ).start()

            print(f"✅ DeepFilter ASYNC CUDA (non-blocking, {self.df_sr}Hz)")

        except Exception as e:
            print(f"❌ DeepFilter failed: {e} — passthrough mode")
            self.model = None
            self.passthrough_mode = True

    # ──────────────────────────────────────────────────────────────────────────

    def _worker(self):
        """Background CUDA thread — never blocks main thread."""
        while self.running:
            try:
                raw = self._input_q.get(timeout=0.1)
            except Empty:
                continue

            try:
                chunk = raw
                if chunk.ndim > 1:
                    chunk = chunk.flatten()
                # Pad / trim to chunk_size
                n = self.chunk_size
                if len(chunk) < n:
                    chunk = np.pad(chunk, (0, n - len(chunk)))
                else:
                    chunk = chunk[:n]

                t = torch.from_numpy(chunk).float()

                if self.needs_resample:
                    t = df_resample(t, self.input_sr, self.df_sr)

                t = t.to(self.device)

                with torch.no_grad():
                    enhanced = enhance(
                        self.model, self.df_state,
                        t.unsqueeze(0).unsqueeze(0)
                    ).squeeze(0).squeeze(0)

                if self.needs_resample:
                    enhanced = df_resample(enhanced.cpu(), self.df_sr, self.input_sr)
                    result = enhanced.numpy()
                else:
                    result = enhanced.cpu().numpy()

                # Put result, drop stale
                try:
                    self._output_q.get_nowait()
                except Empty:
                    pass
                self._output_q.put_nowait(result)

            except Exception:
                pass   # silently continue

    # ──────────────────────────────────────────────────────────────────────────

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Zero-blocking-latency call.

        Submits audio_chunk to background GPU thread.
        Returns previously denoised audio (or raw if not ready yet).
        """
        if len(audio_chunk) == 0:
            return audio_chunk

        if self.passthrough_mode or self.model is None:
            return audio_chunk

        # Submit current chunk (non-blocking — drop stale)
        try:
            self._input_q.get_nowait()
        except Empty:
            pass
        try:
            self._input_q.put_nowait(audio_chunk.copy())
        except Exception:
            pass

        # Return most recent denoised result (non-blocking)
        try:
            denoised = self._output_q.get_nowait()
            self._last_raw = denoised
            # Match length to input
            if len(denoised) > len(audio_chunk):
                return denoised[:len(audio_chunk)]
            return denoised
        except Empty:
            # No result ready yet — return raw audio (0ms latency)
            return audio_chunk

    def flush(self):
        for q in (self._input_q, self._output_q):
            while True:
                try:
                    q.get_nowait()
                except Empty:
                    break

    def __call__(self, audio_chunk: np.ndarray) -> np.ndarray:
        return self.process(audio_chunk)

    def __del__(self):
        self.running = False