"""
asr.py — Streaming Speech Recogniser
=====================================
Wraps faster-whisper with:
  • real-time word-by-word generator  (transcribe_streaming)
  • optional AI voice filtering       (skips TTS feedback)
  • legacy full-text method           (transcribe)

Matches the constructor signature expected by main.py:
    StreamingSpeechRecognizer(
        model_size, device,
        ai_detector_model_path, enable_ai_filtering,
        compute_type,           # NEW – passed from main.py env var
        num_workers,            # NEW – passed from main.py env var
    )
"""

from faster_whisper import WhisperModel
import numpy as np

# ── AI voice detector — soft import so the service starts even without it ──
AIVoiceDetector = None
try:
    from core.ai_voice_detector import AIVoiceDetector
except ImportError:
    try:
        from ai_voice_detector import AIVoiceDetector
    except ImportError:
        pass   # detector simply won't be available


class StreamingSpeechRecognizer:
    """
    Transcribes audio to text with real-time word streaming.
    Includes optional AI voice detection to prevent transcribing TTS output.
    """

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cuda",
        ai_detector_model_path: str = None,
        ai_detection_threshold: float = 0.7,
        enable_ai_filtering: bool = True,
        # ── extra params forwarded from main.py ──────────────────────────
        compute_type: str = None,       # None → auto-select
        num_workers: int = 4,           # parallel Whisper workers
    ):
        # ── resolve compute type ─────────────────────────────────────────
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        print(f"Loading Whisper '{model_size}' on {device.upper()} "
              f"[{compute_type}, {num_workers} workers]...")

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            num_workers=num_workers,
        )

        # ── AI voice detector ────────────────────────────────────────────
        self.ai_detector = None
        self.enable_ai_filtering = enable_ai_filtering and (AIVoiceDetector is not None)

        if self.enable_ai_filtering and ai_detector_model_path:
            try:
                self.ai_detector = AIVoiceDetector(
                    model_path=ai_detector_model_path,
                    device=device,
                    confidence_threshold=ai_detection_threshold,
                )
                print("✅ AI voice filtering enabled in ASR")
            except Exception as exc:
                print(f"⚠️  AI detector failed in ASR: {exc}")
                self.ai_detector = None
        elif enable_ai_filtering and AIVoiceDetector is None:
            print("⚠️  AIVoiceDetector not importable — AI filtering disabled")

        print("✅ ASR ready")

    # ── streaming ────────────────────────────────────────────────────────

    def transcribe_streaming(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """
        Generator — yields individual words as Whisper decodes them.
        Returns immediately (yields nothing) if AI voice is detected.
        """
        if self._is_ai_audio(audio_data, sample_rate):
            return

        segments, _ = self.model.transcribe(
            audio_data,
            beam_size=1,
            word_timestamps=True,
        )

        for segment in segments:
            if hasattr(segment, "words") and segment.words:
                for word_info in segment.words:
                    word = word_info.word.strip()
                    if word:
                        yield word
            else:
                # Fallback: split segment text on whitespace
                for word in segment.text.strip().split():
                    if word:
                        yield word

    # ── legacy / blocking ────────────────────────────────────────────────

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Returns the full transcription as a single string.
        Returns "" if AI voice is detected.
        """
        if self._is_ai_audio(audio_data, sample_rate):
            return ""

        segments, _ = self.model.transcribe(audio_data, beam_size=1)
        return " ".join(s.text for s in segments).strip()

    # ── internal ─────────────────────────────────────────────────────────

    def _is_ai_audio(self, audio: np.ndarray, sr: int) -> bool:
        """Returns True if the audio should be skipped (AI-generated)."""
        if not self.enable_ai_filtering or self.ai_detector is None:
            return False
        try:
            is_ai, confidence, _ = self.ai_detector.is_ai_voice(audio, sr)
            if is_ai:
                print(f"🤖 AI voice detected in ASR ({confidence:.1%}) — skipping")
            return is_ai
        except Exception as exc:
            print(f"⚠️  AI detection error in ASR: {exc}")
            return False   # safer default: transcribe anyway