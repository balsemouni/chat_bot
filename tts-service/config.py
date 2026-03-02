"""
config.py — Centralized configuration via environment variables.
All os.getenv calls are consolidated here. Import `settings` everywhere.
"""

import os


class Settings:
    TTS_BACKEND: str = os.getenv("TTS_BACKEND", "pocket")
    DEVICE: str = os.getenv("DEVICE", "cpu")
    POCKET_VOICE: str = os.getenv("POCKET_VOICE", "alba")
    PIPER_MODEL: str = os.getenv("PIPER_MODEL", "en_US-lessac-medium.onnx")
    PIPER_CONFIG: str | None = os.getenv("PIPER_CONFIG", None)
    TTS_MODEL: str = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    SPEAKER_SAMPLE_RATE: int = int(os.getenv("SPEAKER_SAMPLE_RATE", "24000"))


settings = Settings()