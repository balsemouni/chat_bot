"""
metrics.py — Real-time pipeline metrics collector
==================================================

Tracks:
  - STT / LLM / TTS latencies (rolling window, avg + p95)
  - Active WebSocket sessions
  - Total & today's conversations
  - GPU utilization, VRAM, temperature, power (via pynvml)

Usage:
  from metrics import metrics_store

  # In your pipeline stages:
  metrics_store.record_stt(ms)
  metrics_store.record_llm(ms)
  metrics_store.record_tts(ms)
  metrics_store.inc_sessions() / metrics_store.dec_sessions()
  metrics_store.inc_conversations()

  # GET /metrics  → returns metrics_store.snapshot()
"""

from __future__ import annotations

import time
import threading
from collections import deque
from datetime import date
from typing import Optional

# ── Optional GPU support (pynvml) ────────────────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _GPU_OK = True
except Exception:
    _GPU_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# Rolling latency tracker (thread-safe)
# ══════════════════════════════════════════════════════════════════════════════

class _LatencyRing:
    """Keep the last N latency samples; expose avg + p95."""

    def __init__(self, maxlen: int = 100):
        self._buf: deque[float] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, ms: float):
        with self._lock:
            self._buf.append(ms)

    def stats(self) -> dict:
        with self._lock:
            data = list(self._buf)
        if not data:
            return {"avg": 0, "p95": 0}
        data_sorted = sorted(data)
        avg = sum(data_sorted) / len(data_sorted)
        idx_p95 = int(len(data_sorted) * 0.95)
        p95 = data_sorted[min(idx_p95, len(data_sorted) - 1)]
        return {"avg": round(avg), "p95": round(p95)}


# ══════════════════════════════════════════════════════════════════════════════
# Activity tracker: users per hour bucket (last 12 h)
# ══════════════════════════════════════════════════════════════════════════════

class _ActivityTracker:
    """
    Records conversation starts in hourly buckets.
    Returns the last 12 h as [{"time": "HH:00", "users": N}, ...]
    """

    def __init__(self):
        # bucket key = "YYYY-MM-DD HH"  → count
        self._buckets: dict[str, int] = {}
        self._lock = threading.Lock()

    def record(self):
        key = time.strftime("%Y-%m-%d %H")
        with self._lock:
            self._buckets[key] = self._buckets.get(key, 0) + 1

    def last_12h(self) -> list[dict]:
        now = time.time()
        result = []
        for i in range(11, -1, -1):
            ts = now - i * 3600
            key = time.strftime("%Y-%m-%d %H", time.localtime(ts))
            hour_label = time.strftime("%H:00", time.localtime(ts))
            with self._lock:
                count = self._buckets.get(key, 0)
            result.append({"time": hour_label, "users": count})
        return result


# ══════════════════════════════════════════════════════════════════════════════
# GPU snapshot
# ══════════════════════════════════════════════════════════════════════════════

def _gpu_snapshot() -> Optional[dict]:
    if not _GPU_OK:
        return None
    try:
        util   = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE)
        mem    = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
        temp   = pynvml.nvmlDeviceGetTemperature(_GPU_HANDLE, pynvml.NVML_TEMPERATURE_GPU)
        power  = pynvml.nvmlDeviceGetPowerUsage(_GPU_HANDLE)       # milliwatts
        p_lim  = pynvml.nvmlDeviceGetEnforcedPowerLimit(_GPU_HANDLE)  # milliwatts

        total_gb = round(mem.total / 1024**3, 1)
        used_gb  = round(mem.used  / 1024**3, 1)
        mem_pct  = round(mem.used / mem.total * 100, 1) if mem.total else 0

        return {
            "utilization":    util.gpu,
            "memory":         used_gb,
            "total_memory":   total_gb,
            "memory_percent": mem_pct,
            "temperature":    temp,
            "power":          round(power  / 1000),   # W
            "power_limit":    round(p_lim  / 1000),   # W
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Central store (singleton)
# ══════════════════════════════════════════════════════════════════════════════

class MetricsStore:

    def __init__(self):
        self._stt = _LatencyRing()
        self._llm = _LatencyRing()
        self._tts = _LatencyRing()

        self._active_sessions  = 0
        self._total_convos     = 0
        self._today_convos     = 0
        self._today_date: date = date.today()
        self._lock             = threading.Lock()

        self._activity = _ActivityTracker()

    # ── Recording helpers ─────────────────────────────────────────────────────

    def record_stt(self, ms: float):
        self._stt.record(ms)

    def record_llm(self, ms: float):
        self._llm.record(ms)

    def record_tts(self, ms: float):
        self._tts.record(ms)

    def inc_sessions(self):
        with self._lock:
            self._active_sessions += 1

    def dec_sessions(self):
        with self._lock:
            self._active_sessions = max(0, self._active_sessions - 1)

    def inc_conversations(self):
        """Call once per completed pipeline (i.e. full user → AI turn)."""
        today = date.today()
        with self._lock:
            if today != self._today_date:
                self._today_date  = today
                self._today_convos = 0
            self._total_convos += 1
            self._today_convos += 1
        self._activity.record()

    # ── Snapshot ─────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        stt_stats = self._stt.stats()
        llm_stats = self._llm.stats()
        tts_stats = self._tts.stats()

        with self._lock:
            active   = self._active_sessions
            total    = self._total_convos
            today    = self._today_convos

        avg_total = 0
        if any(v > 0 for v in [stt_stats["avg"], llm_stats["avg"], tts_stats["avg"]]):
            avg_total = round(
                (stt_stats["avg"] + llm_stats["avg"] + tts_stats["avg"]) / 3
            )

        return {
            "overview": {
                "active_sessions":      active,
                "total_conversations":  total,
                "today_conversations":  today,
                "trend":               0,   # extend if you want delta vs. yesterday
                "userActivity":        self._activity.last_12h(),
            },
            "latency": {
                "avg_stt": stt_stats["avg"],
                "avg_llm": llm_stats["avg"],
                "avg_tts": tts_stats["avg"],
                "p95_stt": stt_stats["p95"],
                "p95_llm": llm_stats["p95"],
                "p95_tts": tts_stats["p95"],
                "avg_total": avg_total,
            },
            "gpu": _gpu_snapshot(),
        }


# ── Global singleton ──────────────────────────────────────────────────────────
metrics_store = MetricsStore()