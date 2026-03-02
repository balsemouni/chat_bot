"""
GPU Stats Server
================
Exposes GPU metrics as a REST API + SSE stream.
Add to your existing gateway or run standalone.

Endpoints:
  GET /gpu/stats          → instant JSON snapshot
  GET /gpu/stream         → SSE live stream (1 update/sec)
  GET /gpu/health         → service health

Install:
  pip install fastapi uvicorn pynvml

Run standalone:
  uvicorn gpu_stats_server:app --host 0.0.0.0 --port 8004 --reload

Or import into your main.py:
  from gpu_stats_server import router as gpu_router
  app.include_router(gpu_router, prefix="/gpu")
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

logger = logging.getLogger("gpu_stats")

# ─────────────────────────────────────────────────────────────────────────────
# GPU Stats Reader (NVML-based, zero-dependency on torch)
# ─────────────────────────────────────────────────────────────────────────────

class GPUStats:
    """
    Reads GPU metrics via pynvml (NVIDIA Management Library).
    Falls back to zeros if no GPU / pynvml not installed.
    """

    def __init__(self):
        self._nvml_ok = False
        self._handle = None
        self._init_nvml()

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count > 0:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(self._handle)
                # Handle both bytes and str return types across pynvml versions
                if isinstance(name, bytes):
                    name = name.decode()
                self._nvml_ok = True
                logger.info(f"✅ NVML initialized — GPU: {name}")
            else:
                logger.warning("No NVIDIA GPUs found via NVML")
        except ImportError:
            logger.warning("pynvml not installed — run: pip install pynvml")
        except Exception as e:
            logger.warning(f"NVML init failed: {e}")

    def read(self) -> dict:
        """Return current GPU metrics as a dict."""
        if not self._nvml_ok or self._handle is None:
            return self._fallback()

        try:
            import pynvml

            # GPU utilization %
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            gpu_pct = util.gpu       # 0–100
            mem_pct = util.memory    # 0–100 (encoder/decoder, not VRAM)

            # VRAM
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            vram_used_gb  = mem_info.used  / (1024 ** 3)
            vram_total_gb = mem_info.total / (1024 ** 3)
            vram_pct = (mem_info.used / mem_info.total * 100) if mem_info.total else 0

            # Temperature
            try:
                temp_c = pynvml.nvmlDeviceGetTemperature(
                    self._handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temp_c = 0

            # Power (optional)
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                power_w  = power_mw / 1000.0
            except Exception:
                power_w = 0.0

            # Fan speed (optional)
            try:
                fan_pct = pynvml.nvmlDeviceGetFanSpeed(self._handle)
            except Exception:
                fan_pct = 0

            return {
                "available":    True,
                "gpu_pct":      gpu_pct,
                "mem_pct":      mem_pct,
                "vram_used_gb": round(vram_used_gb, 2),
                "vram_total_gb": round(vram_total_gb, 2),
                "vram_pct":     round(vram_pct, 1),
                "temp_c":       temp_c,
                "power_w":      round(power_w, 1),
                "fan_pct":      fan_pct,
                "timestamp":    time.time(),
            }

        except Exception as e:
            logger.error(f"NVML read error: {e}")
            return self._fallback()

    @staticmethod
    def _fallback() -> dict:
        return {
            "available":    False,
            "gpu_pct":      0,
            "mem_pct":      0,
            "vram_used_gb": 0.0,
            "vram_total_gb": 0.0,
            "vram_pct":     0.0,
            "temp_c":       0,
            "power_w":      0.0,
            "fan_pct":      0,
            "timestamp":    time.time(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_gpu: Optional[GPUStats] = None


def get_gpu() -> GPUStats:
    global _gpu
    if _gpu is None:
        _gpu = GPUStats()
    return _gpu


# ─────────────────────────────────────────────────────────────────────────────
# Router (plug into any existing FastAPI app)
# ─────────────────────────────────────────────────────────────────────────────

router = APIRouter(tags=["gpu"])


@router.get("/stats")
async def gpu_stats():
    """Instant GPU snapshot."""
    return get_gpu().read()


@router.get("/stream")
async def gpu_stream(interval: float = 1.0):
    """
    SSE live stream of GPU stats.
    
    Usage from your frontend:
      const es = new EventSource('/gpu/stream');
      es.onmessage = e => {
        const stats = JSON.parse(e.data);
        // stats.gpu_pct, stats.vram_used_gb, stats.vram_total_gb, stats.temp_c
      };
    """
    async def _gen():
        gpu = get_gpu()
        while True:
            data = gpu.read()
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(max(0.1, interval))

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable nginx buffering
        },
    )


@router.get("/health")
async def gpu_health():
    stats = get_gpu().read()
    return {
        "status": "ok",
        "gpu_available": stats["available"],
        "gpu_pct": stats["gpu_pct"],
        "temp_c": stats["temp_c"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone app (if you run this file directly)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gpu
    _gpu = GPUStats()
    logger.info("🚀 GPU Stats Server started")
    yield
    logger.info("👋 GPU Stats Server stopped")


app = FastAPI(
    title="GPU Stats API",
    version="1.0.0",
    description="Live GPU metrics endpoint for voice agent monitoring.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/gpu")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gpu_stats_server:app", host="0.0.0.0", port=8004, reload=True)