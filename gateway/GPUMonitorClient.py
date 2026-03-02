"""
GPU Monitor Client
==================

Simple Python client to consume GPU monitoring API endpoints.

Usage:
    python gpu_monitor_client.py

API Endpoints:
    GET  /gpu/status      - Get current GPU stats
    GET  /gpu/processes   - Get list of GPU processes
    POST /gpu/cleanup     - Trigger GPU memory cleanup
"""

import requests
import time
import json
from datetime import datetime


class GPUMonitorClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def get_status(self):
        """Get current GPU status"""
        try:
            response = requests.get(f"{self.base_url}/gpu/status", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e), "available": False}
    
    def get_processes(self):
        """Get list of GPU processes"""
        try:
            response = requests.get(f"{self.base_url}/gpu/processes", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e), "available": False}
    
    def cleanup_memory(self):
        """Trigger GPU memory cleanup"""
        try:
            response = requests.post(f"{self.base_url}/gpu/cleanup", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e), "success": False}
    
    def print_status(self, data):
        """Pretty print GPU status"""
        print("\n" + "="*60)
        print("🎮 GPU STATUS")
        print("="*60)
        
        if not data.get("available"):
            print(f"❌ GPU not available: {data.get('error', 'Unknown error')}")
            return
        
        if "error" in data:
            print(f"⚠️  Error: {data['error']}")
            return
        
        print(f"GPU: {data.get('gpu_name', 'Unknown')}")
        print(f"\n📊 GPU Utilization: {data.get('gpu_utilization_percent', 0)}%")
        print(f"🌡️  Temperature: {data.get('temperature_celsius', 0)}°C")
        
        memory = data.get('memory', {})
        used = memory.get('used_mb', 0)
        total = memory.get('total_mb', 0)
        free = memory.get('free_mb', 0)
        util = memory.get('utilization_percent', 0)
        
        print(f"\n💾 VRAM:")
        print(f"   Used:  {used:,.0f} MB")
        print(f"   Free:  {free:,.0f} MB")
        print(f"   Total: {total:,.0f} MB")
        print(f"   Usage: {util:.1f}%")
        
        # Visual bar
        bar_width = 40
        filled = int((util / 100) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"   [{bar}] {util:.1f}%")
        
        timestamp = data.get('timestamp', time.time())
        print(f"\n⏰ Last updated: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
    
    def print_processes(self, data):
        """Pretty print GPU processes"""
        print("\n" + "="*60)
        print("🔧 GPU PROCESSES")
        print("="*60)
        
        if not data.get("available"):
            print(f"❌ GPU not available: {data.get('error', 'Unknown error')}")
            return
        
        processes = data.get("processes", [])
        if not processes:
            print("✅ No GPU processes found")
            return
        
        print(f"Found {len(processes)} process(es) using GPU:\n")
        
        for i, proc in enumerate(processes, 1):
            print(f"{i}. {proc['name']}")
            print(f"   PID: {proc['pid']}")
            print(f"   Memory: {proc['mem']} MB")
            print()
        
        print("="*60)
    
    def monitor_live(self, interval=2):
        """Monitor GPU stats in real-time"""
        print("🚀 Starting live GPU monitoring (Ctrl+C to stop)")
        print(f"📊 Refresh interval: {interval}s\n")
        
        try:
            while True:
                status = self.get_status()
                self.print_status(status)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Monitor Client")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="API base URL (default: http://localhost:8000)")
    parser.add_argument("--interval", type=int, default=2,
                       help="Refresh interval in seconds for live mode (default: 2)")
    parser.add_argument("--mode", choices=["status", "processes", "cleanup", "live"],
                       default="status",
                       help="Operation mode (default: status)")
    
    args = parser.parse_args()
    
    client = GPUMonitorClient(args.url)
    
    if args.mode == "status":
        status = client.get_status()
        client.print_status(status)
    
    elif args.mode == "processes":
        processes = client.get_processes()
        client.print_processes(processes)
    
    elif args.mode == "cleanup":
        print("🧹 Triggering GPU memory cleanup...")
        result = client.cleanup_memory()
        if result.get("success"):
            print("✅ Cleanup successful!")
            if "memory_after" in result and result["memory_after"]:
                mem = result["memory_after"]
                print(f"\n💾 Memory after cleanup:")
                print(f"   Free: {mem.get('free_mb', 0):,.0f} MB")
                print(f"   Used: {mem.get('used_mb', 0):,.0f} MB")
        else:
            print(f"❌ Cleanup failed: {result.get('error', 'Unknown error')}")
    
    elif args.mode == "live":
        client.monitor_live(interval=args.interval)


if __name__ == "__main__":
    main()