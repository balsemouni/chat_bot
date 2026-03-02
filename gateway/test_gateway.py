#!/usr/bin/env python3
"""
Check if all required services are running and accessible.
Run this to debug connectivity issues.
"""

import socket
import requests
import sys

def check_service(host, port, name):
    """Check if a service is reachable."""
    print(f"\nChecking {name} at {host}:{port}...")
    
    # Try socket connection first
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex((host, int(port)))
    sock.close()
    
    if result == 0:
        print(f"✅ {name} is reachable on port {port}")
        
        # Try HTTP health endpoint
        try:
            url = f"http://{host}:{port}/health"
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(f"✅ {name} health check passed: {r.json()}")
            else:
                print(f"⚠️ {name} health check returned {r.status_code}")
        except Exception as e:
            print(f"⚠️ {name} health check failed: {e}")
    else:
        print(f"❌ {name} is NOT reachable on port {port}")
        return False
    return True

def main():
    """Main check function."""
    print("=" * 60)
    print("Voice Gateway Service Checker")
    print("=" * 60)
    
    # Check local services (change these if running elsewhere)
    services = [
        ("localhost", "8001", "STT Service"),
        ("localhost", "8002", "LLM Service"),
        ("localhost", "8003", "TTS Service"),
    ]
    
    all_good = True
    for host, port, name in services:
        if not check_service(host, port, name):
            all_good = False
    
    print("\n" + "=" * 60)
    if all_good:
        print("✅ All services are reachable!")
    else:
        print("❌ Some services are not reachable!")
        print("\nTroubleshooting tips:")
        print("1. Make sure all services are running:")
        print("   - STT on port 8001")
        print("   - LLM on port 8002")
        print("   - TTS on port 8003")
        print("2. Check if services are running in Docker:")
        print("   docker ps | grep -E 'stt|llm|tts'")
        print("3. Check service logs:")
        print("   docker logs <container_name>")
        print("4. Try running services locally with:")
        print("   python stt_service.py")
        print("   python llm_service.py")
        print("   python tts_service.py")
    
    sys.exit(0 if all_good else 1)

if __name__ == "__main__":
    main()