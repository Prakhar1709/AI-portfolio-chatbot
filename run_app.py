import sys
import os
import time
import subprocess
import signal
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def main():
    print("=" * 60)
    print("🤖 AI Portfolio Chatbot Engine Launcher")
    print("=" * 60)
    print("Starting FastAPI Backend (port 8000)...")

    # Start FastAPI Backend
    backend_cmd = [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=str(BASE_DIR)
    )

    # Wait for FastAPI to initialize
    print("Waiting for FastAPI server to initialize...")
    time.sleep(3)

    print("Starting Streamlit Frontend (port 8501)...")
    # Start Streamlit Frontend
    frontend_cmd = [sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=str(BASE_DIR)
    )

    print("\n" + "=" * 60)
    print("🌟 Applications running:")
    print("   👉 Streamlit UI:       http://localhost:8501")
    print("   👉 FastAPI Swagger UI: http://localhost:8000/docs")
    print("   👉 Health Check:       http://localhost:8000/api/health")
    print("=" * 60)
    print("Press Ctrl+C to terminate both servers...\n")

    try:
        while True:
            time.sleep(1)
            # Check if any process terminated unexpectedly
            if backend_proc.poll() is not None:
                print("⚠️ FastAPI backend terminated.")
                break
            if frontend_proc.poll() is not None:
                print("⚠️ Streamlit frontend terminated.")
                break
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    finally:
        if backend_proc.poll() is None:
            backend_proc.terminate()
        if frontend_proc.poll() is None:
            frontend_proc.terminate()
        print("All processes stopped.")

if __name__ == "__main__":
    main()
