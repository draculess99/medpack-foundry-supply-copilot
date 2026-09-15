import os
import sys
import subprocess
import time
import webbrowser

# Automatically load variables from .env file into the environment
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

from backend.data_loader import ensure_directories, load_or_generate_data, generate_inventory_state
from backend.model import train_model
from backend.traceability import ensure_traceability_fields

# Default Ports
# Railway sets $PORT automatically for the web process
FRONTEND_PORT = int(os.environ.get("PORT", os.environ.get("MEDPACK_FRONTEND_PORT", 8503)))
BACKEND_PORT = int(os.environ.get("MEDPACK_BACKEND_PORT", 5001))
FRONTEND_HOST = os.environ.get("MEDPACK_FRONTEND_HOST", "0.0.0.0") # Must bind to 0.0.0.0 in production
API_BASE_URL = os.environ.get("MEDPACK_API_BASE_URL", f"http://127.0.0.1:{BACKEND_PORT}")

def main():
    global FRONTEND_PORT
    print("====================================================")
    print("   Starting MedPack AI / MedAIM Setup & Services    ")
    print("====================================================")
    
    # 1. Ensure all directories exist
    ensure_directories()
    
    # 2. Build or load processed dataset
    print("[1/3] Checking training datasets...")
    df = load_or_generate_data()
    generate_inventory_state()
    ensure_traceability_fields()
    print(f"Dataset ready. Total records: {len(df)}")
    
    # 3. Train ML Model if missing
    print("[2/3] Checking XGBoost / fallback ML Model...")
    model_file = "models/supply_demand_xgboost.pkl"
    if not os.path.exists(model_file):
        train_model()
    else:
        print("Model already exists. Skipping training.")
        
    # 4. Start Flask backend
    print(f"[3/3] Launching Flask API Backend on port {BACKEND_PORT}...")
    env = os.environ.copy()
    env["MEDPACK_BACKEND_PORT"] = str(BACKEND_PORT)
    
    # CRITICAL FIX: Railway injects $PORT which is meant for the frontend.
    # If we pass $PORT to Flask, Flask binds to it instead of Streamlit!
    if "PORT" in env:
        del env["PORT"]
        
    # Freeze Fix v3: keep the demo committee local and non-streaming even if
    # the user's .env contains USE_LLM_AGENTS=true and API keys.
    env["MEDPACK_FORCE_LOCAL_COMMITTEE"] = "true"
    env["MEDPACK_ALLOW_FULL_COMMITTEE_ROUTE"] = "false"
    env["MEDPACK_ALLOW_COMMITTEE_STREAM"] = "false"
    env["USE_LLM_AGENTS"] = "false"
    env["DEFAULT_AGENT_MODE"] = "local"
    
    # Launch server
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "backend.server"],
        env=env
    )
    print(f"[startup] backend process launched PID={backend_proc.pid}")

    # 5. Start Streamlit frontend FIRST so Railway's port binding check passes
    # (Railway requires the process to bind $PORT within 60s of start)
    
    is_railway = "PORT" in os.environ or "RAILWAY_ENVIRONMENT" in os.environ
    if not is_railway:
        import socket
        base_port = FRONTEND_PORT
        for test_port in range(base_port, base_port + 10):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', test_port)) != 0:
                    if test_port != base_port:
                        print(f"[startup] Port {base_port} already in use.")
                        print(f"[startup] Using available local Streamlit port {test_port}.")
                    FRONTEND_PORT = test_port
                    break
        else:
            print(f"[startup] Warning: Could not find an open local port in range {base_port}-{base_port+9}")

    print(f"Launching Streamlit Dashboard on port {FRONTEND_PORT}...")
    env["MEDPACK_API_BASE_URL"] = API_BASE_URL

    # We can pass port and address directly to Streamlit
    frontend_proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "frontend/dashboard.py",
            "--server.port", str(FRONTEND_PORT),
            "--server.address", FRONTEND_HOST,
            "--server.headless", "true"
        ],
        env=env
    )

    # Wait for Flask backend to be ready (up to 60s) before declaring success
    print(f"Waiting for Flask backend on port {BACKEND_PORT}...")
    import urllib.request
    health_url = f"http://127.0.0.1:{BACKEND_PORT}/health"
    for attempt in range(60):
        if backend_proc.poll() is not None:
            print(f"FATAL: Backend process terminated unexpectedly with exit code {backend_proc.returncode} during startup.")
            break
            
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    print(f"Backend ready after {attempt + 1}s.")
                    break
        except Exception:
            pass
        time.sleep(1)
    else:
        print("WARNING: Backend did not respond within 60s — continuing anyway.")

    print("\n----------------------------------------------------")
    print(f"MedPack AI backend running at http://127.0.0.1:{BACKEND_PORT}")
    print(f"MedPack AI dashboard running at http://127.0.0.1:{FRONTEND_PORT}")
    print("----------------------------------------------------")
    print("Press Ctrl+C to terminate both servers.")
    print("----------------------------------------------------")

    # Auto-open browser (skip on Railway / production where PORT is set)
    if not os.environ.get("RAILWAY_ENVIRONMENT") and not os.environ.get("PORT"):
        url = f"http://127.0.0.1:{FRONTEND_PORT}"
        print(f"Opening browser at {url} ...")
        time.sleep(2)  # give Streamlit a moment to bind
        webbrowser.open(url)
    
    # Monitor and auto-restart processes if they crash
    # (On Railway, a crash here would exit app.py and trigger a full redeploy)
    frontend_restarts = 0
    backend_restarts = 0
    MAX_RESTARTS = 5

    try:
        while True:
            # Check Flask backend
            if backend_proc.poll() is not None:
                print(f"Backend terminated (restart #{backend_restarts + 1}).")
                if backend_restarts < MAX_RESTARTS:
                    backend_restarts += 1
                    restart_env = env.copy()
                    if "PORT" in restart_env:
                        del restart_env["PORT"]
                    backend_proc = subprocess.Popen(
                        [sys.executable, "-m", "backend.server"],
                        env=restart_env
                    )
                    print("Backend restarted.")
                else:
                    print("Backend exceeded max restarts. Exiting.")
                    break

            # Check Streamlit frontend
            if frontend_proc.poll() is not None:
                print(f"Frontend terminated (restart #{frontend_restarts + 1}).")
                if frontend_restarts < MAX_RESTARTS:
                    frontend_restarts += 1
                    frontend_proc = subprocess.Popen(
                        [
                            sys.executable, "-m", "streamlit", "run", "frontend/dashboard.py",
                            "--server.port", str(FRONTEND_PORT),
                            "--server.address", FRONTEND_HOST,
                            "--server.headless", "true"
                        ],
                        env=env
                    )
                    print("Frontend restarted.")
                else:
                    print("Frontend exceeded max restarts. Exiting.")
                    break

            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Services stopped.")


if __name__ == "__main__":
    main()
