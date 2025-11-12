#!/usr/bin/env python3
import os, sys, json, time, base64
from pathlib import Path
import runpod
from runpod.error import AuthenticationError

# --- Load .env files FIRST (before credential checks) ---
def _load_dotenv():
    """Load .env files into environment variables"""
    for p in (".env", ".env.runpod", os.path.expanduser("~/.runpod.env")):
        pth = Path(p).expanduser()
        if pth.is_file():
            for line in pth.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# Load .env BEFORE checking credentials
_load_dotenv()

# --- credentials loader/saver (fallback if .env doesn't exist) ---
CONFIG_JSON = Path.home() / ".config/gap/runpod.json"
FALLBACK_DOTENV = Path.home() / ".runpod.env"
LOCAL_DOTENV = Path(".runpod.local")

def _parse_env_file(p: Path) -> dict:
    data = {}
    try:
        for line in p.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            data[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return data

def _clean(s: str | None) -> str | None:
    return s.strip().strip('"').strip("'") if isinstance(s, str) else s

def _load_saved_creds() -> tuple[str | None, str | None]:
    # 1) JSON config
    if CONFIG_JSON.is_file():
        try:
            j = json.loads(CONFIG_JSON.read_text())
            return _clean(j.get("RUNPOD_API_KEY")), _clean(j.get("RUNPOD_ENDPOINT_ID"))
        except Exception:
            pass
    # 2) ~/.runpod.env (KEY=VALUE)
    if FALLBACK_DOTENV.is_file():
        d = _parse_env_file(FALLBACK_DOTENV)
        return _clean(d.get("RUNPOD_API_KEY")), _clean(d.get("RUNPOD_ENDPOINT_ID"))
    # 3) repo-local .runpod.local (gitignored)
    if LOCAL_DOTENV.is_file():
        d = _parse_env_file(LOCAL_DOTENV)
        return _clean(d.get("RUNPOD_API_KEY")), _clean(d.get("RUNPOD_ENDPOINT_ID"))
    return None, None

def _save_creds(api_key: str, endpoint_id: str):
    CONFIG_JSON.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_JSON.write_text(json.dumps({
        "RUNPOD_API_KEY": api_key,
        "RUNPOD_ENDPOINT_ID": endpoint_id
    }, indent=2))
    # Best-effort perms
    try:
        os.chmod(CONFIG_JSON, 0o600)
    except Exception:
        pass

def _ensure_creds() -> tuple[str, str]:
    # 0) env vars take precedence (now includes .env files from _load_dotenv)
    api = os.environ.get("RUNPOD_API_KEY", "").strip().strip('"').strip("'")
    eid = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip().strip('"').strip("'")
    if api and eid:
        return api, eid
    # 1) load from saved files
    api2, eid2 = _load_saved_creds()
    if api2 and eid2:
        return api2, eid2
    # 2) prompt once and save
    if sys.stdin.isatty():
        print("RunPod credentials not found. Enter once to save for future runs.")
        api2 = input("RUNPOD_API_KEY: ").strip()
        eid2 = input("RUNPOD_ENDPOINT_ID: ").strip()
        if not api2 or not eid2:
            raise SystemExit("Both API key and Endpoint ID are required.")
        _save_creds(api2, eid2)
        print(f"Saved credentials to {CONFIG_JSON}")
        return api2, eid2
    raise SystemExit("RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID not set and no saved credentials found.")
# --- end creds block ---

# Defaults and options
MANIFEST_PATH = os.environ.get("GAP_MANIFEST", "output/gaps/supplement_mv/multiview_manifest.json")
OUT_PATH = os.environ.get("GAP_VIDEO_OUT", "output/videos/supplement_mv_turntable.gif")

PROMPT = os.environ.get("GAP_PROMPT", "studio product rotation, clean white background, soft shadow")
FRAMES = int(os.environ.get("GAP_FRAMES", "24"))
STEPS = int(os.environ.get("GAP_STEPS", "25"))
GUIDANCE = float(os.environ.get("GAP_GUIDANCE", "7.5"))
CN_SCALE = float(os.environ.get("GAP_CN_SCALE", "0.6"))
FPS = int(os.environ.get("GAP_FPS", "8"))
FORMAT = os.environ.get("GAP_FORMAT", "gif")

POLL_INTERVAL = 5
MAX_ENDPOINT_WAIT = 300
MAX_JOB_WAIT = 1800

def fail(msg):
    print(msg)
    raise SystemExit(1)

def get_endpoint_info(eid: str):
    try:
        eps = runpod.get_endpoints()
    except Exception as e:
        print(f"Could not list endpoints yet: {e}")
        return None
    for ep in eps or []:
        if ep.get("id") == eid:
            return ep
    return None

def has_running_worker(info):
    workers = info.get("workers") or []
    return any(w.get("status") == "RUNNING" for w in workers)

def wait_for_endpoint(endpoint_id: str):
    print(f"Checking endpoint readiness: {endpoint_id}")
    start = time.time()
    while time.time() - start < MAX_ENDPOINT_WAIT:
        try:
            info = get_endpoint_info(endpoint_id)
        except AuthenticationError:
            # Credentials are invalid; surface a clear error immediately.
            fail("Unauthorized: Your RUNPOD_API_KEY is invalid or expired. Update credentials and retry.")
        if info:
            if info.get("status") == "FAILED":
                fail("Endpoint status=FAILED.")
            if has_running_worker(info):
                print("Endpoint has a running worker.")
                return
            print("Rollout in progress... no RUNNING workers yet.")
        else:
            print("Endpoint not visible yet.")
        time.sleep(POLL_INTERVAL)
    fail("Timeout waiting for endpoint workers.")

def poll_job(job):
    print("Polling job status...")
    start = time.time()
    while time.time() - start < MAX_JOB_WAIT:
        status = job.status()
        if status in ("COMPLETED", "FAILED", "CANCELLED"):
            return job.output()
        print(f"Status: {status}")
        time.sleep(POLL_INTERVAL)
    fail("Timeout waiting for job completion.")

def main():
    api_key, endpoint_id = _ensure_creds()
    if len(endpoint_id) < 6:
        fail("RUNPOD_ENDPOINT_ID seems too short.")

    runpod.api_key = api_key

    # Validate API key early and allow interactive re-entry if invalid.
    try:
        _ = runpod.get_endpoints()
    except AuthenticationError:
        if sys.stdin.isatty():
            print("The saved API key is unauthorized. Please enter a new one (will be saved).")
            new_key = input("RUNPOD_API_KEY: ").strip().strip('"').strip("'")
            if not new_key:
                fail("No API key provided.")
            # persist and set
            _save_creds(new_key, endpoint_id)
            runpod.api_key = new_key
            try:
                _ = runpod.get_endpoints()
            except AuthenticationError:
                fail("Provided API key still unauthorized. Double-check from RunPod Dashboard > API Keys.")
        else:
            fail("Unauthorized API key and cannot prompt in non-interactive mode. Update ~/.config/gap/runpod.json.")

    if not Path(MANIFEST_PATH).is_file():
        fail(f"Manifest not found: {MANIFEST_PATH}")

    wait_for_endpoint(endpoint_id)

    rp = runpod.Endpoint(endpoint_id)
    manifest_bytes = Path(MANIFEST_PATH).read_bytes()
    job_payload = {
        "input": {
            "manifest_b64": base64.b64encode(manifest_bytes).decode(),
            "prompt": PROMPT,
            "frames": FRAMES,
            "steps": STEPS,
            "guidance": GUIDANCE,
            "controlnet_scale": CN_SCALE,
            "fps": FPS,
            "format": FORMAT,
        }
    }

    print("Submitting job...")
    job = rp.run(job_payload)
    print("Job ID:", job.job_id)

    output = poll_job(job)
    if output is None:
        fail("Job output is None.")

    if not isinstance(output, dict):
        fail(f"Unexpected raw output: {output}")

    if output.get("status") != "ok":
        print("Job error response:")
        print(json.dumps(output, indent=2))
        fail("Job failed.")

    b64_data = output.get("video_b64")
    if not b64_data:
        fail("Missing video_b64 in response.")

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(b64_data))
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
