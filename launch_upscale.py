from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser


ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
REQUIREMENTS_FILE = ROOT_DIR / "requirements.txt"
STAMP_FILE = VENV_DIR / ".requirements.sha256"
APP_URL = "http://127.0.0.1:8000"
HEALTH_URL = f"{APP_URL}/api/health"


def sha256_file(path: Path) -> str:
    digest = sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def is_server_running() -> bool:
    try:
        with urlopen(HEALTH_URL, timeout=1.0) as response:
            return response.status == 200
    except URLError:
        return False


def ensure_virtualenv() -> None:
    if VENV_PYTHON.exists():
        return

    subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)], cwd=ROOT_DIR)


def ensure_requirements() -> None:
    expected_hash = sha256_file(REQUIREMENTS_FILE)
    if STAMP_FILE.exists() and STAMP_FILE.read_text(encoding="utf-8") == expected_hash:
        return

    subprocess.check_call(
        [str(VENV_PYTHON), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
        cwd=ROOT_DIR,
    )
    STAMP_FILE.write_text(expected_hash, encoding="utf-8")


def start_server() -> None:
    creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    subprocess.Popen(
        [
            str(VENV_PYTHON),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=ROOT_DIR,
        creationflags=creation_flags,
    )


def wait_for_server(timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if is_server_running():
            return
        time.sleep(0.5)

    raise RuntimeError("The UpScale server did not start within 30 seconds.")


def main() -> int:
    ensure_virtualenv()
    ensure_requirements()

    if not is_server_running():
        start_server()
        wait_for_server()

    webbrowser.open_new_tab(APP_URL)
    print(f"UpScale is running at {APP_URL}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - launcher failure path
        print(f"Unable to launch UpScale: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc