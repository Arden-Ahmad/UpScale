from __future__ import annotations

import argparse
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser


ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"
VENV_PYTHON = VENV_DIR / (Path("Scripts") / "python.exe" if os.name == "nt" else Path("bin") / "python")
REQUIREMENTS_FILE = ROOT_DIR / "requirements.txt"
COLAB_REQUIREMENTS_FILE = ROOT_DIR / "requirements-colab.txt"
DEFAULT_PORT = 8000


def is_running_in_colab() -> bool:
    return any(
        os.environ.get(variable)
        for variable in ("COLAB_RELEASE_TAG", "COLAB_BACKEND_VERSION", "COLAB_GPU")
    )


def sha256_file(path: Path) -> str:
    digest = sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def get_stamp_file(requirements_file: Path) -> Path:
    return VENV_DIR / f".{requirements_file.name}.sha256"


def resolve_requirements_file(requirements_arg: str | None, running_in_colab: bool) -> Path:
    if requirements_arg is not None:
        candidate = Path(requirements_arg)
        if not candidate.is_absolute():
            candidate = ROOT_DIR / candidate
        requirements_file = candidate.resolve()
    elif running_in_colab and COLAB_REQUIREMENTS_FILE.exists():
        requirements_file = COLAB_REQUIREMENTS_FILE
    else:
        requirements_file = REQUIREMENTS_FILE

    if not requirements_file.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_file}")
    return requirements_file


def build_app_url(host: str, port: int) -> str:
    browser_host = "127.0.0.1" if host == "0.0.0.0" else host
    return f"http://{browser_host}:{port}"


def build_health_url(host: str, port: int) -> str:
    return f"{build_app_url(host, port)}/api/health"


def is_server_running(host: str, port: int) -> bool:
    try:
        with urlopen(build_health_url(host, port), timeout=1.0) as response:
            return response.status == 200
    except URLError:
        return False


def ensure_virtualenv() -> None:
    if VENV_PYTHON.exists():
        return

    subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)], cwd=ROOT_DIR)


def ensure_requirements(requirements_file: Path) -> None:
    expected_hash = sha256_file(requirements_file)
    stamp_file = get_stamp_file(requirements_file)
    if stamp_file.exists() and stamp_file.read_text(encoding="utf-8") == expected_hash:
        return

    subprocess.check_call(
        [str(VENV_PYTHON), "-m", "pip", "install", "-r", str(requirements_file)],
        cwd=ROOT_DIR,
    )
    stamp_file.write_text(expected_hash, encoding="utf-8")


def build_server_command(host: str, port: int, reload_enabled: bool) -> list[str]:
    command = [
        str(VENV_PYTHON),
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload_enabled:
        command.append("--reload")
    return command


def build_server_env(device_preference: str | None) -> dict[str, str]:
    environment = os.environ.copy()
    if device_preference is not None:
        environment["UPSCALE_DEVICE"] = device_preference
    return environment


def start_server(command: list[str], environment: dict[str, str]) -> None:
    creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    subprocess.Popen(
        command,
        cwd=ROOT_DIR,
        env=environment,
        creationflags=creation_flags,
    )


def run_server_foreground(command: list[str], environment: dict[str, str]) -> int:
    completed_process = subprocess.run(command, cwd=ROOT_DIR, env=environment, check=False)
    return completed_process.returncode


def wait_for_server(host: str, port: int, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if is_server_running(host, port):
            return
        time.sleep(0.5)

    raise RuntimeError("The UpScale server did not start within 30 seconds.")


def parse_args(running_in_colab: bool) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap and run the UpScale server.")
    parser.add_argument("--host", default="0.0.0.0" if running_in_colab else "127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--reload", action="store_true", help="Start Uvicorn with autoreload enabled.")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        help="Set the UPSCALE_DEVICE value passed to the app.",
    )
    parser.add_argument(
        "--requirements-file",
        help="Install dependencies from a different requirements file before launching.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency installation and launch with the existing virtual environment.",
    )

    browser_group = parser.add_mutually_exclusive_group()
    browser_group.add_argument(
        "--browser",
        dest="open_browser",
        action="store_true",
        default=None,
        help="Open the app URL in a browser after booting.",
    )
    browser_group.add_argument(
        "--no-browser",
        dest="open_browser",
        action="store_false",
        help="Do not try to open a browser automatically.",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--foreground",
        action="store_true",
        help="Run the server in the current terminal instead of spawning a background process.",
    )
    mode_group.add_argument(
        "--background",
        action="store_true",
        help="Spawn the server as a background process and return after it becomes healthy.",
    )
    return parser.parse_args()


def main() -> int:
    running_in_colab = is_running_in_colab()
    args = parse_args(running_in_colab)
    requirements_file = resolve_requirements_file(args.requirements_file, running_in_colab)

    open_browser = args.open_browser if args.open_browser is not None else not running_in_colab
    run_in_foreground = args.foreground or (running_in_colab and not args.background)
    device_preference = args.device or ("cuda" if running_in_colab else None)

    ensure_virtualenv()
    if not args.skip_install:
        ensure_requirements(requirements_file)

    app_url = build_app_url(args.host, args.port)
    server_command = build_server_command(args.host, args.port, args.reload)
    server_environment = build_server_env(device_preference)

    if run_in_foreground:
        if is_server_running(args.host, args.port):
            print(f"UpScale is already running at {app_url}")
            return 0
        if device_preference is not None:
            print(f"Starting UpScale at {app_url} with UPSCALE_DEVICE={device_preference}")
        else:
            print(f"Starting UpScale at {app_url}")
        return run_server_foreground(server_command, server_environment)

    if not is_server_running(args.host, args.port):
        start_server(server_command, server_environment)
        wait_for_server(args.host, args.port)

    if open_browser:
        webbrowser.open_new_tab(app_url)

    print(f"UpScale is running at {app_url}")
    if device_preference is not None:
        print(f"Requested device: {device_preference}")
    if running_in_colab:
        print("Use the Colab port preview or proxy for external access.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - launcher failure path
        print(f"Unable to launch UpScale: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc