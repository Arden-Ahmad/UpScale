from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import os
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser


ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQUIREMENTS_FILE = ROOT_DIR / "requirements.txt"
COLAB_REQUIREMENTS_FILE = ROOT_DIR / "requirements-colab.txt"
DEFAULT_HOST = "127.0.0.1"
COLAB_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
SUPPORTED_DEVICE_PREFERENCES = ("auto", "cuda", "cpu")
RUNTIME_IMPORTS = ("fastapi", "numpy", "PIL", "torch", "uvicorn")


def sha256_file(path: Path) -> str:
    digest = sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def is_running_in_colab() -> bool:
    try:
        import google.colab  # type: ignore[import-not-found]
    except ImportError:
        return False
    return True


def resolve_requirements_file(requirements_arg: str | None, prefer_colab: bool | None = None) -> Path:
    if requirements_arg:
        candidate = Path(requirements_arg).expanduser()
        if not candidate.is_absolute():
            candidate = (ROOT_DIR / candidate).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Requirements file was not found: {candidate}")
        return candidate

    if prefer_colab is None:
        prefer_colab = is_running_in_colab()

    if prefer_colab and COLAB_REQUIREMENTS_FILE.exists():
        return COLAB_REQUIREMENTS_FILE
    return REQUIREMENTS_FILE


def get_stamp_file(use_virtualenv: bool, requirements_file: Path) -> Path:
    if use_virtualenv:
        return VENV_DIR / f".{requirements_file.stem}.sha256"
    return ROOT_DIR / f".{requirements_file.stem}.runtime.sha256"


def runtime_dependencies_available() -> bool:
    for module_name in RUNTIME_IMPORTS:
        try:
            __import__(module_name)
        except ImportError:
            return False
    return True


def should_skip_install(stamp_file: Path, expected_hash: str, use_virtualenv: bool) -> bool:
    if not stamp_file.exists():
        return False
    if stamp_file.read_text(encoding="utf-8") != expected_hash:
        return False
    if use_virtualenv:
        return True
    return runtime_dependencies_available()


def build_app_url(host: str, port: int) -> str:
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    return f"http://{display_host}:{port}"


def is_server_running(host: str, port: int) -> bool:
    health_url = f"{build_app_url(host, port)}/api/health"
    try:
        with urlopen(health_url, timeout=1.0) as response:
            return response.status == 200
    except URLError:
        return False


def ensure_virtualenv(use_virtualenv: bool) -> None:
    if not use_virtualenv:
        return
    if VENV_PYTHON.exists():
        return

    subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)], cwd=ROOT_DIR)


def ensure_requirements(python_executable: Path, requirements_file: Path, use_virtualenv: bool) -> None:
    expected_hash = sha256_file(requirements_file)
    stamp_file = get_stamp_file(use_virtualenv, requirements_file)
    if should_skip_install(stamp_file, expected_hash, use_virtualenv):
        return

    subprocess.check_call(
        [str(python_executable), "-m", "pip", "install", "-r", str(requirements_file)],
        cwd=ROOT_DIR,
    )
    stamp_file.parent.mkdir(parents=True, exist_ok=True)
    stamp_file.write_text(expected_hash, encoding="utf-8")


def start_server(
    python_executable: Path,
    host: str,
    port: int,
    reload_enabled: bool,
    device_preference: str | None,
    foreground: bool,
) -> int:
    command = [
        str(python_executable),
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

    environment = os.environ.copy()
    if device_preference:
        environment["UPSCALE_DEVICE"] = device_preference

    if foreground:
        return subprocess.call(command, cwd=ROOT_DIR, env=environment)

    creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" else 0
    subprocess.Popen(
        command,
        cwd=ROOT_DIR,
        env=environment,
        creationflags=creation_flags,
    )
    return 0


def wait_for_server(host: str, port: int, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if is_server_running(host, port):
            return
        time.sleep(0.5)

    raise RuntimeError("The UpScale server did not start within 30 seconds.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch UpScale in browser or terminal mode.")
    parser.add_argument("--requirements", help="Requirements file to install before launching")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI web app")
    serve_parser.add_argument("--host", help="Bind host for the web server")
    serve_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port")
    serve_parser.add_argument("--reload", action="store_true", help="Enable Uvicorn reload mode")
    serve_parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run the server in the current terminal",
    )
    serve_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser after startup",
    )
    serve_parser.add_argument(
        "--device",
        choices=SUPPORTED_DEVICE_PREFERENCES,
        help="Compute device preference",
    )

    cli_parser = subparsers.add_parser("cli", help="Run one terminal upscale job")
    cli_parser.add_argument("--input", "-i", help="Input image path")
    cli_parser.add_argument("--model", "-m", help="Model key or 1-based model index")
    cli_parser.add_argument("--factor", "-f", type=float, help="Upscale factor between 1 and 8")
    cli_parser.add_argument("--denoise", type=float, help="Diffusion denoise strength between 0 and 1")
    cli_parser.add_argument("--output", "-o", help="Output image path")
    cli_parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="Tile size override, or 0 to auto-select",
    )
    cli_parser.add_argument(
        "--device",
        choices=SUPPORTED_DEVICE_PREFERENCES,
        help="Compute device preference",
    )
    cli_parser.add_argument(
        "--list-models",
        action="store_true",
        help="List discovered models and exit",
    )

    parser.set_defaults(command="serve")
    return parser


def ensure_runtime(requirements_arg: str | None, prefer_colab: bool) -> Path:
    use_virtualenv = not prefer_colab
    ensure_virtualenv(use_virtualenv)
    python_executable = VENV_PYTHON if use_virtualenv else Path(sys.executable)
    requirements_file = resolve_requirements_file(requirements_arg, prefer_colab)
    ensure_requirements(python_executable, requirements_file, use_virtualenv)
    return python_executable


def run_cli_command(args: argparse.Namespace, python_executable: Path) -> int:
    command = [str(python_executable), "-m", "app.cli"]
    if args.input:
        command.extend(["--input", args.input])
    if args.model:
        command.extend(["--model", args.model])
    if args.factor is not None:
        command.extend(["--factor", f"{args.factor:g}"])
    if args.denoise is not None:
        command.extend(["--denoise", f"{args.denoise:g}"])
    if args.output:
        command.extend(["--output", args.output])
    if args.tile_size:
        command.extend(["--tile-size", str(args.tile_size)])
    if args.device:
        command.extend(["--device", args.device])
    if args.list_models:
        command.append("--list-models")
    return subprocess.call(command, cwd=ROOT_DIR)


def run_server_command(args: argparse.Namespace, python_executable: Path, prefer_colab: bool) -> int:
    host = args.host or (COLAB_HOST if prefer_colab else DEFAULT_HOST)
    port = args.port
    foreground = args.foreground or prefer_colab
    app_url = build_app_url(host, port)

    if foreground:
        print(f"Starting UpScale at {host}:{port}")
        if prefer_colab:
            print("The server is bound to 0.0.0.0. Use a Colab tunnel or notebook proxy to reach it.")
        return start_server(
            python_executable=python_executable,
            host=host,
            port=port,
            reload_enabled=args.reload,
            device_preference=args.device,
            foreground=True,
        )

    if not is_server_running(host, port):
        start_server(
            python_executable=python_executable,
            host=host,
            port=port,
            reload_enabled=args.reload,
            device_preference=args.device,
            foreground=False,
        )
        wait_for_server(host, port)

    if not args.no_browser:
        webbrowser.open_new_tab(app_url)
    print(f"UpScale is running at {app_url}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    prefer_colab = is_running_in_colab()
    python_executable = ensure_runtime(args.requirements, prefer_colab)

    if args.command == "cli":
        return run_cli_command(args, python_executable)

    return run_server_command(args, python_executable, prefer_colab)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - launcher failure path
        print(f"Unable to launch UpScale: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc