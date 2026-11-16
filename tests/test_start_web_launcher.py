"""Exercise the Windows launcher with tiny local packages and a real venv."""

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from zipfile import ZipFile

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher")
ROOT = Path(__file__).resolve().parents[1]
WEB_STUB = """\
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

with Path(os.environ['KOWW_LAUNCH_TEST_LOG']).open('a', encoding='utf-8') as stream:
    stream.write(json.dumps({
        'args': sys.argv[1:],
        'prefix': sys.prefix,
        'version': version('korean-openwakeword'),
        'frontend_version': version('openwakeword'),
        'cache': os.environ['KOWW_CACHE_ROOT'],
        'temp': os.environ['TEMP'],
    }) + '\\n')
"""
CONSOLE_SERVER = WEB_STUB + """\
from http.server import HTTPServer, SimpleHTTPRequestHandler

port = int(sys.argv[sys.argv.index('--port') + 1])
with HTTPServer(('127.0.0.1', port), SimpleHTTPRequestHandler) as server:
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
"""
CONSOLE_ACTION = """\
import ctypes
from ctypes import wintypes
import sys

kernel = ctypes.WinDLL('kernel32', use_last_error=True)
user = ctypes.WinDLL('user32', use_last_error=True)
kernel.GetConsoleWindow.restype = wintypes.HWND
user.PostMessageW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
kernel.FreeConsole()
if not kernel.AttachConsole(int(sys.argv[1])):
    raise ctypes.WinError(ctypes.get_last_error())
try:
    if sys.argv[2] == 'close':
        window = kernel.GetConsoleWindow()
        if not window or not user.PostMessageW(window, 0x0010, 0, 0):  # WM_CLOSE
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        if not kernel.SetConsoleCtrlHandler(None, True) or not kernel.GenerateConsoleCtrlEvent(0, 0):
            raise ctypes.WinError(ctypes.get_last_error())
finally:
    kernel.FreeConsole()
"""


def make_wheel(directory, module, version, main=None):
    dist_info = f"{module}-{version}.dist-info"
    files = {
        f"{module}/__init__.py": "",
        f"{dist_info}/METADATA": f"Metadata-Version: 2.1\nName: {module}\nVersion: {version}\n\n",
        f"{dist_info}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    if main is not None:
        files[f"{module}/__main__.py"] = main
        # The fake app needs no real dependencies, but the launcher's readiness
        # probe should still see the modules a real installation provides.
        for name in ("numpy", "scipy", "soundfile", "onnxruntime", "requests", "tqdm", "sklearn"):
            files[f"{name}/__init__.py"] = ""
    record = f"{dist_info}/RECORD"
    files[record] = "\n".join(f"{name},," for name in (*files, record)) + "\n"
    wheel = directory / f"{module}-{version}-py3-none-any.whl"
    with ZipFile(wheel, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return wheel


def make_source(install, wheels):
    wheel = make_wheel(wheels, "korean_openwakeword", "0.33.2", WEB_STUB)
    # Build an offline source fixture using pip's normal PEP 517 path.
    (install / "pyproject.toml").write_text(
        '[build-system]\nrequires = []\nbuild-backend = "fixture_backend"\nbackend-path = ["."]\n'
        '[project]\nname = "korean-openwakeword"\nversion = "0.33.2"\n',
        encoding="utf-8",
    )
    (install / "fixture_backend.py").write_text(
        "from pathlib import Path\nfrom shutil import copyfile\n"
        "def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):\n"
        f"    wheel = Path({str(wheel)!r})\n"
        "    copyfile(wheel, Path(wheel_directory) / wheel.name)\n"
        "    return wheel.name\n",
        encoding="utf-8",
    )
    package = install / "korean_openwakeword"
    package.mkdir()
    (package / "__init__.py").touch()


@pytest.fixture
def launcher(tmp_path):
    install = tmp_path / "web install"
    scripts = install / "scripts"
    scripts.mkdir(parents=True)
    for name in ("Start-Web.cmd", "Start-Web.ps1", "Use-ProjectStorage.ps1"):
        shutil.copyfile(ROOT / "scripts" / name, scripts / name)
    wheels = tmp_path / "wheel house"
    wheels.mkdir()
    log = tmp_path / "web-calls.jsonl"
    env = dict(PIP_CONFIG_FILE=os.devnull, PIP_NO_INDEX="1",
               PIP_FIND_LINKS=wheels.as_uri(), KOWW_LAUNCH_TEST_LOG=str(log))
    # A CI environment may expose Python through its executable rather than py.exe.
    env["PATH"] = str(Path(sys._base_executable).parent) + os.pathsep + os.environ["PATH"]
    return install, scripts, wheels, log, env


def cmd_invocation(script, args, env):
    cmd = os.environ.get("ComSpec", "cmd.exe")
    prefix = subprocess.list2cmdline([cmd, "/d", "/s", "/c"])
    command = subprocess.list2cmdline([str(script), *args])
    process_env = os.environ.copy()
    process_env.pop("PIP_CONSTRAINT", None)
    process_env.update(env)
    return f'{prefix} "{command}"', cmd, process_env


def run_cmd(script, args, *, cwd, env):
    command, cmd, process_env = cmd_invocation(script, args, env)
    return subprocess.run(command, executable=cmd, cwd=cwd, env=process_env,
                          stdin=subprocess.DEVNULL, capture_output=True, timeout=150)


def assert_launched(log, install, workspace, *, port, count):
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    cache = workspace / "storage" / "runtime_cache"
    assert len(rows) == count
    for row in rows:
        assert row["args"] == ["web", "--workspace", str(workspace), "--port", str(port), "--no-browser"]
        assert Path(row["prefix"]) == install / ".venv"
        assert Path(row["cache"]) == cache
        assert Path(row["temp"]) == cache / "tmp"
        assert row["version"] == "0.33.2"
        assert row["frontend_version"] == "0.6.0"


def test_start_web_cmd_installs_and_reuses_venv_with_workspace_arguments(launcher, tmp_path):
    install, scripts, wheels, log, env = launcher
    workspace = tmp_path / "학습 workspace"
    workspace.mkdir()
    make_source(install, wheels)
    make_wheel(wheels, "openwakeword", "0.6.0")
    cmd = install / "Start-Web.cmd"
    shutil.copyfile(scripts / "Start-Web.cmd", cmd)
    args = ["-Workspace", str(workspace), "-Port", "8893", "-NoBrowser"]
    first = run_cmd(cmd, args, cwd=tmp_path, env=env)
    assert first.returncode == 0, (first.stdout, first.stderr)
    assert_launched(log, install, workspace, port=8893, count=1)

    # Any repeat pip installation would fail while opening this nonexistent file.
    repeat_env = dict(env, PIP_CONSTRAINT=str(tmp_path / "missing-constraints.txt"))
    second = run_cmd(cmd, args, cwd=tmp_path, env=repeat_env)
    assert second.returncode == 0, (second.stdout, second.stderr)
    assert_launched(log, install, workspace, port=8893, count=2)


def test_start_web_retries_incomplete_dependency_install(launcher, tmp_path):
    install, scripts, wheels, log, env = launcher
    make_wheel(install, "korean_openwakeword", "0.33.2", WEB_STUB)
    cmd = install / "Start-Web.cmd"
    shutil.copyfile(scripts / "Start-Web.cmd", cmd)
    first = run_cmd(cmd, ["-NoBrowser"], cwd=tmp_path, env=env)
    assert first.returncode != 0
    assert not log.exists()
    assert (install / ".venv" / "Scripts" / "python.exe").is_file()

    make_wheel(wheels, "openwakeword", "0.6.0")
    second = run_cmd(cmd, ["-NoBrowser"], cwd=tmp_path, env=env)
    assert second.returncode == 0, (second.stdout, second.stderr)
    assert_launched(log, install, install, port=8793, count=1)


def test_start_web_replaces_a_different_same_version_wheel(launcher, tmp_path):
    install, scripts, wheels, log, env = launcher
    make_wheel(wheels, "openwakeword", "0.6.0")
    selected = {}
    for revision in ("first", "second"):
        directory = tmp_path / revision
        directory.mkdir()
        main = WEB_STUB.replace("'args':", f"'artifact': {revision!r}, 'args':")
        selected[revision] = make_wheel(directory, "korean_openwakeword", "0.33.2", main)

    cmd = scripts / "Start-Web.cmd"
    for revision in ("first", "second", "second"):
        result = run_cmd(cmd, ["-NoBrowser", "-WheelPath", str(selected[revision])],
                         cwd=tmp_path, env=env)
        assert result.returncode == 0, (result.stdout, result.stderr)
    assert b"Installing the library and web dependencies" not in result.stdout
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [row["artifact"] for row in rows] == ["first", "second", "second"]
    assert_launched(log, install, install, port=8793, count=3)


def test_start_web_console_exit_stops_listener_and_allows_restart(launcher, tmp_path):
    install, scripts, wheels, calls, env = launcher
    make_wheel(install, "korean_openwakeword", "0.33.2", CONSOLE_SERVER)
    make_wheel(wheels, "openwakeword", "0.6.0")
    workspace = tmp_path / "separate workspace"
    workspace.mkdir()
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]

    args = ["-Workspace", str(workspace), "-Port", str(port), "-NoBrowser"]
    command, cmd, process_env = cmd_invocation(scripts / "Start-Web.cmd", args, env)
    console = subprocess.STARTUPINFO(dwFlags=subprocess.STARTF_USESHOWWINDOW, wShowWindow=subprocess.SW_HIDE)
    for action in ("close", "interrupt"):
        log_path = workspace / f"{action}.log"
        with log_path.open("wb") as log:
            launcher_process = subprocess.Popen(command, executable=cmd, cwd=tmp_path, env=process_env,
                startupinfo=console, creationflags=subprocess.CREATE_NEW_CONSOLE,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 60
                while True:
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=.2):
                            break
                    except OSError:
                        assert launcher_process.poll() is None and time.monotonic() < deadline, log_path.read_bytes()
                        time.sleep(.05)

                result = subprocess.run([sys.executable, "-I", "-c", CONSOLE_ACTION, str(launcher_process.pid), action],
                    capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW, timeout=10)
                assert result.returncode == 0, (result.stdout, result.stderr)
                launcher_process.wait(timeout=10)
                with pytest.raises(ConnectionRefusedError), socket.create_connection(("127.0.0.1", port), timeout=5):
                    pytest.fail("Web server still listens after the Start-Web console exits")
            finally:
                if launcher_process.poll() is None:
                    subprocess.run([sys.executable, "-I", "-c", CONSOLE_ACTION, str(launcher_process.pid), "close"],
                        capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW, timeout=10)
                    try:
                        launcher_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        launcher_process.kill()
                        launcher_process.wait(timeout=5)
    assert_launched(calls, install, workspace, port=port, count=2)
