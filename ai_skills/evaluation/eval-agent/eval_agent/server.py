"""vLLM server process manager.

Accepts a complete command string (typically `chg run --gpus N -- vllm serve ...`),
starts it as a subprocess, polls the health endpoint, captures logs, and provides
clean teardown.
"""

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import psutil
import requests

from eval_agent.audit import RunAudit


class ServerError(Exception):
    """Raised when the server fails to start or respond."""


class VLLMServer:
    """Manages a vLLM server process lifecycle."""

    HEALTH_CHECK_TIMEOUT = 5
    SIGTERM_WAIT = 15
    SIGKILL_WAIT = 5

    def __init__(
        self,
        server_cmd: str,
        audit: RunAudit,
        port: int = 8000,
        health_timeout: int = 600,
        health_interval: int = 5,
    ):
        self.server_cmd = server_cmd
        self.audit = audit
        self.port = port
        self.health_timeout = health_timeout
        self.health_interval = health_interval
        self._process: Optional[subprocess.Popen] = None
        self._log_file = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process else None

    def start(self) -> None:
        """Start the server process and wait for it to become healthy."""
        log_path = self.audit.server_log_path()
        self._log_file = open(log_path, "w")

        self.audit.log_command(self.server_cmd, description="start vllm server")
        self.audit.log_event("server_started", cmd=self.server_cmd, port=self.port)

        self._process = subprocess.Popen(
            self.server_cmd,
            shell=True,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            preexec_fn=os.setsid,
        )

        if not self._wait_healthy():
            self.stop()
            raise ServerError(
                f"Server did not become healthy within {self.health_timeout}s. "
                f"Check {log_path}"
            )

        self.audit.log_event("server_ready", pid=self.pid, port=self.port)

    def _wait_healthy(self) -> bool:
        """Poll /v1/models until 200 with data, or timeout."""
        url = f"{self.base_url}/v1/models"
        deadline = time.monotonic() + self.health_timeout

        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                return False
            try:
                resp = requests.get(url, timeout=self.HEALTH_CHECK_TIMEOUT)
                if resp.status_code == 200 and resp.json().get("data"):
                    return True
            except (requests.RequestException, ValueError):
                pass
            time.sleep(self.health_interval)
        return False

    def stop(self) -> None:
        """Kill the server process tree and close the log file."""
        if self._process and self._process.poll() is None:
            try:
                pgid = os.getpgid(self._process.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    self._process.wait(timeout=self.SIGTERM_WAIT)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    self._process.wait(timeout=self.SIGKILL_WAIT)
            except (ProcessLookupError, OSError):
                pass

        if self._log_file and not self._log_file.closed:
            self._log_file.close()

        self.audit.log_event("server_stopped", pid=self.pid)

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def __del__(self):
        if self._log_file and not self._log_file.closed:
            self._log_file.close()
