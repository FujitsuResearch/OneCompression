"""Port allocation and HTTP readiness polling.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import socket
import time
from logging import getLogger
from typing import Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = getLogger(__name__)


def find_free_port(host: str = "127.0.0.1") -> int:
    """Return a free TCP port on host.

    Uses the kernel-allocated port from a bound-then-closed socket. There
    is an unavoidable race between this call and the consumer binding the
    port, but in practice it is fine for single-tenant launches.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def wait_for_http(
    url: str,
    timeout_sec: float = 300.0,
    interval_sec: float = 2.0,
    expected_status: int = 200,
    *,
    is_alive: Callable[[], bool] | None = None,
) -> None:
    """Block until url returns expected_status or timeout_sec elapses.

    Args:
        url: HTTP endpoint to probe.
        timeout_sec: Hard wall-clock deadline.
        interval_sec: Sleep between attempts.
        expected_status: Status code that counts as ready.
        is_alive: Optional callback returning False when the server
            process has died; raises immediately if so.

    Raises:
        TimeoutError: deadline reached without a successful probe.
        RuntimeError: is_alive reported the server died.
    """
    deadline = time.monotonic() + timeout_sec
    attempts = 0
    while True:
        if is_alive is not None and not is_alive():
            raise RuntimeError(f"Server process died before {url} became ready")

        attempts += 1
        try:
            req = Request(url, method="GET")
            with urlopen(req, timeout=5) as resp:
                if resp.status == expected_status:
                    logger.info("Endpoint ready after %d attempts: %s", attempts, url)
                    return
        except (URLError, ConnectionError, TimeoutError, OSError) as e:
            logger.debug("Probe %d failed for %s: %s", attempts, url, e)

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Endpoint {url} did not become ready within {timeout_sec}s "
                f"({attempts} probes)"
            )
        time.sleep(interval_sec)
