"""
DatOps Agent SDK — Background heartbeat worker

Sends periodic heartbeat to DAT reputation service to prevent
the activity watchdog from penalizing the agent for silence.
"""

import logging
import threading
import time
from typing import Optional, Callable

logger = logging.getLogger("datops_agent")


class HeartbeatWorker:
    """Background daemon thread that sends heartbeats."""

    def __init__(
        self,
        heartbeat_fn: Callable[[], None],
        interval: int = 300,
    ):
        self._heartbeat_fn = heartbeat_fn
        self._interval = interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the heartbeat worker thread."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="datops-heartbeat",
            daemon=True,
        )
        self._thread.start()
        logger.debug(f"Heartbeat worker started (interval={self._interval}s)")

    def stop(self) -> None:
        """Stop the heartbeat worker thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.debug("Heartbeat worker stopped")

    def _run(self) -> None:
        """Worker loop — sends heartbeat every interval seconds."""
        # Initial delay before first heartbeat
        if self._stop_event.wait(timeout=30):
            return

        while not self._stop_event.is_set():
            try:
                self._heartbeat_fn()
            except Exception as e:
                logger.debug(f"Heartbeat failed (best effort): {e}")

            # Wait for next interval or stop signal
            if self._stop_event.wait(timeout=self._interval):
                break

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
