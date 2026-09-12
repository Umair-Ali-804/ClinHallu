"""Cooperative shutdown so an interrupted run stops at a resumable boundary.

Colab disconnects, `Ctrl-C`, and most laptop-sleep/shutdown sequences deliver
``SIGINT``/``SIGTERM``/``SIGHUP`` before the process is killed. Without a
handler Python raises immediately wherever execution happens to be, which for a
training loop is almost always in the middle of a batch -- so the epoch is lost.

``InterruptGuard`` turns those signals into a flag. The training loop polls the
flag at a safe point (an optimizer-step boundary), writes a checkpoint, and
exits with :data:`INTERRUPTED_EXIT_CODE`. A second signal is left to the default
handler so an unresponsive process can always be forced down.

Nothing here can protect against ``SIGKILL`` or a sudden power loss. Those cases
are covered by frequent atomic checkpointing instead.
"""

from __future__ import annotations

import logging
import signal
import threading

logger = logging.getLogger(__name__)

#: Exit code used when a run stopped cleanly at a resumable boundary. Chosen to
#: match the shell convention for "terminated by SIGINT" (128 + 2).
INTERRUPTED_EXIT_CODE = 130

_HANDLED_SIGNALS = ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK")


class InterruptGuard:
    """Record shutdown signals instead of raising at an arbitrary instruction.

    Usage::

        with InterruptGuard() as guard:
            for batch in loader:
                ...
                if guard.triggered:
                    save_checkpoint(...)
                    break
    """

    def __init__(self, *, install: bool = True) -> None:
        self._event = threading.Event()
        self._signal_name: str | None = None
        self._previous: dict[int, object] = {}
        self._install = install

    @property
    def triggered(self) -> bool:
        return self._event.is_set()

    @property
    def signal_name(self) -> str | None:
        return self._signal_name

    def trigger(self, name: str = "manual") -> None:
        """Request shutdown programmatically (used by tests and time budgets)."""
        self._signal_name = name
        self._event.set()

    def _handle(self, signal_number: int, _frame) -> None:
        name = signal.Signals(signal_number).name
        if self._event.is_set():
            # Second signal: restore the default behaviour and re-raise so the
            # user can always force an exit.
            logger.warning("Second %s received -- exiting immediately.", name)
            signal.signal(signal_number, signal.SIG_DFL)
            signal.raise_signal(signal_number)
            return
        self._signal_name = name
        self._event.set()
        logger.warning(
            "%s received. Finishing the current step, saving a checkpoint, then "
            "stopping. Re-run the same command to continue. Send the signal again "
            "to stop immediately (the current partial step would be lost).",
            name,
        )

    def __enter__(self) -> "InterruptGuard":
        if not self._install:
            return self
        for name in _HANDLED_SIGNALS:
            number = getattr(signal, name, None)
            if number is None:
                continue  # SIGHUP/SIGBREAK are platform specific.
            try:
                self._previous[number] = signal.getsignal(number)
                signal.signal(number, self._handle)
            except (ValueError, OSError):
                # Raised when not on the main thread; polling simply never fires.
                logger.debug("Could not install a handler for %s", name)
        return self

    def __exit__(self, *_exc) -> None:
        for number, handler in self._previous.items():
            try:
                signal.signal(number, handler)  # type: ignore[arg-type]
            except (ValueError, OSError, TypeError):
                pass
        self._previous.clear()


class InterruptedRun(RuntimeError):
    """Raised when a stage stopped cleanly and can be resumed."""
