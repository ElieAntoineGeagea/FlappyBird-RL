"""Cooperative controls for headless training on Windows and other systems."""
import os
import signal
import sys
import threading
from pathlib import Path


class TrainingControl:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.paused = False
        self.stop = False
        self.checkpoint = False
        self._handlers = {}

    def _interrupt(self, signum, frame):
        # A signal handler requests a stop; disk writes happen in the training loop.
        self.stop = True

    def install(self):
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                self._handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._interrupt)

    def restore(self):
        for signum, handler in self._handlers.items():
            signal.signal(signum, handler)
        self._handlers.clear()

    def poll(self):
        if (self.directory / 'STOP').exists():
            self.stop = True
        request = self.directory / 'CHECKPOINT'
        if request.exists():
            self.checkpoint = True
            request.unlink(missing_ok=True)
        if os.name == 'nt' and sys.stdin is not None and sys.stdin.isatty():
            import msvcrt
            while msvcrt.kbhit():
                key = msvcrt.getwch()
                if key in ('\x00', '\xe0'):
                    msvcrt.getwch()  # Discard extended-key scan code (e.g. Down is 'P').
                    continue
                key = key.lower()
                if key == 'p':
                    self.paused = not self.paused
                elif key == 's':
                    self.stop = True
                elif key == 'c':
                    self.checkpoint = True
        return self.paused or (self.directory / 'PAUSE').exists()
