import sys
import threading
from contextlib import contextmanager

from api.models import StepLog, PipelineRun


class RunLogger:
    """
    Captura stdout y lo guarda en StepLog en tiempo real.
    """
    def __init__(self, run: PipelineRun, step: str):
        self.run = run
        self.step = step
        self._buffer = ""
        self._lock = threading.Lock()

    def write(self, message: str):
        if not message.strip():
            return

        with self._lock:
            self._buffer += message
            if "\n" in self._buffer:
                lines = self._buffer.splitlines(keepends=False)
                self._buffer = ""
                for line in lines:
                    StepLog.objects.create(
                        run=self.run,
                        step=self.step,
                        message=line,
                    )

    def flush(self):
        pass


@contextmanager
def capture_stdout(run: PipelineRun, step: str):
    """
    Context manager para capturar print() en tiempo real.
    """
    old_stdout = sys.stdout
    logger = RunLogger(run, step)
    sys.stdout = logger
    try:
        yield
    finally:
        sys.stdout = old_stdout
