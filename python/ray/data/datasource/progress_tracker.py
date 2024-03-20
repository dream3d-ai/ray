import atexit
import json
import logging
import signal
from collections import defaultdict
from dataclasses import dataclass, field

import ray
from ray.util.queue import Queue

logger = logging.getLogger(__name__)

CACHED_PROGRESS_TRACKERS = {}


@dataclass
class Progress:
    pending: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set), metadata="path -> keys"
    )
    completed: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set), metadata="path -> keys"
    )

    @property
    def skip_files(self) -> set[str]:
        return set(self.completed.keys()) - set(self.pending.keys())

    def to_json(self) -> str:
        return json.dumps(
            {
                "completed": {
                    path: list(keys) for path, keys in self.completed.items() if keys
                },
                "in_progress": {
                    path: list(keys) for path, keys in self.in_progress.items() if keys
                },
            }
        )

    @classmethod
    def load(cls, json_bytes: bytes) -> "Progress":
        raw_data = json.loads(json_bytes)
        return cls(
            pending=defaultdict(
                set, {path: set(keys) for path, keys in raw_data["in_progress"].items()}
            ),
            completed=defaultdict(
                set, {path: set(keys) for path, keys in raw_data["completed"].items()}
            ),
        )

    def deepcopy(self) -> "Progress":
        return Progress(
            pending=defaultdict(
                set, {path: set(keys) for path, keys in self.pending.items()}
            ),
            completed=defaultdict(
                set, {path: set(keys) for path, keys in self.completed.items()}
            ),
        )


Key = str
Path = str


@ray.remote
class ProgressTracker:
    def __init__(self, save_path: str, save_interval: int = 100_000):
        self.save_path = save_path
        self.initial_progress = self.load()

        self.progress = self.initial_progress.deepcopy()
        self.pending_queue = Queue()
        self.completed_queue = Queue(max_size=save_interval)

        self.lock = False

        atexit.register(self.write)
        self.init_signal_handlers()

    def sigkill_handler(self, signum, frame):
        self.write()

        if self.original_handlers.get(signum):
            self.original_handlers[signum](signum, frame)

    def init_signal_handlers(self):
        self.original_handlers = {
            signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        }
        signal.signal(signal.SIGTERM, self.sigkill_handler)

    def get_initial_progress(self) -> Progress:
        return self.initial_progress

    def get_pending_queue(self) -> Queue:
        # Call put_nowait_batch on this queue
        return self.pending_queue

    def get_completed_queue(self) -> Queue:
        # Call put on this queue, and write when it is full
        return self.completed_queue

    def sync(self):
        # get everything from the completed queue
        completed_keys: set[Key] = set()
        while self.completed_queue.qsize() > 0:
            key = self.completed_queue.get()
            completed_keys.add(key)

        # get everything from the pending queue
        pending: dict[Path, set[Key]] = {}
        while self.pending_queue.qsize() > 0:
            path, keys = self.pending_queue.get()
            pending[path] |= keys

        # update pending in self.progress
        for path, keys in pending.items():
            self.progress.pending[path].update(keys)

        # update completed in self.progress, and remove from pending
        for key in completed_keys:
            all_search = list(self.progress.completed.items()) + list(
                self.progress.pending.items()
            )
            for path, keys in all_search:
                if key in keys:
                    self.progress.completed[path].add(key)
                    self.progress.pending[path].remove(key)
                    break

    def write(self):
        try:
            import fsspec
        except ImportError:
            raise ImportError("Please install fsspec")

        self.sync()

        logger.debug(f"Writing progress tracker to {self.save_path}")
        with fsspec.open(self.save_path, "wb", compression="gzip") as f:
            f.write(self.progress.to_json().encode("utf-8"))

        return True

    def load(self):
        try:
            import fsspec
        except ImportError:
            raise ImportError("Please install fsspec")

        try:
            with fsspec.open(self.save_path, "rb", compression="gzip") as f:
                progress = Progress.load(f.read())
            logger.info(f"Loading progress from {self.save_path}")
        except FileNotFoundError:
            logger.info(f"Creating new progress tracker at {self.save_path}")
            progress = Progress()

        return progress
