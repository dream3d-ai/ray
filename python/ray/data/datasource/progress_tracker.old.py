import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import ray

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


Key = str
Path = str


@ray.remote
class PendingQueue:
    def __init__(self, pending: dict[Path, set[Key]] = defaultdict(set)):
        if isinstance(pending, dict):
            pending = defaultdict(set, pending)
        self.pending: dict[Path, set[Key]] = pending

    def add(self, key: list[Key], path: Path):
        self.pending[path].update(key)

    def get(self) -> dict[Path, set[Key]]:
        return self.pending


@ray.remote
class CompletedQueue:
    def __init__(self, completed: dict[Path, set[Key]] = defaultdict(set)):
        if isinstance(completed, dict):
            completed = defaultdict(set, completed)
        self.completed = completed
        self.completed_keys = set.union(*completed.values())

    def update(self, key: Key):
        self.completed_keys.add(key)

    def get(self) -> set[Key]:
        return self.completed_keys

    def sync(self, completed: dict[Path, set[Key]]) -> dict[Path, set[Key]]:
        for path, keys in completed.items():
            self.completed[path] |= keys
        return completed


@ray.remote
class ProgressTracker:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.initial_progress = self.load()

        self.pending_queue = PendingQueue.remote(pending=self.initial_progress.pending)
        self.completed_queue = CompletedQueue.remote(
            completed=self.initial_progress.completed
        )
        self.lock = False

    def get_initial_progress(self) -> Progress:
        return self.initial_progress

    def acquire_lock(self) -> bool:
        return not self.lock

    def mark_pending_fn(self) -> Callable[[list[str], str], None] | None:
        return self.pending_queue.add if self.acquire_lock() else None

    def mark_completed_fn(self) -> Callable[[list[str]], None] | None:
        return self.completed_queue.update if self.acquire_lock() else None

    def sync(self):
        self.lock = True

        up_to_date = False
        while not up_to_date:
            pending, completed_keys = ray.get(
                [self.pending_queue.get.remote(), self.completed_queue.get.remote()]
            )
            up_to_date = not (completed_keys - set.union(*pending.values()))

        completed_paths = set()
        for path, keys in pending.items():
            if not keys - completed_keys:
                completed_paths.add(path)

        completed = ray.get(self.completed_queue.sync.remote(completed_paths))

        self.lock = False

        return Progress(
            pending=pending,
            completed=completed,
        )

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

        # add all completed keys to pending
        for path, keys in progress.completed.items():
            progress.pending[path] |= keys

        return progress

    def write(self):
        try:
            import fsspec
        except ImportError:
            raise ImportError("Please install fsspec")

        progress = self.sync()

        # remove any completed keys from pending
        for path, keys in progress.completed.items():
            progress.pending[path] -= keys

        # remove any empty paths from pending
        progress.pending = {
            path: keys for path, keys in progress.pending.items() if keys
        }

        logger.debug(f"Writing progress tracker to {self.save_path}")
        with fsspec.open(self.save_path, "wb", compression="gzip") as f:
            f.write(progress.to_json().encode("utf-8"))

        return True
