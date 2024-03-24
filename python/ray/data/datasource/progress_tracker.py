import atexit
import json
import logging
import queue
import time
from collections import defaultdict
from dataclasses import dataclass, field

import ray

logger = logging.getLogger(__name__)


Key = str
Path = str


class RequiresFlush(Exception):
    pass


@dataclass
class Progress:
    pending: dict[Path, set[Key]] = field(
        default_factory=lambda: defaultdict(set), metadata="path -> keys"
    )
    completed: dict[Path, set[Key]] = field(
        default_factory=lambda: defaultdict(set), metadata="path -> keys"
    )

    @property
    def skip_files(self) -> set[Path]:
        return set(self.completed.keys()) - set(self.pending.keys())

    @property
    def skip_keys(self) -> set[Key]:
        return set().union(*self.completed.values())

    def to_json(self) -> str:
        return json.dumps(
            {
                "completed": {
                    path: list(keys) for path, keys in self.completed.items() if keys
                },
                "in_progress": {
                    path: list(keys) for path, keys in self.pending.items() if keys
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


@ray.remote(concurrency_groups={"pending": 1000, "completed": 1000, "write": 1})
class ProgressTracker:
    def __init__(
        self,
        progress_path: str,
        save_interval: int = 10_000,
    ):
        if save_interval < 1:
            raise ValueError("save_interval must be greater than 0")

        if not progress_path.endswith(".progress"):
            raise ValueError("load_path must end with '.progress'")

        self.load_path = progress_path
        self.save_path = None

        self.progress = self.load()
        self.initial_progress_ref = ray.put(self.progress)

        self.pending_queue = queue.Queue()
        self.completed_queue = queue.Queue(
            maxsize=save_interval,
        )

    def get_initial_progress(self) -> ray.ObjectRef:
        return self.initial_progress_ref

    def set_save_path(self, save_path: str):
        if not save_path.endswith(".progress"):
            raise ValueError("load_path must end with '.progress'")

        self.save_path = save_path
        atexit.register(self.write)

    @ray.method(concurrency_group="pending")
    def put_pending(self, items: list[tuple[Path, Key]]) -> None:
        for item in items:
            sleep = 1
            path, key = item
            while True:
                try:
                    self.pending_queue.put_nowait((path, key))
                    break
                except queue.Full:
                    logger.debug(f"Pending queue is full, retrying in {sleep} seconds.")
                    time.sleep(sleep)
                    sleep *= 2

    @ray.method(concurrency_group="completed")
    def put_completed(self, keys: list[Key]) -> None:
        if self.completed_queue.full():
            raise RequiresFlush()
        for key in keys:
            self.completed_queue.put(key)

    def get_pending(self) -> list[tuple[Path, Key]]:
        return [
            self.pending_queue.get_nowait() for _ in range(self.pending_queue.qsize())
        ]

    def get_completed(self) -> list[Key]:
        return [
            self.completed_queue.get_nowait()
            for _ in range(self.completed_queue.qsize())
        ]

    def _flush(self):
        logger.debug("Syncing progress tracker")

        # flush the queues
        completed_keys: list[Key] = self.get_completed()
        pending_path_and_keys: list[tuple[Path, Key]] = self.get_pending()

        pending: dict[Path, set[Key]] = defaultdict(set)
        for path, key in pending_path_and_keys:
            pending[path].add(key)

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

    @ray.method(concurrency_group="write")
    def write(self):
        try:
            import fsspec
        except ImportError:
            raise ImportError("Please install fsspec")

        self._flush()

        logger.debug(f"Writing progress tracker to {self.save_path}")
        with fsspec.open(self.save_path, "wb", compression="gzip") as f:
            f.write(self.progress.to_json().encode("utf-8"))

    @ray.method(concurrency_group="write")
    def write_and_put_completed(self, keys: list[Key]):
        while True:
            try:
                self.write()
                self.put_completed(keys)
                return
            except RequiresFlush:
                pass

    def load(self) -> Progress:
        try:
            import fsspec
        except ImportError:
            raise ImportError("Please install fsspec")

        with fsspec.open(self.load_path, "rb", compression="gzip") as f:
            progress = Progress.load(f.read())
        logger.info(f"Loading progress from {self.load_path}")
        return progress

    def shutdown(self):
        if self.save_path is not None:
            self.write()

    def __del__(self):
        self.shutdown()
