import atexit
import json
import logging
import signal
from collections import defaultdict
from dataclasses import dataclass, field

import ray

logger = logging.getLogger(__name__)

CACHED_PROGRESS_TRACKERS = {}


@dataclass
class Progress:
    completed_paths: set[str] = field(default_factory=set)
    completed_keys: set[str] = field(default_factory=set)
    in_progress: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set), metadata="path -> keys"
    )

    @property
    def skip_files(self) -> set[str]:
        unfinished_paths = set(
            path for (path, keys) in self.in_progress.items() if keys
        )
        return self.completed_paths - unfinished_paths

    def to_json(self) -> str:
        return json.dumps(
            {
                "completed_paths": list(self.completed_paths),
                "completed_keys": list(self.completed_keys),
                "in_progress": {
                    path: list(keys) for path, keys in self.in_progress.items()
                },
            }
        )

    @classmethod
    def load(cls, json_bytes: bytes) -> "Progress":
        raw_data = json.loads(json_bytes)
        return cls(
            completed_paths=set(raw_data["completed_paths"]),
            completed_keys=set(raw_data["completed_keys"]),
            in_progress={
                path: set(keys) for path, keys in raw_data["in_progress"].items()
            },
        )

    def deepcopy(self):
        return Progress(
            completed_paths=self.completed_paths.copy(),
            completed_keys=self.completed_keys.copy(),
            in_progress=self.in_progress.copy(),
        )


@ray.remote
class ProgressTracker:
    def __init__(
        self,
        save_path: str,
        save_interval: int = 1,
        write_paths_: bool = False,
    ):
        try:
            import fsspec
        except ImportError:
            raise ImportError("Please install fsspec")

        self.save_path = save_path
        self.write_paths_ = write_paths_

        if not fsspec.exists(self.save_path):
            logger.info(f"Creating new progress tracker at {self.save_path}")
            self.current_progress = Progress()
        else:
            logger.info(f"Loading progress tracker from {self.save_path}")
            with fsspec.open(self.save_path, "rb", compression="gzip") as f:
                self.current_progress = Progress.load(f.read())

        self.initial_progress = self.current_progress.deepcopy()

        self.counter = 0
        self.save_interval = save_interval

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

    def set_save_interval(self, save_interval: int):
        self.save_interval = save_interval

    def update_in_progress(self, items: list[dict[str, str]]):
        assert all("__key__" in item for item in items) and all(
            "path" in item for item in items
        )
        for item in items:
            self.current_progress.in_progress[item["path"]].add(item["__key__"])

    def update_completed(self, items: list[dict[str, str]]):
        assert all("__key__" in item for item in items) and all(
            "path" in item for item in items
        )

        self.current_progress.completed_paths.update([item["path"] for item in items])
        self.current_progress.completed_keys.update([item["__key__"] for item in items])

        for item in items:
            self.current_progress.in_progress[item["path"]].discard(item["__key__"])

        self.counter += 1
        if self.counter % self.save_interval == 0:
            self.write()

    def should_write_paths_(self) -> bool:
        return self.write_paths_

    def get_current_progress(self) -> Progress:
        return self.current_progress

    def get_initial_progress(self) -> Progress:
        return self.initial_progress

    def write(self) -> bool:
        try:
            import fsspec
        except ImportError:
            raise ImportError("Please install fsspec")

        logger.debug(f"Writing progress tracker to {self.save_path}")
        with fsspec.open(self.save_path, "wb", compression="gzip") as f:
            f.write(self.get_current_progress().to_json().encode("utf-8"))
        return True
