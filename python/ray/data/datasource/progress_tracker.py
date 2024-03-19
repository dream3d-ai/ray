import atexit
import json
import logging
import signal
from dataclasses import dataclass

import pandas as pd

import ray

logger = logging.getLogger(__name__)

CACHED_PROGRESS_TRACKERS = {}


@dataclass
class Progress:
    completed_paths: list[str]
    completed_keys: list[str]
    in_progress_paths: list[str]

    @property
    def skip_files(self) -> set[str]:
        return set(self.completed_paths) + set(self.in_progress_paths)

    def to_json(self) -> bytes:
        return json.dumps(self.__dict__).encode("utf-8")

    @classmethod
    def from_json(cls, json_str: str) -> "Progress":
        return cls(**json.loads(json_str))


@ray.remote
class ProgressTracker:
    def __init__(
        self,
        save_path: str,
        save_interval: int = 1,
        write_paths_: bool = False,
    ):
        self.save_path = save_path
        self.write_paths_ = write_paths_

        self.load_progress()
        self.initial_progress = self.get_current_progress()

        self.counter = 0
        self.save_interval = save_interval

        atexit.register(self.write)
        self.init_signal_handlers()

    def load_progress(self):
        try:
            import fsspec
        except ImportError:
            raise ImportError("Please install fsspec")

        if not fsspec.exists(self.save_path):
            self.progress = Progress(
                completed_paths=[], completed_keys=[], in_progress_paths=[]
            )

        with fsspec.open(self.save_path, "r") as f:
            self.progress = Progress.from_json(f.read())

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
        self.in_progress = pd.concat(
            [self.in_progress, pd.DataFrame(items)], ignore_index=True
        )

    def update_completed(self, items: list[dict[str, str]]):
        assert all("__key__" in item for item in items)
        self.completed = pd.concat(
            [self.completed, pd.DataFrame(items)], ignore_index=True
        )

        self.counter += 1
        if self.counter % self.save_interval == 0:
            self.write()

    def update_path(self, key: str, path: str):
        self.completed.loc[self.completed["__key__"] == key, "path"] = path

    def should_write_paths_(self) -> bool:
        return self.write_paths_

    def get_current_progress(self) -> Progress:
        return Progress(
            completed_paths=self.completed["path"].tolist(),
            completed_keys=self.completed["__key__"].tolist(),
            in_progress_paths=self.in_progress["path"].tolist(),
        )

    def get_initial_progress(self) -> Progress:
        return self.initial_progress

    def write(self) -> bool:
        try:
            import fsspec
        except ImportError:
            raise ImportError("Please install fsspec")

        logger.debug(f"Writing progress tracker to {self.save_path}")
        with fsspec.open(self.save_path, "w") as f:
            f.write(self.get_current_progress().to_json())
        return True
