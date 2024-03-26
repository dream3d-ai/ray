import io
import logging
import tarfile
import time
import uuid
from typing import Optional, Union

import pyarrow

from ray.data.block import BlockAccessor
from ray.data.datasource.file_datasink import BlockBasedFileDatasink
from ray.data.datasource.webdataset_datasource import (
    _apply_list,
    _default_encoder,
    _make_iterable,
)

logger = logging.getLogger(__name__)


class _WebDatasetDatasink(BlockBasedFileDatasink):
    def __init__(
        self,
        path: str,
        encoder: Optional[Union[bool, str, callable, list]] = True,
        *,
        file_format: str = "tar",
        **file_datasink_kwargs,
    ):
        super().__init__(
            path,
            file_format=file_format,
            **file_datasink_kwargs,
        )

        self.encoder = encoder

    def write_block_to_file(self, block: BlockAccessor, file: "pyarrow.NativeFile"):
        stream = tarfile.open(fileobj=file, mode="w|")
        samples = _make_iterable(block)

        completed_keys = []
        for sample in samples:
            if not isinstance(sample, dict):
                sample = sample.as_pydict()
            if self.encoder is not None:
                sample = _apply_list(self.encoder, sample, default=_default_encoder)
            if "__key__" not in sample:
                sample["__key__"] = uuid.uuid4().hex
            key = sample["__key__"]

            # add to progress tracker, remove path is not needed
            completed_keys.append(key)

            for k, v in sample.items():
                if v is None or k.startswith("__"):
                    continue
                assert isinstance(v, bytes) or isinstance(v, str)
                if not isinstance(v, bytes):
                    v = v.encode("utf-8")
                ti = tarfile.TarInfo(f"{key}.{k}")
                ti.size = len(v)
                ti.mtime = time.time()
                ti.mode, ti.uname, ti.gname = 0o644, "data", "data"
                stream.addfile(ti, io.BytesIO(v))
        stream.close()
