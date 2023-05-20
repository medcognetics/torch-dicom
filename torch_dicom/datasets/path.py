#!/usr/bin/env python
# -*- coding: utf-8 -*-
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, cast

from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm


class PathInput(IterableDataset):
    def __init__(self, paths: Iterable[Path]):
        self.files = paths

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __iter__(self) -> Iterator[Path]:
        for path in self.files:
            yield path


class PathDataset(Dataset):
    def __init__(self, paths: Iterable[Path], manager: Optional[SyncManager] = None):
        files = [Path(p) for p in tqdm(paths, desc="Scanning files", leave=False)]

        # We may store a very large number of paths, so we provide an option to use a shared memory manager
        self.files: List[Path] = cast(List, manager.list(files)) if manager is not None else files

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(len={len(self)})"

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Path:
        return self.files[index]

    def __iter__(self) -> Iterator[Path]:
        for path in self.files:
            yield path
