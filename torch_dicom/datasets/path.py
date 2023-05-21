#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Iterable, Iterator, cast

import pandas as pd
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
    def __init__(self, paths: Iterable[Path]):
        # NOTE: We store the paths as a pandas Series to reduce memory usage
        # in multiprocessing.
        self.files = pd.Series([Path(p) for p in tqdm(paths, desc="Scanning files", leave=False)])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(len={len(self)})"

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Path:
        return cast(Path, self.files[index])

    def __iter__(self) -> Iterator[Path]:
        for path in self.files:
            yield path
