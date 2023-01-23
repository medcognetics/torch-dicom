#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractclassmethod
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from torch.utils.data import Dataset


class ManifestDataset(Dataset, ABC):
    def __init__(self, path: Path, **kwargs):
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        self.path = path
        self.root = path.parent
        self.manifest = self.read_manifest(path, **kwargs)

    @abstractclassmethod
    def read_manifest(cls, path: Path, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path}, len={len(self)})"

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> Any:
        return self.manifest.iloc[index]

    def __iter__(self) -> Iterator[Path]:
        for i in range(len(self)):
            yield self[i]


class CSVManifestDataset(ManifestDataset):
    @classmethod
    def read_manifest(cls, path: Path, **kwargs) -> pd.DataFrame:
        df = pd.read_csv(path, **kwargs)
        assert isinstance(df, pd.DataFrame)
        return df
