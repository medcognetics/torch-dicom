import math
from abc import ABC, abstractmethod, abstractproperty
from os import PathLike
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Union

import pandas as pd
import torch
from torch.utils.data import BatchSampler, Sampler, WeightedRandomSampler


def get_sopuid(paths: Iterable[PathLike]) -> Iterator[str]:
    for path in paths:
        yield Path(path).stem


class WeightedCSVSampler(WeightedRandomSampler):
    r"""Sampler that uses information in a CSV file to sample from a dataset with weights.
    Column matching will be done using string comparision.

    Args:
        metadata: Path to a CSV file containing metadata about the dataset.
        example_paths: Paths to the examples in the dataset.
        colname: Name of the column in the CSV file containing the class names.
        weights: Mapping of classes to weights. Weight values should sum to 1.
        num_samples: Number of samples to draw. If not specified, the entire dataset will be used.
        generator: Generator used to sample from the dataset.
    """

    def __init__(
        self,
        metadata: Path,
        example_paths: Sequence[Path],
        colname: str,
        weights: Dict[str, float],
        num_samples: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ):
        if not math.isclose(sum(weights.values()), 1):
            raise ValueError(f"Weights must sum to 1: {weights}")

        # Read the metadata CSV and create a weight column.
        # If a weight value cannot be found for a class, use 0.
        df = pd.read_csv(metadata, index_col="SOPInstanceUID")
        df.index = df.index.astype(str)
        df[colname] = df[colname].astype(str)
        df["weight"] = df[colname].apply(lambda v: weights.get(str(v), 0))

        # Read the SOPInstanceUIDs from the example paths.
        # If an example is not in the metadata, add an entry with a weight of 0.
        sop_uids = list(get_sopuid(example_paths))
        missing_sop_uids = set(sop_uids) - set(df.index.map(str))
        for uid in missing_sop_uids:
            df.loc[uid] = 0

        # Sort the dataframe by SOPInstanceUID
        df = df.loc[sop_uids]

        # Normalize weights within each group. Only include non-zero weights in the total.
        counts_per_group = df[df["weight"] > 0].groupby(colname).count().clip(lower=1)
        for k in weights.keys():
            if k not in counts_per_group.index:
                counts_per_group.loc[k] = 1
        df["weight"] = df.apply(lambda row: row["weight"] / counts_per_group.loc[row[colname]], axis=1)

        super().__init__(
            [float(x) for x in df["weight"]],
            num_samples or len(example_paths),
            replacement=True,
            generator=generator,
        )


class BatchComplementSampler(BatchSampler, ABC):
    r"""Base class for a batch sampler that builds batches by adding complements to examples.
    The metadata CSV file should be indexed by SOPInstanceUID.

    Args:
        sampler: Sampler used to sample examples.
        batch_size: Size of each batch. Should be a multiple of complement_size.
        metadata: Path to a CSV file containing metadata about the dataset.
        example_paths: Paths to the examples in the dataset.
    """

    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,
        metadata: Path,
        example_paths: Sequence[Path],
    ):
        super().__init__(sampler, batch_size, drop_last=True)
        if batch_size % self.complement_size != 0:
            raise ValueError(f"Batch size must be a multiple of complement size {self.complement_size}")

        # Read the metadata CSV
        self.metadata = pd.read_csv(metadata, index_col="SOPInstanceUID")
        self.metadata.index = self.metadata.index.astype(str)

        # Read the SOPInstanceUIDs from the example paths.
        # We will assume that all examples are in the metadata.
        sop_uids = list(get_sopuid(example_paths))

        # Sort the dataframe by SOPInstanceUID
        self.metadata = self.metadata.loc[sop_uids]

    @abstractproperty
    def complement_size(self) -> int:
        r"""The number of complements per example."""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def find_complement(self, idx: int) -> List[int]:
        r"""Given an index, find the index or indices of any complements.
        The returned complement should include the example itself.
        and should have size equal to ``self.complement_size``.
        It is recommended to pad non-compliant complements with ``idx``.
        """
        raise NotImplementedError  # pragma: no cover

    def __iter__(self) -> Iterator[List[int]]:
        sampler_iter = iter(self.sampler)

        batch: List[int] = []
        for idx in sampler_iter:
            # Find the complement indexes for the example and validate
            complement = self.find_complement(idx)
            if len(complement) != self.complement_size:
                raise ValueError(f"Expected {self.complement_size} complements, got {len(complement)}")
            if not all(isinstance(i, int) for i in complement):
                raise TypeError(f"Expected complement indexes to be integers, got {complement}")

            # Update the batch and yield if full
            batch += complement
            if len(batch) >= self.batch_size:
                assert len(batch) == self.batch_size
                yield batch
                batch = []

    def __len__(self) -> int:
        return len(self.sampler) * self.complement_size // self.batch_size  # type: ignore[arg-type]
