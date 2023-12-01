import math
from os import PathLike
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


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
