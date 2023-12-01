import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from torch_dicom.datasets.sampler import WeightedCSVSampler


class TestWeightedCSVSampler:
    @pytest.fixture
    def paths(self):
        return [Path(f).with_suffix(".dcm") for f in ("1.2.3", "4.5.6", "7.8.9")]

    @pytest.fixture
    def metadata(self, tmp_path, paths):
        path = tmp_path / "metadata.csv"
        df = pd.DataFrame(
            {
                "SOPInstanceUID": [p.stem for p in paths],
                "value": [i % 2 for i in range(len(paths))],
            }
        )
        df.to_csv(path, index=False)
        return path

    def test_len(self, metadata, paths):
        sampler = WeightedCSVSampler(metadata, paths, "value", {"0": 0.2, "1": 0.8})
        assert len(sampler) == len(paths)

    @pytest.mark.parametrize(
        "weight,exp",
        [
            ({"0": 0.0, "1": 1.0}, [1, 1, 1]),
            ({"0": 1.0, "1": 0.0}, [0, 0, 0]),
            ({"0": 0.5, "1": 0.5}, [0, 0, 0]),
        ],
    )
    def test_iter(self, metadata, paths, weight, exp):
        torch.random.manual_seed(42)
        sampler = WeightedCSVSampler(metadata, paths, "value", weight, num_samples=3)
        result = list(sampler)
        assert result == exp

    def test_iter_average(self, metadata, paths):
        torch.random.manual_seed(42)
        weight = {"0": 0.25, "1": 0.75}
        sampler = WeightedCSVSampler(metadata, paths, "value", weight, num_samples=1000)
        result = [idx % 2 for idx in sampler]
        assert math.isclose(sum(result) / len(result), 0.75, abs_tol=0.05)
