import math
from pathlib import Path
from typing import List

import pandas as pd
import pytest
import torch

from torch_dicom.datasets.sampler import BatchComplementSampler, WeightedCSVSampler


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
                "dummy": [i % 2 for i in range(len(paths))],
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


class TestBatchComplementSampler:
    @pytest.fixture
    def paths(self):
        return [Path(f"sop-{i}").with_suffix(".dcm") for i in range(9)]

    @pytest.fixture
    def batch_size(self):
        return 4

    @pytest.fixture
    def complement_size(self):
        return 2

    @pytest.fixture
    def metadata(self, tmp_path, paths):
        path = tmp_path / "metadata.csv"
        df = pd.DataFrame(
            {
                "SOPInstanceUID": [p.stem for p in paths],
                "StudyInstanceUID": [f"study-{i // 4}" for i in range(len(paths))],
            }
        )
        df.to_csv(path, index=False)
        return path

    @pytest.fixture
    def sampler(self, metadata, paths, batch_size, complement_size):
        class BatchStudySampler(BatchComplementSampler):
            @property
            def complement_size(self):
                return complement_size

            def find_complement(self, idx: int) -> List[int]:
                return self.select_random_complement(idx, "StudyInstanceUID")

        return BatchStudySampler(range(len(paths)), batch_size, metadata, paths)

    def test_len(self, sampler, paths, complement_size, batch_size):
        assert len(sampler) == len(paths) * complement_size // batch_size

    def test_sampler(self, sampler, batch_size):
        output = list(sampler)
        assert isinstance(output, list)
        assert len(output) == len(sampler)
        for i in output:
            assert isinstance(i, list)
            assert len(i) == batch_size

    @pytest.mark.parametrize(
        "idx, match_column, exp",
        [
            (0, "StudyInstanceUID", [1, 2, 3]),
            (1, "StudyInstanceUID", [0, 2, 3]),
            (4, "StudyInstanceUID", [5, 6, 7]),
        ],
    )
    def test_get_all_complementary_indices(self, sampler, idx: int, match_column: str, exp: List[int]):
        complementary_indices = sampler.get_all_complementary_indices(idx, match_column)
        assert isinstance(complementary_indices, list)
        assert all(isinstance(i, int) for i in complementary_indices)
        assert complementary_indices == exp

    @pytest.mark.parametrize(
        "idx, match_column, exp",
        [
            (0, "StudyInstanceUID", [1, 2, 3]),
            (1, "StudyInstanceUID", [0, 2, 3]),
            (4, "StudyInstanceUID", [5, 6, 7]),
            pytest.param(8, "StudyInstanceUID", [8], id="no-complement"),
        ],
    )
    def test_select_random_complement(self, sampler, idx: int, match_column: str, exp: List[int]):
        complement = sampler.select_random_complement(idx, match_column)
        assert isinstance(complement, list)
        assert all(isinstance(i, int) for i in complement)
        assert len(complement) == sampler.complement_size
        assert set(complement).issubset(set(exp))
