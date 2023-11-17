from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
from torch.utils.data import Dataset

from torch_dicom.datasets.metadata import MetadataDatasetWrapper, MetadataInputWrapper


class DummyMetadataInputWrapper(MetadataInputWrapper):
    def load_metadata(self, metadata: Path) -> Any:
        return pd.read_csv(metadata)

    def get_metadata(self, example: Dict[str, Any]) -> Dict[str, Any]:
        return {"dummy": "metadata"}


class DummyMetadataDatasetWrapper(MetadataDatasetWrapper):
    def load_metadata(self, metadata: Path) -> Any:
        return pd.read_csv(metadata)

    def get_metadata(self, example: Dict[str, Any]) -> Dict[str, Any]:
        return {"dummy": "metadata"}


class DummyDataset(Dataset):
    def __init__(self):
        self.data = [{"dummy": "data"} for _ in range(10)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return DummyDataset()


class TestMetadataInputWrapper:
    @pytest.fixture
    def metadata(self, tmp_path: Path) -> Path:
        metadata_file = tmp_path / "metadata.csv"
        metadata_file.write_text("dummy,metadata\n1,test")
        return metadata_file

    def test_init(self, dataset: Dataset, metadata: Path):
        wrapper = DummyMetadataInputWrapper(dataset, metadata)
        assert wrapper.dataset == dataset
        assert isinstance(wrapper.metadata, pd.DataFrame)

    def test_load_metadata(self, dataset: Dataset, metadata: Path):
        wrapper = DummyMetadataInputWrapper(dataset, metadata)
        assert isinstance(wrapper.load_metadata(metadata), pd.DataFrame)

    def test_get_metadata(self, dataset: Dataset, metadata: Path):
        wrapper = DummyMetadataInputWrapper(dataset, metadata)
        assert wrapper.get_metadata({}) == {"dummy": "metadata"}

    def test_iter(self, dataset: Dataset, metadata: Path):
        wrapper = DummyMetadataInputWrapper(dataset, metadata)
        for item in wrapper:
            assert isinstance(item, dict)
            assert "dummy" in item


class TestMetadataDatasetWrapper:
    @pytest.fixture
    def metadata(self, tmp_path: Path) -> Path:
        metadata_file = tmp_path / "metadata.csv"
        metadata_file.write_text("dummy,metadata\n1,test")
        return metadata_file

    def test_init(self, dataset: Dataset, metadata: Path):
        wrapper = DummyMetadataDatasetWrapper(dataset, metadata)
        assert wrapper.dataset == dataset
        assert isinstance(wrapper.metadata, pd.DataFrame)

    def test_load_metadata(self, dataset: Dataset, metadata: Path):
        wrapper = DummyMetadataDatasetWrapper(dataset, metadata)
        assert isinstance(wrapper.load_metadata(metadata), pd.DataFrame)

    def test_get_metadata(self, dataset: Dataset, metadata: Path):
        wrapper = DummyMetadataDatasetWrapper(dataset, metadata)
        assert wrapper.get_metadata({}) == {"dummy": "metadata"}

    def test_len(self, dataset, metadata: Path):
        wrapper = DummyMetadataDatasetWrapper(dataset, metadata)
        assert len(wrapper) == len(dataset)

    def test_getitem(self, dataset, metadata: Path):
        wrapper = DummyMetadataDatasetWrapper(dataset, metadata)
        for i in range(len(dataset)):
            assert wrapper[i] == dataset[i]
