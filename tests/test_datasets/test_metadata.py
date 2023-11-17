from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
from torch.utils.data import Dataset

from torch_dicom.datasets import DicomPathDataset
from torch_dicom.datasets.metadata import MetadataDatasetWrapper, MetadataInputWrapper, PreprocessingConfigMetadata
from torch_dicom.preprocessing.pipeline import PreprocessingPipeline


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

    def test_repr(self, dataset: Dataset, metadata: Path):
        wrapper = DummyMetadataInputWrapper(dataset, metadata)
        assert isinstance(repr(wrapper), str)

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

    def test_repr(self, dataset: Dataset, metadata: Path):
        wrapper = DummyMetadataDatasetWrapper(dataset, metadata)
        assert isinstance(repr(wrapper), str)

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

    def test_iter(self, dataset: Dataset, metadata: Path):
        wrapper = DummyMetadataDatasetWrapper(dataset, metadata)
        for item in wrapper:
            assert isinstance(item, dict)
            assert "dummy" in item


class TestPreprocessingConfigMetadata:
    @pytest.fixture(scope="class")
    def preprocessed_data(self, tmp_path_factory, dicoms):
        dest = tmp_path_factory.mktemp("data")
        pipeline = PreprocessingPipeline(dicoms=dicoms)
        out_files = pipeline(dest)
        assert out_files
        return dest

    @pytest.fixture(scope="class")
    def dataset(self, preprocessed_data) -> Dataset:
        paths = preprocessed_data.rglob("*.dcm")
        dataset = DicomPathDataset(paths)
        assert len(dataset), "Failed to create dataset"
        return dataset

    def test_init(self, dataset: Dataset):
        wrapper = PreprocessingConfigMetadata(dataset)
        assert wrapper.dataset == dataset

    def test_repr(self, dataset: Dataset):
        wrapper = PreprocessingConfigMetadata(dataset)
        assert isinstance(repr(wrapper), str)

    @pytest.fixture
    def metadata(self, tmp_path: Path) -> Path:
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text('{"preprocessing": "test"}')
        return metadata_file

    def test_load_metadata(self, dataset: Dataset, metadata: Path):
        wrapper = PreprocessingConfigMetadata(dataset)
        loaded_metadata = wrapper.load_metadata(metadata)
        assert isinstance(loaded_metadata, dict)
        assert "preprocessing" in loaded_metadata
        assert loaded_metadata["preprocessing"] == "test"

    def test_get_metadata(self, dataset: Dataset, metadata: Path):
        wrapper = PreprocessingConfigMetadata(dataset)
        example = wrapper[0]
        metadata_dict = wrapper.get_metadata(example)
        assert isinstance(metadata_dict, dict)
        assert "preprocessing" in metadata_dict
        assert metadata_dict["preprocessing"]
