import hashlib
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import Dataset
from torchvision.transforms.v2 import RandomHorizontalFlip
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from torch_dicom.datasets import ImagePathDataset, collate_fn
from torch_dicom.datasets.metadata import (
    BoundingBoxMetadata,
    DataFrameMetadata,
    MetadataDatasetWrapper,
    MetadataInputWrapper,
    PreprocessingConfigMetadata,
)
from torch_dicom.preprocessing import MinMaxCrop, Resize
from torch_dicom.preprocessing.pipeline import OutputFormat, PreprocessingPipeline


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
        pipeline = PreprocessingPipeline(dicoms=dicoms, output_format=OutputFormat.TIFF)
        out_files = pipeline(dest)
        assert out_files
        return dest

    @pytest.fixture(scope="class")
    def dataset(self, preprocessed_data) -> Dataset:
        paths = preprocessed_data.rglob("*.tiff")
        dataset = ImagePathDataset(paths)
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

    def test_dest_key(self, dataset: Dataset, metadata: Path):
        dest_key = "new_key"
        wrapper = PreprocessingConfigMetadata(dataset, dest_key=dest_key)
        example = wrapper[0]
        metadata_dict = wrapper.get_metadata(example)
        assert isinstance(metadata_dict, dict)
        assert metadata_dict[dest_key]


class TestBoundingBoxMetadata:
    @pytest.fixture(scope="class")
    def preprocessed_data(self, tmp_path_factory, dicoms):
        dest = tmp_path_factory.mktemp("data")
        transforms = [
            MinMaxCrop(),
            Resize((512, 384)),
        ]
        pipeline = PreprocessingPipeline(dicoms=dicoms, transforms=transforms, output_format=OutputFormat.TIFF)
        out_files = pipeline(dest)
        assert out_files
        return dest

    @pytest.fixture(scope="class")
    def dataset(self, preprocessed_data) -> Dataset:
        paths = preprocessed_data.rglob("*.tiff")
        dataset = ImagePathDataset(paths)
        assert len(dataset), "Failed to create dataset"
        return PreprocessingConfigMetadata(dataset)

    @pytest.fixture(scope="class")
    def box_data(self, dicoms, tmp_path_factory) -> Path:
        dest = tmp_path_factory.mktemp("boxes")
        boxes = []
        for dicom in dicoms[::2]:
            seed = int(hashlib.md5(dicom.SOPInstanceUID.encode()).hexdigest(), 16) % (2**32)
            np.random.seed(seed)
            rows, cols = dicom.Rows, dicom.Columns
            x1 = np.random.randint(0, cols)
            y1 = np.random.randint(0, rows)
            x2 = np.random.randint(x1, cols)
            y2 = np.random.randint(y1, rows)
            metadata = {
                "SOPInstanceUID": dicom.SOPInstanceUID,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "extra": "metadata",
            }
            boxes.append(metadata)

        # Write boxes to a CSV file
        path = dest / "boxes.csv"
        boxes_df = pd.DataFrame(boxes)
        boxes_df.to_csv(path, index=False)
        return path

    @pytest.fixture(scope="class")
    def sopuids_without_boxes(self, dicoms):
        # Every odd index has no boxes
        return [dicom.SOPInstanceUID for dicom in dicoms[1::2]]

    def test_init(self, dataset: Dataset, box_data):
        wrapper = BoundingBoxMetadata(dataset, box_data)
        assert wrapper.dataset == dataset
        assert isinstance(wrapper.metadata, pd.DataFrame)

    def test_repr(self, dataset: Dataset, box_data):
        wrapper = BoundingBoxMetadata(dataset, box_data)
        assert isinstance(repr(wrapper), str)

    def test_getitem_with_trace(self, dataset: Dataset, box_data, sopuids_without_boxes):
        wrapper = BoundingBoxMetadata(dataset, box_data, extra_keys=["extra"])
        for example in wrapper:
            if example["path"].stem not in sopuids_without_boxes:
                break
        else:
            raise AssertionError("No example with boxes found")

        assert isinstance(example, dict)
        boxes = example["bounding_boxes"]["boxes"]
        assert isinstance(boxes, BoundingBoxes)
        assert boxes.format == BoundingBoxFormat.XYXY
        assert boxes.canvas_size == (512, 384)
        assert 0 <= boxes[..., -2] <= 384
        assert 0 <= boxes[..., -1] <= 512
        assert example["bounding_boxes"]["extra"] == ["metadata"]

    def test_getitem_without_trace(self, dataset: Dataset, box_data, sopuids_without_boxes):
        wrapper = BoundingBoxMetadata(dataset, box_data)
        for example in wrapper:
            if example["path"].stem in sopuids_without_boxes:
                break
        else:
            raise AssertionError("No example without boxes found")

        assert isinstance(example, dict)
        assert example["bounding_boxes"] == {}

    @pytest.mark.skip(reason="This needs to be fixed")
    def test_transform(self, dataset: Dataset, box_data):
        wrapper = BoundingBoxMetadata(dataset, box_data)

        # The ordering of samples in `wrapper` is not deterministic,
        # so we look for a particular sample that has `boxes`
        index = 0
        while str(wrapper[index]["record"].path)[-5] != "2":
            index += 1

        example1 = wrapper[index]
        wrapper.transform = RandomHorizontalFlip(1.0)
        example2 = wrapper[index]

        boxes1 = example1["bounding_boxes"]["boxes"]
        boxes2 = example2["bounding_boxes"]["boxes"]
        assert not (boxes1 == boxes2).all()

    def test_dest_key(self, dataset: Dataset, box_data):
        dest_key = "new_key"
        wrapper = BoundingBoxMetadata(dataset, box_data, dest_key=dest_key)
        for i in range(2):
            example = wrapper[i]
            assert dest_key in example
            assert isinstance(example[dest_key], dict)


class TestDataFrameMetadata:
    @pytest.fixture(scope="class")
    def preprocessed_data(self, tmp_path_factory, dicoms):
        dest = tmp_path_factory.mktemp("data")
        pipeline = PreprocessingPipeline(dicoms=dicoms, output_format=OutputFormat.TIFF)
        out_files = pipeline(dest)
        assert out_files
        return dest

    @pytest.fixture(scope="class")
    def dataset(self, preprocessed_data) -> Dataset:
        paths = preprocessed_data.rglob("*.tiff")
        dataset = ImagePathDataset(paths)
        assert len(dataset), "Failed to create dataset"
        return dataset

    @pytest.fixture(scope="class")
    def metadata(self, dicoms, tmp_path_factory) -> Path:
        dest = tmp_path_factory.mktemp("metadata")
        rows = []
        for dicom in dicoms[::2]:
            metadata = {
                "SOPInstanceUID": dicom.SOPInstanceUID,
                "rows": dicom.Rows,
                "columns": dicom.Columns,
            }
            rows.append(metadata)

        # Write boxes to a CSV file
        path = dest / "metadata.csv"
        boxes_df = pd.DataFrame(rows)
        boxes_df.to_csv(path, index=False)
        return path

    @pytest.fixture(scope="class")
    def sopuids_without_boxes(self, dicoms):
        return [dicom.SOPInstanceUID for dicom in dicoms[1::2]]

    def test_init(self, dataset: Dataset, metadata):
        wrapper = DataFrameMetadata(dataset, metadata)
        assert wrapper.dataset == dataset
        assert isinstance(wrapper.metadata, pd.DataFrame)

    def test_repr(self, dataset: Dataset, metadata):
        wrapper = DataFrameMetadata(dataset, metadata)
        assert isinstance(repr(wrapper), str)

    def test_getitem_with_metadata(self, dataset: Dataset, metadata: Path, sopuids_without_boxes, dicom_size):
        H, W = dicom_size
        wrapper = DataFrameMetadata(dataset, metadata)
        for example in wrapper:
            if example["path"].stem not in sopuids_without_boxes:
                break
        else:
            raise AssertionError("No example with metadata found")
        assert isinstance(example["metadata"], dict)
        assert example["metadata"]["rows"] == H
        assert example["metadata"]["columns"] == W

    def test_getitem_without_metadata(self, dataset: Dataset, metadata: Path, sopuids_without_boxes):
        wrapper = DataFrameMetadata(dataset, metadata)
        for example in wrapper:
            if example["path"].stem in sopuids_without_boxes:
                break
        else:
            raise AssertionError("No example without metadata found")
        assert example["metadata"] == {}

    def test_collate_fn(self, dataset: Dataset, metadata: Path):
        wrapper = DataFrameMetadata(dataset, metadata)
        batch = [wrapper[i] for i in range(len(wrapper))]
        collated = collate_fn(batch, default_fallback=False)
        assert isinstance(collated, dict)
        assert "metadata" in collated
        assert isinstance(collated["metadata"], list)
        assert len(collated["metadata"]) == len(batch)

    def test_dest_key(self, dataset: Dataset, metadata: Path):
        dest_key = "custom_key"
        wrapper = DataFrameMetadata(dataset, metadata, dest_key=dest_key)
        for i in range(2):
            example = wrapper[i]
            assert isinstance(example[dest_key], dict)
