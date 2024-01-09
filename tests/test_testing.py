from typing import Any, List, cast

import pandas as pd
import pytest
import torch
from dicom_utils.container import DicomImageFileRecord
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from torch_dicom.datasets import ImagePathDataset
from torch_dicom.preprocessing.datamodule import PreprocessedPNGDataModule
from torch_dicom.testing import DicomTestFactory, MammogramTestFactory, choose_modulus, create_random_box


@pytest.mark.parametrize(
    "val, choices, expected",
    [
        (5, [1, 2, 3, 4, 5], 1),
        (0, [1, 2, 3, 4, 5], 1),
        (1, [1, 2, 3, 4, 5], 2),
        (10, ["a", "b", "c"], "b"),
        (2, ["a", "b", "c"], "c"),
    ],
)
def test_choose_modulus(val: int, choices: list, expected: Any):
    assert choose_modulus(val, choices) == expected


@pytest.mark.parametrize(
    "seed, canvas_size, expected",
    [
        (0, (10, 10), (5, 0, 9, 4)),
        (1, (20, 20), (5, 11, 18, 12)),
        (2, (30, 30), (8, 15, 22, 24)),
    ],
)
def test_create_random_box(seed: int, canvas_size: tuple, expected: tuple):
    act = create_random_box(seed, canvas_size)
    exp = BoundingBoxes(torch.tensor(expected), format=BoundingBoxFormat.XYXY, canvas_size=canvas_size)
    assert torch.allclose(act, exp)


class TestDicomTestFactory:
    @pytest.fixture
    def root(self, tmp_path):
        return tmp_path

    @pytest.fixture
    def factory(self, root):
        return DicomTestFactory(root)

    @pytest.fixture
    def expected_images(self, factory) -> int:
        return factory.num_studies

    def test_create_dicom_files(self, root, factory, expected_images):
        result = factory.create_dicom_files(root)
        assert all([p.is_file() for p in result])
        records = cast(List[DicomImageFileRecord], [DicomImageFileRecord.from_file(p) for p in result])
        assert len(result) == len(set(rec.SOPInstanceUID for rec in records)) == expected_images

        assert all(rec.Rows == factory.dicom_size[0] for rec in records)
        assert all(rec.Columns == factory.dicom_size[1] for rec in records)
        assert len(set(rec.StudyInstanceUID for rec in records)) == factory.num_studies

    @pytest.fixture
    def dicom_files(self, root, factory):
        return factory.create_dicom_files(root)

    def test_create_preprocessed_data(self, root, factory, dicom_files):
        result = factory.create_preprocessed_data(root, dicom_files)
        ds = ImagePathDataset(iter(result))
        assert len(ds) == len(dicom_files)
        assert isinstance(ds[0]["img"], torch.Tensor)

    @pytest.fixture
    def expected_manifest_columns(self):
        return {"StudyInstanceUID", "Patient", "path"}

    def test_create_manifest(self, factory, dicom_files, expected_manifest_columns):
        result = factory.create_manifest(dicom_files)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(dicom_files)
        assert result.index.name == "SOPInstanceUID"
        assert set(result.columns.tolist()) == expected_manifest_columns

    def test_create_annotation_manifest(self, factory, dicom_files):
        result = factory.create_annotation_manifest(dicom_files)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(dicom_files)
        assert result.index.name == "SOPInstanceUID"
        assert "trait" in result.columns.tolist()

    def test_create_trace_manifest(self, factory, dicom_files):
        result = factory.create_trace_manifest(dicom_files)
        assert isinstance(result, pd.DataFrame)
        assert 0 < len(result) <= len(dicom_files)
        assert result.index.name == "SOPInstanceUID"
        assert set(result.columns.tolist()) == {"x1", "y1", "x2", "y2", "trait", "types"}
        assert isinstance(result.iloc[0].x1, float)

    @pytest.mark.parametrize("setup", [False, True])
    def test_call(self, mocker, factory, setup):
        spy = mocker.spy(PreprocessedPNGDataModule, "setup")
        dm = factory(setup=setup)
        assert isinstance(dm, PreprocessedPNGDataModule)

        if setup:
            assert spy.call_count == 2
            # Check that all dataloaders are valid.
            # We dont check the example members, this should be tested in PreprocessedPNGDataModule
            for dl in (
                dm.train_dataloader(),
                dm.val_dataloader(),
                dm.test_dataloader(),
            ):
                batch = next(iter(dl))
                assert isinstance(batch, dict)
                # This will always be present
                assert "img" in batch
                # These are set by the args to __call__()
                assert "manifest" in batch
                assert "annotation" in batch
                assert "bounding_boxes" in batch

        else:
            spy.assert_not_called()


class TestMammogramTestFactory(TestDicomTestFactory):
    @pytest.fixture
    def factory(self, root):
        return MammogramTestFactory(root)

    @pytest.fixture
    def expected_images(self, factory) -> int:
        # 4x FFDM, DBT, S-view = 12 images
        return factory.num_studies * 12

    @pytest.fixture
    def expected_manifest_columns(self):
        cols = {"StudyInstanceUID", "Patient", "path"}
        return cols.union({"ViewPosition", "Laterality", "standard_view"})
