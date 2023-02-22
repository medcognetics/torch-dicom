#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import ClassVar

import pytest
import torch
from dicom_utils.container import DicomImageFileRecord
from dicom_utils.dicom import Dicom
from dicom_utils.dicom_factory import DicomFactory
from torch import Tensor
from torch.utils.data import DataLoader

from torch_dicom.datasets.dicom import DUMMY_PATH, DicomInput, DicomPathDataset, DicomPathInput, collate_fn


class TestDicomInput:
    TEST_CLASS: ClassVar = DicomInput

    @pytest.fixture
    def dataset_input(self, dicom_iterator):
        return dicom_iterator

    @pytest.mark.parametrize("num_frames", [1, 5])
    def test_load_pixels(self, num_frames):
        H, W = 512, 384
        fact = DicomFactory(Modality="MG", Rows=H, Columns=W, NumberOfFrames=num_frames)
        dcm = fact()
        img = self.TEST_CLASS.load_pixels(dcm)
        assert img.shape == (1, H, W)

    def test_normalize_pixels(self):
        pixels = torch.randint(0, 2**10, (1, 2048, 1536), dtype=torch.long)
        out = self.TEST_CLASS.normalize_pixels(pixels)
        assert out.is_floating_point()
        assert out.min() == 0 and out.max() == 1

    def test_iter(self, dataset_input):
        ds = iter(self.TEST_CLASS(dataset_input))
        seen = 0
        for example in ds:
            seen += 1
            assert example["img"].shape == (1, 2048, 1536) and example["img"].dtype == torch.float
            assert isinstance(example["img_size"], Tensor) and example["img_size"].shape == (2,)
            assert isinstance(example["record"], DicomImageFileRecord)
            assert example["record"].path == DUMMY_PATH
            assert isinstance(example["dicom"], Dicom), "Dicom object not returned"
            assert not example["dicom"].get("PixelData", None), "PixelData not removed"
            assert not example["dicom"].get("pixel_array", None), "pixel_array not removed"
        assert seen == 12

    def test_collate(self, dataset_input):
        ds = iter(self.TEST_CLASS(dataset_input))
        e1 = next(ds)
        e2 = next(ds)
        batch = collate_fn([deepcopy(e1), deepcopy(e2)], False)
        assert isinstance(batch, dict)
        assert isinstance(batch["img"], Tensor) and batch["img"].shape == (2, 1, 2048, 1536)
        assert isinstance(batch["img_size"], Tensor) and batch["img_size"].shape == (2, 2)
        assert isinstance(batch["record"], list) and len(batch["record"]) == 2
        assert isinstance(batch["dicom"], list) and len(batch["dicom"]) == 2
        assert all(isinstance(r, DicomImageFileRecord) for r in batch["record"])
        assert all(b.path == DUMMY_PATH for b in batch["record"])

    def test_repr(self, dataset_input):
        ds = self.TEST_CLASS(dataset_input)
        assert isinstance(repr(ds), str)

    def test_iter_multiworker_dataloader(self, dataset_input):
        ds = self.TEST_CLASS(dataset_input)
        dl = DataLoader(ds, batch_size=2, num_workers=2, collate_fn=collate_fn)
        sop_uids = set()
        for batch in dl:
            for rec in batch["record"]:
                sop_uids.add(rec.SOPInstanceUID)
        assert len(sop_uids) == 12


class TestDicomPathInput(TestDicomInput):
    TEST_CLASS: ClassVar = DicomPathInput

    @pytest.fixture
    def dataset_input(self, file_iterator):
        return file_iterator

    def test_iter(self, dataset_input):
        dataset_input = list(dataset_input)
        ds = iter(self.TEST_CLASS(dataset_input))
        seen = 0
        for i, example in enumerate(ds):
            seen += 1
            assert example["img"].shape == (1, 2048, 1536) and example["img"].dtype == torch.float
            assert isinstance(example["img_size"], Tensor) and example["img_size"].shape == (2,)
            assert isinstance(example["record"], DicomImageFileRecord)
            assert example["record"].path == dataset_input[i]
            assert isinstance(example["dicom"], Dicom), "Dicom object not returned"
            assert not example["dicom"].get("PixelData", None), "PixelData not removed"
            assert not example["dicom"].get("pixel_array", None), "pixel_array not removed"
        assert seen == 12

    def test_collate(self, dataset_input):
        dataset_input = list(dataset_input)
        ds = iter(self.TEST_CLASS(dataset_input))
        e1 = next(ds)
        e2 = next(ds)
        batch = collate_fn([e1, e2])
        assert isinstance(batch["img"], Tensor) and batch["img"].shape == (2, 1, 2048, 1536)
        assert isinstance(batch["img_size"], Tensor) and batch["img_size"].shape == (2, 2)
        assert isinstance(batch["record"], list) and len(batch["record"]) == 2
        assert isinstance(batch["dicom"], list) and len(batch["dicom"]) == 2
        assert all(isinstance(r, DicomImageFileRecord) for r in batch["record"])
        assert [b.path for b in batch["record"]] == dataset_input[:2]


class TestDicomPathDataset(TestDicomPathInput):
    TEST_CLASS: ClassVar = DicomPathDataset

    @pytest.mark.skip(reason="Not implemented")
    def test_load_pixels(self):
        pass

    @pytest.mark.skip(reason="Not implemented")
    def test_normalize_pixels(self):
        pass

    def test_len(self, dataset_input):
        ds = self.TEST_CLASS(dataset_input)
        assert len(ds) == 12

    def test_getitem(self, dataset_input):
        dataset_input = list(dataset_input)
        ds = self.TEST_CLASS(iter(dataset_input))
        example = ds[0]
        assert example["img"].shape == (1, 2048, 1536) and example["img"].dtype == torch.float
        assert isinstance(example["img_size"], Tensor) and example["img_size"].shape == (2,)
        assert isinstance(example["record"], DicomImageFileRecord)
        assert example["record"].path == dataset_input[0]
