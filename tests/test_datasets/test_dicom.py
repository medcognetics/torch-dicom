#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar

import pytest
import torch
from dicom_utils.container import MammogramFileRecord
from dicom_utils.dicom import Dicom
from dicom_utils.dicom_factory import DicomFactory
from dicom_utils.volume import KeepVolume, ReduceVolume
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.tv_tensors import Image, Video

import torch_dicom
from torch_dicom.datasets.dicom import DUMMY_PATH, DicomInput, DicomPathDataset, DicomPathInput, collate_fn, uncollate


class TestUncollate:
    def test_uncollate_tensors(self):
        batch = {"t1": torch.rand(2, 4), "t2": torch.rand(2, 8)}
        for i, example in enumerate(uncollate(batch)):
            assert isinstance(example, dict)
            assert example.keys() == batch.keys()
            for k, v in example.items():
                assert isinstance(v, Tensor)
                assert (v == batch[k][i]).all()

    def test_uncollate_mixed(self):
        batch = {"t1": torch.rand(2, 4), "paths": [DUMMY_PATH, DUMMY_PATH]}
        for i, example in enumerate(uncollate(batch)):
            assert isinstance(example, dict)
            assert example.keys() == batch.keys()
            for k, v in example.items():
                if isinstance(v, Tensor):
                    assert (v == batch[k][i]).all()
                else:
                    assert v == batch[k][i]

    def test_repeat(self):
        batch = {"t1": torch.rand(2, 4), "paths": 32}
        for i, example in enumerate(uncollate(batch)):
            assert isinstance(example, dict)
            assert example.keys() == batch.keys()
            for k, v in example.items():
                if isinstance(v, Tensor):
                    assert (v == batch[k][i]).all()
                else:
                    assert v == batch[k]


class TestCollate:
    def test_collate_tensors(self):
        batch = [{"t1": torch.rand(4), "t2": torch.rand(8)}, {"t1": torch.rand(4), "t2": torch.rand(8)}]
        collated = collate_fn(batch, False)
        assert isinstance(collated, dict)
        assert collated.keys() == batch[0].keys()
        for k, v in collated.items():
            assert isinstance(v, Tensor)
            assert v.shape == (2, *batch[0][k].shape)

    def test_collate_mixed(self):
        batch = [{"t1": torch.rand(4), "path": DUMMY_PATH}, {"t1": torch.rand(4), "path": DUMMY_PATH}]
        collated = collate_fn(batch, False)
        assert isinstance(collated, dict)
        assert collated.keys() == batch[0].keys()
        for k, v in collated.items():
            if isinstance(v, Tensor):
                assert v.shape == (2, *batch[0][k].shape)
            else:
                assert v == [batch[0][k], batch[1][k]]

    @pytest.mark.parametrize("dataclasses_as_lists", [True, pytest.param(False, marks=pytest.mark.xfail(strict=True))])
    def test_collate_dataclass(self, dataclasses_as_lists):
        @dataclass
        class Foo:
            x: int = 1

        batch = [{"t1": torch.rand(4), "foo": Foo()}, {"t1": torch.rand(4), "foo": Foo()}]
        collated = collate_fn(batch, False, dataclasses_as_lists=dataclasses_as_lists)
        assert isinstance(collated, dict)
        assert collated.keys() == batch[0].keys()
        for k, v in collated.items():
            if isinstance(v, Tensor):
                assert v.shape == (2, *batch[0][k].shape)
            else:
                assert v == [batch[0][k], batch[1][k]]


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

    @pytest.mark.parametrize("inversion", [True, False])
    @pytest.mark.parametrize("voi_lut", [True, False])
    @pytest.mark.parametrize("rescale", [True, False])
    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("volume_handler", [ReduceVolume(), KeepVolume()])
    @pytest.mark.parametrize("img_size", [(2048, 1536), (1024, 768)])
    def test_iter(self, mocker, dataset_input, normalize, volume_handler, img_size, voi_lut, inversion, rescale):
        ds = iter(
            self.TEST_CLASS(
                dataset_input,
                normalize=normalize,
                volume_handler=volume_handler,
                img_size=img_size,
                voi_lut=voi_lut,
                inversion=inversion,
                rescale=rescale,
            )
        )
        expected_num_frames = 3 if isinstance(volume_handler, KeepVolume) else None
        seen = 0
        spy = mocker.spy(torch_dicom.datasets.dicom, "read_dicom_image")  # type: ignore
        for example in ds:
            seen += 1
            expecting_3d = isinstance(volume_handler, KeepVolume) and not example["record"].is_2d
            expected_shape = (1, expected_num_frames, *img_size) if expecting_3d else (1, *img_size)
            assert isinstance(example["img"], Video if expecting_3d else Image)
            assert example["img"].shape == expected_shape
            assert example["img"].dtype == (torch.float if normalize else torch.int32)
            assert isinstance(example["img_size"], Tensor) and example["img_size"].shape == (2,)
            assert isinstance(example["record"], MammogramFileRecord)
            assert example["record"].path == DUMMY_PATH
            assert isinstance(example["dicom"], Dicom), "Dicom object not returned"
            assert not example["dicom"].get("PixelData", None), "PixelData not removed"
            assert not example["dicom"].get("pixel_array", None), "pixel_array not removed"
        assert seen == 12

        # Check kwargs forwarded to read_dicom_image
        assert spy.call_count == seen
        for call in spy.mock_calls:
            assert call.kwargs["voi_lut"] == voi_lut
            assert call.kwargs["volume_handler"] == volume_handler
            assert call.kwargs["inversion"] == inversion
            assert call.kwargs["rescale"] == rescale

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
        assert all(isinstance(r, MammogramFileRecord) for r in batch["record"])
        assert all(b.path == DUMMY_PATH for b in batch["record"])

    def test_uncollate(self, dataset_input):
        ds = iter(self.TEST_CLASS(dataset_input))
        e1 = next(ds)
        e2 = next(ds)
        batch = collate_fn([deepcopy(e1), deepcopy(e2)], False)
        examples = list(uncollate(batch))
        assert examples[0]["record"] == e1["record"]
        assert examples[1]["record"] == e2["record"]

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

    @pytest.mark.parametrize("normalize", [True, False])
    def test_iter(self, dataset_input, normalize):
        dataset_input = list(dataset_input)
        ds = iter(self.TEST_CLASS(dataset_input, normalize=normalize))
        seen = 0
        for i, example in enumerate(ds):
            seen += 1
            assert isinstance(example["img"], Image)
            assert example["img"].shape == (1, 2048, 1536)
            assert example["img"].dtype == (torch.float if normalize else torch.int32)
            assert isinstance(example["img_size"], Tensor) and example["img_size"].shape == (2,)
            assert isinstance(example["record"], MammogramFileRecord)
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
        assert all(isinstance(r, MammogramFileRecord) for r in batch["record"])
        assert [b.path for b in batch["record"]] == dataset_input[:2]


class TestDicomPathDataset(TestDicomPathInput):
    TEST_CLASS: ClassVar = DicomPathDataset

    @pytest.mark.skip(reason="Not implemented")
    def test_load_pixels(self):
        pass

    def test_len(self, dataset_input):
        ds = self.TEST_CLASS(dataset_input)
        assert len(ds) == 12

    @pytest.mark.parametrize("inversion", [True, False])
    @pytest.mark.parametrize("voi_lut", [True, False])
    @pytest.mark.parametrize("volume_handler", [ReduceVolume(), KeepVolume()])
    def test_getitem(self, mocker, dataset_input, volume_handler, voi_lut, inversion):
        dataset_input = list(dataset_input)
        img_size = (2048, 1536)
        ds = self.TEST_CLASS(
            iter(dataset_input),
            img_size=img_size,
            voi_lut=voi_lut,
            volume_handler=volume_handler,
            inversion=inversion,
        )
        spy = mocker.spy(torch_dicom.datasets.dicom, "read_dicom_image")  # type: ignore
        example = ds[0]

        expected_num_frames = 3 if isinstance(volume_handler, KeepVolume) else None
        expecting_3d = isinstance(volume_handler, KeepVolume) and not example["record"].is_2d
        expected_shape = (1, expected_num_frames, 2048, 1536) if expecting_3d else (1, *img_size)

        assert isinstance(example["img"], Video if expecting_3d else Image)
        assert example["img"].shape == expected_shape
        assert example["img"].dtype == torch.float
        assert isinstance(example["img_size"], Tensor) and example["img_size"].shape == (2,)
        assert isinstance(example["record"], MammogramFileRecord)
        assert example["record"].path == dataset_input[0]

        # Check kwargs forwarded to read_dicom_image
        assert spy.call_count == 1
        for call in spy.mock_calls:
            assert call.kwargs["voi_lut"] == voi_lut
            assert call.kwargs["volume_handler"] == volume_handler
            assert call.kwargs["inversion"] == inversion
