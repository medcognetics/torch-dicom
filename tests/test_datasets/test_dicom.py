#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, cast

import pytest
import torch
from dicom_utils.container import MammogramFileRecord
from dicom_utils.dicom import Dicom
from dicom_utils.dicom_factory import DicomFactory
from dicom_utils.volume import KeepVolume, ReduceVolume
from torch import Tensor
from torch.testing import assert_close
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import ColorJitter
from torchvision.tv_tensors import Image, Video

import torch_dicom
from torch_dicom.datasets.dicom import (
    DUMMY_PATH,
    DicomInput,
    DicomINRDataset,
    DicomPathDataset,
    DicomPathInput,
    collate_fn,
    uncollate,
)


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

    def test_collate_nested_dicts(self):
        batch = [
            {"d1": {"d2": torch.rand(3)}},
            {"d1": {"d2": torch.rand(3)}},
        ]
        collated = cast(Dict[str, Any], collate_fn(batch, False))
        assert isinstance(collated, dict)
        assert isinstance(collated["d1"], list)
        assert len(collated["d1"]) == 2
        assert isinstance(collated["d1"][0], dict)

    @pytest.mark.parametrize("missing_value", [None, -1])
    def test_collate_missing_values(self, missing_value):
        batch = [
            {"d1": Path("foo")},
            {},
        ]
        collated = collate_fn(batch, False, missing_value=missing_value)
        assert isinstance(collated, dict)
        assert collated["d1"] == [Path("foo"), missing_value]


class TestDicomInput:
    TEST_CLASS: ClassVar = DicomInput

    @pytest.fixture
    def dataset_input(self, dicom_iterator):
        return dicom_iterator

    @pytest.mark.parametrize("num_frames", [1, 5])
    def test_load_pixels(self, num_frames, dicom_size):
        H, W = dicom_size
        fact = DicomFactory(Modality="MG", Rows=H, Columns=W, NumberOfFrames=num_frames)
        dcm = fact()
        img = self.TEST_CLASS.load_pixels(dcm)
        assert img.shape == (1, H, W)
        assert 0 <= img.min() <= img.max() <= 1
        assert img.unique().numel() > 1

    @pytest.mark.parametrize("inversion", [True, False])
    @pytest.mark.parametrize("voi_lut", [True, False])
    @pytest.mark.parametrize("rescale", [True, False])
    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("volume_handler", [ReduceVolume(), KeepVolume()])
    @pytest.mark.parametrize("img_size", [(64, 32), (32, 16)])
    @pytest.mark.parametrize("transform", [None])
    def test_iter(
        self, mocker, dataset_input, normalize, volume_handler, img_size, voi_lut, inversion, rescale, transform
    ):
        ds = iter(
            self.TEST_CLASS(
                dataset_input,
                normalize=normalize,
                volume_handler=volume_handler,
                img_size=img_size,
                voi_lut=voi_lut,
                inversion=inversion,
                rescale=rescale,
                transform=transform,
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
            assert example["img"].unique().numel() > 1
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

    def test_collate(self, dataset_input, dicom_size):
        H, W = dicom_size
        ds = iter(self.TEST_CLASS(dataset_input))
        e1 = next(ds)
        e2 = next(ds)
        batch = collate_fn([deepcopy(e1), deepcopy(e2)], False)
        assert isinstance(batch, dict)
        assert isinstance(batch["img"], Tensor) and batch["img"].shape == (2, 1, H, W)
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

    @pytest.mark.parametrize("volume_handler", [ReduceVolume(), KeepVolume()])
    def test_transform(self, dataset_input, volume_handler, mocker):
        dataset_input = list(dataset_input)
        transform = ColorJitter(brightness=0.1)
        spy = mocker.spy(transform, "forward")
        ds1 = self.TEST_CLASS(iter(dataset_input), volume_handler=volume_handler)
        ds2 = self.TEST_CLASS(iter(dataset_input), volume_handler=volume_handler, transform=transform)
        example1 = next(iter(ds1))
        example2 = next(iter(ds2))
        spy.assert_called_once()
        assert (example1["img"] != example2["img"]).any()
        assert example1["img_size"].shape == example2["img_size"].shape


class TestDicomPathInput(TestDicomInput):
    TEST_CLASS: ClassVar = DicomPathInput

    @pytest.fixture
    def dataset_input(self, file_iterator):
        return file_iterator

    @pytest.mark.parametrize("normalize", [True, False])
    def test_iter(self, dataset_input, normalize, dicom_size):
        H, W = dicom_size
        dataset_input = list(dataset_input)
        ds = iter(self.TEST_CLASS(dataset_input, normalize=normalize))
        seen = 0
        for i, example in enumerate(ds):
            seen += 1
            assert isinstance(example["img"], Image)
            assert example["img"].shape == (1, H, W)
            assert example["img"].dtype == (torch.float if normalize else torch.int32)
            assert example["img"].unique().numel() > 1
            assert isinstance(example["img_size"], Tensor) and example["img_size"].shape == (2,)
            assert isinstance(example["record"], MammogramFileRecord)
            assert example["record"].path == dataset_input[i]
            assert isinstance(example["dicom"], Dicom), "Dicom object not returned"
            assert not example["dicom"].get("PixelData", None), "PixelData not removed"
            assert not example["dicom"].get("pixel_array", None), "pixel_array not removed"
        assert seen == 12

    def test_collate(self, dataset_input, dicom_size):
        H, W = dicom_size
        dataset_input = list(dataset_input)
        ds = iter(self.TEST_CLASS(dataset_input))
        e1 = next(ds)
        e2 = next(ds)
        batch = collate_fn([e1, e2])
        assert isinstance(batch["img"], Tensor) and batch["img"].shape == (2, 1, H, W)
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
    def test_getitem(self, mocker, dataset_input, volume_handler, voi_lut, inversion, dicom_size):
        dataset_input = list(dataset_input)
        img_size = H, W = dicom_size
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
        expected_shape = (1, expected_num_frames, H, W) if expecting_3d else (1, *img_size)

        assert isinstance(example["img"], Video if expecting_3d else Image)
        assert example["img"].shape == expected_shape
        assert example["img"].dtype == torch.float
        assert example["img"].unique().numel() > 1
        assert isinstance(example["img_size"], Tensor) and example["img_size"].shape == (2,)
        assert isinstance(example["record"], MammogramFileRecord)
        assert example["record"].path == dataset_input[0]

        # Check kwargs forwarded to read_dicom_image
        assert spy.call_count == 1
        for call in spy.mock_calls:
            assert call.kwargs["voi_lut"] == voi_lut
            assert call.kwargs["volume_handler"] == volume_handler
            assert call.kwargs["inversion"] == inversion


class TestDicomINRDataset:
    TEST_CLASS: ClassVar = DicomINRDataset

    @pytest.fixture(params=["2D", "3D"])
    def dicom(self, request):
        fact = DicomFactory(Modality="MG", Rows=64, Columns=32, NumberOfFrames=5 if request.param == "3D" else 1)
        return fact(seed=0)

    @pytest.fixture
    def dicom_path(self, tmp_path, dicom):
        path = tmp_path / "dicom.dcm"
        dicom.save_as(path)
        return path

    @pytest.mark.parametrize("chunk_size", [1, 128])
    def test_len(self, dicom, dicom_path, chunk_size):
        H = dicom.Rows
        W = dicom.Columns
        D = dicom.NumberOfFrames
        ds = self.TEST_CLASS(dicom_path, chunk_size=chunk_size)
        assert len(ds) == math.ceil(H * W * D / chunk_size)

    @pytest.mark.parametrize("chunk_size", [50, 128])
    def test_getitem(self, dicom, dicom_path, chunk_size):
        ds = self.TEST_CLASS(dicom_path, chunk_size=chunk_size)
        flat_img = torch.cat([example["img"] for example in ds], dim=1)
        flat_grid = torch.cat([example["grid"] for example in ds], dim=1)

        img = ds.restore_shape(flat_img.unsqueeze(0)).squeeze(0)
        grid = ds.restore_shape(flat_grid.unsqueeze(0)).squeeze(0)

        N = dicom.NumberOfFrames
        H = dicom.Rows
        W = dicom.Columns
        if N > 1:
            assert img.shape == (1, N, H, W)
            assert grid.shape == (3, N, H, W)
        else:
            assert img.shape == (1, H, W)
            assert grid.shape == (2, H, W)

        assert_close(grid.sum(), grid.new_tensor(0), rtol=0, atol=1e-3)
