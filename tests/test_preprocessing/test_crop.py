#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
import torch
from dicom_utils.container import DicomImageFileRecord

from torch_dicom.preprocessing.crop import MinMaxCrop, ROICrop


class TestMinMaxCrop:
    @pytest.fixture(params=[torch.float32, torch.int32])
    def inp(self, request, height, width, x1, y1, x2, y2):
        dtype = request.param
        x = torch.zeros((1, 1, height, width), dtype=dtype)
        x[..., y1:y2, x1:x2] = 1
        return x

    @pytest.mark.parametrize(
        "height, width, x1, y1, x2, y2, exp",
        [
            pytest.param(10, 10, 0, 0, 0, 0, (0, 0, 10, 10), id="all black"),
            (10, 10, 0, 0, 10, 10, (0, 0, 10, 10)),
            (10, 10, 0, 0, 10, 5, (0, 0, 10, 5)),
            (10, 10, 0, 0, 5, 10, (0, 0, 5, 10)),
            (10, 10, 0, 5, 10, 10, (0, 5, 10, 10)),
            (10, 10, 5, 0, 10, 10, (5, 0, 10, 10)),
            (10, 10, 5, 5, 10, 10, (5, 5, 10, 10)),
            (10, 10, 0, 0, 5, 5, (0, 0, 5, 5)),
            (10, 10, 0, 0, 5, 5, (0, 0, 5, 5)),
        ],
    )
    def test_get_bounds(self, inp, exp):
        bounds = MinMaxCrop.get_bounds(inp)
        exp = inp.new_tensor(exp, dtype=torch.long).view(1, 4)
        assert torch.allclose(bounds, exp)

    @pytest.mark.parametrize(
        "height, width, x1, y1, x2, y2, exp",
        [
            pytest.param(10, 10, 0, 0, 0, 0, (0, 0, 10, 10), id="all black"),
            (10, 10, 0, 0, 10, 10, (0, 0, 10, 10)),
            (10, 10, 0, 0, 10, 5, (0, 0, 10, 5)),
            (10, 10, 0, 0, 5, 10, (0, 0, 5, 10)),
            (10, 10, 0, 5, 10, 10, (0, 5, 10, 10)),
            (10, 10, 5, 0, 10, 10, (5, 0, 10, 10)),
            (10, 10, 5, 5, 10, 10, (5, 5, 10, 10)),
            (10, 10, 0, 0, 5, 5, (0, 0, 5, 5)),
            (10, 10, 0, 0, 5, 5, (0, 0, 5, 5)),
        ],
    )
    def test_call(self, inp, height, width, exp):
        crop = MinMaxCrop()
        example = {"img": inp, "img_size": torch.tensor([height, width])}
        result = crop(example)
        act = result["img"]

        # Compute bounds on the transformed input. The bounds should be equal to the full size
        # of the transformed input.
        bounds = MinMaxCrop.get_bounds(act)
        assert torch.allclose(bounds, bounds.new_tensor([0, 0, act.shape[-1], act.shape[-2]]))

        assert torch.allclose(result["crop_bounds"], MinMaxCrop.get_bounds(inp))

    @pytest.mark.parametrize(
        "clip, x, y, bounds, exp",
        [
            pytest.param(True, 0, 0, (0, 0, 10, 10), (0, 0)),
            pytest.param(True, 10, 10, (0, 0, 10, 10), (10, 10)),
            pytest.param(True, 10, 10, (0, 0, 5, 5), (5, 5)),
            pytest.param(False, 10, 10, (0, 0, 5, 5), (10, 10)),
            pytest.param(True, 10, 10, (10, 10, 20, 20), (0, 0)),
            pytest.param(True, 5, 5, (10, 10, 20, 20), (0, 0)),
            pytest.param(False, 5, 5, (10, 10, 20, 20), (-5, -5)),
        ],
    )
    def test_apply_to_coords(self, clip, x, y, bounds, exp):
        coords = torch.tensor([[x, y]], dtype=torch.long)
        bounds = torch.tensor(bounds, dtype=torch.long).view(1, 4)
        exp = torch.tensor(exp, dtype=torch.long).view_as(coords)
        act = MinMaxCrop.apply_to_coords(coords, bounds, clip)
        assert torch.allclose(act, exp)

    @pytest.mark.parametrize(
        "clip, x, y, bounds, exp",
        [
            pytest.param(True, 0, 0, (0, 0, 10, 10), (0, 0)),
            pytest.param(True, 10, 10, (0, 0, 10, 10), (10, 10)),
            pytest.param(True, 5, 5, (0, 0, 5, 5), (5, 5)),
            pytest.param(False, 10, 10, (0, 0, 5, 5), (10, 10)),
            pytest.param(True, 0, 0, (10, 10, 20, 20), (10, 10)),
            pytest.param(True, 5, 5, (10, 10, 20, 20), (15, 15)),
            pytest.param(False, 20, 20, (10, 10, 20, 20), (30, 30)),
            pytest.param(True, 20, 20, (10, 10, 20, 20), (20, 20)),
        ],
    )
    def test_unapply_to_coords(self, clip, x, y, bounds, exp):
        coords = torch.tensor([[x, y]], dtype=torch.long)
        bounds = torch.tensor(bounds, dtype=torch.long).view(1, 4)
        exp = torch.tensor(exp, dtype=torch.long).view_as(coords)
        act = MinMaxCrop.unapply_to_coords(coords, bounds, clip)
        assert torch.allclose(act, exp)


class TestROICrop:
    @pytest.fixture
    def crop(self, roi_crop_csv, min_size):
        return ROICrop(path=roi_crop_csv, min_size=min_size)

    @pytest.fixture(params=[torch.float32, torch.int32])
    def inp(self, mocker, request, height, width, sopuid):
        dtype = request.param
        x = torch.rand(1, height, width).to(dtype)
        rec = mocker.MagicMock(spec_set=DicomImageFileRecord, SOPInstanceUID=sopuid)
        return {
            "img": x,
            "record": rec,
        }

    @pytest.mark.parametrize(
        "height, width, sopuid, min_size, exp",
        [
            (30, 30, "1.2.3", (30, 30), (0, 0, 30, 30)),
            (30, 30, "1.2.3", (5, 5), (0, 0, 10, 10)),
            (30, 30, "2.3.4", (30, 30), (0, 0, 30, 30)),
            (30, 30, "4.5.6", (5, 5), (10, 10, 15, 15)),
        ],
    )
    def test_get_bounds(self, crop, sopuid, inp, exp):
        np.random.seed(0)
        pd.set_option("mode.chained_assignment", None)
        bounds = crop.get_bounds(inp["img"], sopuid)
        exp = inp["img"].new_tensor(exp, dtype=torch.long).view(4)
        assert torch.allclose(bounds, exp)

    @pytest.mark.parametrize(
        "height, width, sopuid, min_size, exp",
        [
            (30, 30, "1.2.3", (30, 30), (30, 30)),
            (30, 30, "1.2.3", (5, 5), (10, 10)),
        ],
    )
    def test_crop(self, crop, inp, exp):
        np.random.seed(0)
        pd.set_option("mode.chained_assignment", None)
        result = crop(inp)
        assert result["img"].shape[-2:] == exp
