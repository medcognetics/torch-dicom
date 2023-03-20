#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch

from torch_dicom.preprocessing.resize import Resize


class TestResize:
    @pytest.fixture(params=[torch.float32, torch.int32])
    def inp(self, request, depth, height, width):
        dtype = request.param
        if depth > 1:
            x = torch.ones((1, 1, depth, height, width), dtype=dtype)
        else:
            x = torch.ones((1, 1, height, width), dtype=dtype)
        return x

    @pytest.mark.parametrize("smart_pad", [True, False])
    @pytest.mark.parametrize("preserve_aspect_ratio", [True, False])
    @pytest.mark.parametrize(
        "depth, height, width, target_h, target_w, aspect_match",
        [
            (1, 64, 64, 32, 32, True),
            (1, 64, 64, 64, 64, True),
            (1, 64, 64, 32, 16, False),
            (3, 64, 64, 32, 32, True),
            (3, 64, 64, 64, 64, True),
            (3, 64, 64, 32, 16, False),
        ],
    )
    def test_resize(self, inp, target_h, target_w, aspect_match, preserve_aspect_ratio, smart_pad):
        resized = Resize.resize(inp, (target_h, target_w), preserve_aspect_ratio, smart_pad=smart_pad)
        img = resized["img"]
        assert img.ndim == inp.ndim
        assert img.shape[-2:] == (target_h, target_w)

        if aspect_match or not preserve_aspect_ratio:
            assert resized["resized_h"] == target_h
            assert resized["resized_w"] == target_w
            assert (img == 1).all()
        else:
            assert (img != 1).any()

    @pytest.mark.parametrize(
        "depth, height, width, target_h, target_w, coords, exp",
        [
            (1, 64, 64, 32, 32, (32, 32), (16, 16)),
            (1, 64, 64, 32, 16, (32, 32), (8, 16)),
        ],
    )
    def test_apply_to_coords(self, inp, target_h, target_w, coords, exp):
        resized = Resize.resize(inp, (target_h, target_w))
        coords = torch.tensor([coords])
        exp = torch.tensor([exp])
        coords_resized = Resize.apply_to_coords(coords, resized)
        assert torch.allclose(coords_resized, exp)

    @pytest.mark.parametrize(
        "depth, height, width, target_h, target_w",
        [
            (1, 64, 64, 32, 32),
            (1, 64, 64, 64, 64),
            (1, 64, 64, 32, 16),
        ],
    )
    def test_coords_round_trip(self, inp, target_h, target_w):
        resized = Resize.resize(inp, (target_h, target_w))
        coords = torch.tensor(
            [
                [0, 0],
                [8, 8],
                [16, 16],
            ]
        )

        coords_resized = Resize.apply_to_coords(coords, resized)
        coords_orig = Resize.unapply_to_coords(coords_resized, resized)
        assert torch.allclose(coords, coords_orig)
