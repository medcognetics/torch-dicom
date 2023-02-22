#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import ClassVar

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from torch_dicom.datasets.dicom import collate_fn
from torch_dicom.datasets.tensor import TensorInput, TensorPathDataset, TensorPathInput


class TestTensorInput:
    TEST_CLASS: ClassVar = TensorInput

    @pytest.fixture
    def dataset_input(self, tensor_input):
        return tensor_input

    def test_iter(self, dataset_input):
        ds = iter(self.TEST_CLASS(dataset_input))
        seen = 0
        for example in ds:
            seen += 1
            assert example["img"].shape == (1, 2048, 1536) and example["img"].dtype == torch.float
            assert isinstance(example["img_size"], Tensor) and example["img_size"].shape == (2,)
        assert seen == 12

    def test_collate(self, dataset_input):
        ds = iter(self.TEST_CLASS(dataset_input))
        e1 = next(ds)
        e2 = next(ds)
        batch = collate_fn([deepcopy(e1), deepcopy(e2)], False)
        assert isinstance(batch, dict)
        assert isinstance(batch["img"], Tensor) and batch["img"].shape == (2, 1, 2048, 1536)
        assert isinstance(batch["img_size"], Tensor) and batch["img_size"].shape == (2, 2)

    def test_repr(self, dataset_input):
        ds = self.TEST_CLASS(dataset_input)
        assert isinstance(repr(ds), str)

    def test_iter_multiworker_dataloader(self, dataset_input):
        ds = self.TEST_CLASS(dataset_input)
        dl = DataLoader(ds, batch_size=2, num_workers=2, collate_fn=collate_fn)
        example_sums = set(sum(e["img"]) for e in dl)
        # expect half as many unique sums for batch size 2
        assert len(example_sums) == 12 / 2


class TestTensorPathInput(TestTensorInput):
    TEST_CLASS: ClassVar = TensorPathInput

    @pytest.fixture
    def dataset_input(self, tensor_files):
        return tensor_files

    def test_iter(self, dataset_input):
        dataset_input = list(dataset_input)
        ds = iter(self.TEST_CLASS(dataset_input))
        seen = 0
        for i, example in enumerate(ds):
            seen += 1
            assert example["img"].shape == (1, 2048, 1536) and example["img"].dtype == torch.float
            assert isinstance(example["img_size"], Tensor) and example["img_size"].shape == (2,)
            assert example["path"] == dataset_input[i]
        assert seen == 12

    def test_collate(self, dataset_input):
        dataset_input = list(dataset_input)
        ds = iter(self.TEST_CLASS(dataset_input))
        e1 = next(ds)
        e2 = next(ds)
        batch = collate_fn([e1, e2])
        assert isinstance(batch["img"], Tensor) and batch["img"].shape == (2, 1, 2048, 1536)
        assert isinstance(batch["img_size"], Tensor) and batch["img_size"].shape == (2, 2)
        assert batch["path"] == dataset_input[:2]


class TestTensorPathDataset(TestTensorPathInput):
    TEST_CLASS: ClassVar = TensorPathDataset

    def test_len(self, dataset_input):
        ds = self.TEST_CLASS(dataset_input)
        assert len(ds) == 12

    def test_getitem(self, dataset_input):
        dataset_input = list(dataset_input)
        ds = self.TEST_CLASS(iter(dataset_input))
        example = ds[0]
        assert example["img"].shape == (1, 2048, 1536) and example["img"].dtype == torch.float
        assert isinstance(example["img_size"], Tensor) and example["img_size"].shape == (2,)
        assert example["path"] == dataset_input[0]
