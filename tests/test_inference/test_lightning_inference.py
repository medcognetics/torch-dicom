from pathlib import Path
from typing import Any, Optional

import pytest
import pytorch_lightning as pl
import torch.nn as nn
from torch import Tensor

from torch_dicom.inference.lightning import LightningInferencePipeline


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(1, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        img = batch["img"]
        return {"pred": self(img)}


class TestLightningInferencePipeline:
    @pytest.fixture
    def transform(self, mocker):
        transform = LightningInferencePipeline.create_default_transform(img_size=(8, 8))
        spy = mocker.spy(transform, "forward")
        return spy

    def test_dicoms(self, dicoms, file_iterator, transform):
        model = Model()
        files = list(file_iterator)
        pipeline = LightningInferencePipeline(
            dicom_paths=iter(files),
            dicoms=iter(dicoms),
            models=[model],
            skip_errors=False,
            transform=transform,
        )
        results = list(pipeline)
        assert len(results) == len(dicoms) + len(files)
        transform.assert_called()
        for example, pred in results:
            assert isinstance(example, dict)
            assert isinstance(pred, dict)
            assert isinstance(pred["pred"], Tensor)

    def test_iter_tensors(self, tensors, tensor_files, transform):
        model = Model()
        files = list(tensor_files)
        pipeline = LightningInferencePipeline(
            tensor_paths=iter(files),
            tensors=iter(tensors),
            models=[model],
            skip_errors=False,
            transform=transform,
        )
        results = list(pipeline)
        assert len(results) == len(tensors) + len(files)
        transform.assert_called()
        for example, pred in results:
            assert isinstance(example, dict)
            assert isinstance(pred, dict)
            assert isinstance(pred["pred"], Tensor)

    def test_iter_images(self, images, image_files, transform):
        model = Model()
        files = list(image_files)
        pipeline = LightningInferencePipeline(
            image_paths=iter(files),
            images=iter(images),
            models=[model],
            skip_errors=False,
            transform=transform,
        )
        results = list(pipeline)
        assert len(results) == len(images) + len(files)
        transform.assert_called()
        for example, pred in results:
            assert isinstance(example, dict)
            assert isinstance(pred, dict)
            assert isinstance(pred["pred"], Tensor)

    @pytest.mark.parametrize(
        "skip",
        [
            True,
            pytest.param(False, marks=pytest.mark.xfail(raises=FileNotFoundError, strict=True)),
        ],
    )
    def test_skip_errors(self, skip):
        model = Model()
        pipeline = LightningInferencePipeline(
            tensor_paths=[Path("not/a/real/path")],
            models=[model],
            skip_errors=skip,
        )
        results = list(pipeline)
        assert not len(results)

    def test_enumerate_inputs(self, file_iterator, tensor_files, image_files):
        model = Model()
        dicom_files = list(file_iterator)
        tensor_files = list(tensor_files)
        image_files = list(image_files)
        pipeline = LightningInferencePipeline(
            dicom_paths=iter(dicom_files),
            image_paths=iter(image_files),
            tensor_paths=iter(tensor_files),
            models=[model],
            skip_errors=False,
            enumerate_inputs=True,
        )
        results = list(pipeline)
        assert len(results) == len(dicom_files) + len(image_files) + len(tensor_files)
        for example, pred in results:
            assert isinstance(example, dict)
            assert isinstance(pred, dict)
            assert isinstance(pred["pred"], Tensor)
