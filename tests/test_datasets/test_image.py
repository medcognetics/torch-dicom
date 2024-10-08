from copy import deepcopy
from typing import ClassVar

import numpy as np
import pytest
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.tv_tensors import Image as TVImage

from torch_dicom.datasets.dicom import collate_fn
from torch_dicom.datasets.image import ImageInput, ImagePathDataset, ImagePathInput, load_image, save_image


@pytest.mark.parametrize(
    "inp,dtype,format",
    [
        (torch.rand(32, 32), np.uint8, "png"),
        (torch.rand(3, 32, 32), np.uint8, "png"),
        (torch.rand(32, 32), np.uint16, "png"),
        (torch.rand(1, 32, 32), np.uint16, "png"),
        (torch.randint(0, 256, (32, 32), dtype=torch.uint8), np.uint8, "png"),
        (torch.rand(32, 32), np.uint8, "tiff"),
        (torch.rand(3, 32, 32), np.uint8, "tiff"),
        (torch.rand(32, 32), np.uint16, "tiff"),
        (torch.rand(1, 32, 32), np.uint16, "tiff"),
        (torch.randint(0, 256, (32, 32), dtype=torch.uint8), np.uint8, "tiff"),
    ],
)
def test_save_image(mocker, tmp_path, inp, dtype, format):
    spy = mocker.spy(Image, "fromarray")
    path = tmp_path / f"test.{format}"
    save_image(inp, path, dtype)
    assert path.is_file()

    spy.assert_called_once()
    args, _ = spy.call_args
    assert args[0].dtype == dtype
    if inp.is_floating_point():
        assert args[0].max() <= np.iinfo(dtype).max
        assert args[0].min() >= 0


@pytest.mark.parametrize("pil", [False, True])
@pytest.mark.parametrize(
    "inp,dtype,format",
    [
        (torch.rand(32, 32), np.uint8, "png"),
        (torch.rand(3, 32, 32), np.uint8, "png"),
        (torch.rand(32, 32), np.uint16, "png"),
        (torch.rand(1, 32, 32), np.uint16, "png"),
        (torch.randint(0, 256, (32, 32), dtype=torch.uint8), np.uint8, "png"),
        (torch.rand(32, 32), np.uint8, "tiff"),
        (torch.rand(3, 32, 32), np.uint8, "tiff"),
        (torch.rand(32, 32), np.uint16, "tiff"),
        (torch.rand(1, 32, 32), np.uint16, "tiff"),
        (torch.randint(0, 256, (32, 32), dtype=torch.uint8), np.uint8, "tiff"),
    ],
)
def test_load_image(mocker, tmp_path, inp, dtype, format, pil):
    spy = mocker.spy(Image, "fromarray")
    path = tmp_path / f"test.{format}"
    save_image(inp, path, dtype)

    inp = Image.open(path) if pil else path
    result = load_image(inp)
    assert isinstance(result, TVImage)
    spy.assert_called_once()


class TestImageInput:
    TEST_CLASS: ClassVar = ImageInput

    @pytest.fixture
    def dataset_input(self, image_input):
        return image_input

    @pytest.mark.parametrize("normalize", [True, False])
    def test_iter(self, dataset_input, normalize, dicom_size):
        ds = iter(self.TEST_CLASS(dataset_input, normalize=normalize))
        seen = 0
        for example in ds:
            seen += 1
            assert example["img"].shape == (1, *dicom_size) and example["img"].dtype == torch.float
            assert isinstance(example["img_size"], Tensor) and example["img_size"].shape == (2,)
            if normalize:
                assert example["img"].min() == 0 and example["img"].max() == 1
        assert seen == 12

    def test_collate(self, dataset_input, dicom_size):
        ds = iter(self.TEST_CLASS(dataset_input))
        e1 = next(ds)
        e2 = next(ds)
        batch = collate_fn([deepcopy(e1), deepcopy(e2)], False)
        assert isinstance(batch, dict)
        assert isinstance(batch["img"], Tensor) and batch["img"].shape == (2, 1, *dicom_size)
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


class TestImagePathInput(TestImageInput):
    TEST_CLASS: ClassVar = ImagePathInput

    @pytest.fixture
    def dataset_input(self, image_files):
        return image_files

    @pytest.mark.parametrize("normalize", [True, False])
    def test_iter(self, dataset_input, normalize, dicom_size):
        dataset_input = list(dataset_input)
        ds = iter(self.TEST_CLASS(dataset_input, normalize=normalize))
        seen = 0
        for i, example in enumerate(ds):
            seen += 1
            assert example["img"].shape == (1, *dicom_size) and example["img"].dtype == torch.float
            assert isinstance(example["img_size"], Tensor) and example["img_size"].shape == (2,)
            assert example["path"] == dataset_input[i]
            if normalize:
                assert example["img"].min() == 0 and example["img"].max() == 1
        assert seen == 12

    def test_collate(self, dataset_input, dicom_size):
        dataset_input = list(dataset_input)
        ds = iter(self.TEST_CLASS(dataset_input))
        e1 = next(ds)
        e2 = next(ds)
        batch = collate_fn([e1, e2])
        assert isinstance(batch["img"], Tensor) and batch["img"].shape == (2, 1, *dicom_size)
        assert isinstance(batch["img_size"], Tensor) and batch["img_size"].shape == (2, 2)
        assert batch["path"] == dataset_input[:2]


class TestImagePathDataset(TestImagePathInput):
    TEST_CLASS: ClassVar = ImagePathDataset

    def test_len(self, dataset_input):
        ds = self.TEST_CLASS(dataset_input)
        assert len(ds) == 12

    @pytest.mark.parametrize("normalize", [True, False])
    def test_getitem(self, dataset_input, normalize, dicom_size):
        dataset_input = list(dataset_input)
        ds = self.TEST_CLASS(iter(dataset_input), normalize=normalize)
        example = ds[0]
        assert example["img"].shape == (1, *dicom_size) and example["img"].dtype == torch.float
        assert isinstance(example["img_size"], Tensor) and example["img_size"].shape == (2,)
        assert example["path"] == dataset_input[0]
        if normalize:
            assert example["img"].min() == 0 and example["img"].max() == 1


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("format", ["png", "tiff"])
def test_save_and_load_image(tmp_path, dtype, format, dicom_size):
    img_tensor = torch.rand(1, *dicom_size)
    img_path = tmp_path / f"test.{format}"
    save_image(img_tensor, img_path, dtype)
    assert img_path.is_file()

    ds = ImagePathDataset(iter([img_path]), normalize=False)
    loaded_img = ds[0]["img"]

    assert torch.allclose(img_tensor, loaded_img, atol=1 / np.iinfo(dtype).max)
