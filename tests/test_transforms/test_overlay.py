from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image as PILImage
from torchvision.tv_tensors import Image as TVImage

from torch_dicom.transforms import AddOverlay, AddWatermark


@pytest.fixture(scope="module")
def overlay_img(tmpdir_factory, dicom_size):
    path = Path(tmpdir_factory.mktemp("overlay"), "overlay.png")
    # Create float img
    H, W = dicom_size
    img = torch.rand(H, W)

    # Convert to uint8 and create alpha channel
    img = (img * 255).numpy().astype(np.uint8)
    alpha = np.ones((H, W), dtype=np.uint8) * 255

    # Save image
    img = PILImage.fromarray(np.stack([img, alpha], axis=-1), mode="LA")
    img.save(path)
    return path


class TestAddOverlay:
    @pytest.fixture
    def class_test(self):
        return AddOverlay

    def test_load_image(self, class_test, overlay_img, dicom_size):
        img = class_test(overlay_img.parent)._load_image(0)
        assert img.shape == (2, *dicom_size)
        assert img.is_floating_point()

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_transform(self, class_test, overlay_img, dicom_size, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available.")
        img = torch.zeros(1, *dicom_size).to(device)
        transform = class_test(overlay_img.parent, p=1.0)
        output = transform(TVImage(img))
        assert 0 <= output.min() <= output.max() <= 1
        assert output.unique().numel() > 1


class TestAddWatermark(TestAddOverlay):
    @pytest.fixture
    def class_test(self):
        return AddWatermark

    @pytest.mark.parametrize(
        "leading_dims",
        [
            (1,),
            (
                2,
                1,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "img, exp",
        [
            (
                torch.tensor(
                    [
                        [1, 1, 0, 0],
                        [1, 1, 0, 0],
                        [1, 1, 0, 0],
                        [1, 1, 0, 0],
                    ]
                ),
                {1, 3},
            ),
            (
                torch.tensor(
                    [
                        [1, 1, 1, 0],
                        [1, 1, 1, 0],
                        [1, 1, 0, 0],
                        [1, 1, 0, 0],
                    ]
                ),
                {3},
            ),
            (
                torch.tensor(
                    [
                        [0, 0, 1, 0],
                        [0, 0, 1, 0],
                        [1, 1, 0, 0],
                        [1, 1, 0, 0],
                    ]
                ),
                {0, 3},
            ),
            (
                torch.tensor(
                    [
                        [1, 0, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ]
                ),
                {2, 3},
            ),
        ],
    )
    def test_select_grid_cell(self, class_test, img, leading_dims, exp):
        torch.random.manual_seed(42)
        img = torch.broadcast_to(img, leading_dims + img.shape)
        act = class_test._select_grid_cell(img)
        for t in act:
            assert t.item() in exp
