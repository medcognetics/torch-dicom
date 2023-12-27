import torch

from torch_dicom.datasets.helpers import normalize_pixels


def test_normalize_pixels(dicom_size):
    H, W = dicom_size
    pixels = torch.randint(0, 2**10, (1, H, W), dtype=torch.long)
    out = normalize_pixels(pixels)
    assert out.is_floating_point()
    assert out.min() == 0 and out.max() == 1
