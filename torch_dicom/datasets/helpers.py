from torch import Tensor


def normalize_pixels(pixels: Tensor, eps: float = 1e-6) -> Tensor:
    r"""
    Normalize the pixel values of a tensor to the range :math:`\[0, 1\]`.

    Args:
        pixels: Input tensor containing pixel values.
        eps: Small value to prevent division by zero, defaults to 1e-6.

    Returns:
        Normalized tensor.
    """
    pmin, pmax = pixels.aminmax()
    delta = (pmax - pmin).clip(min=eps)
    pixels = (pixels.float() - pmin).div_(delta)
    return pixels
