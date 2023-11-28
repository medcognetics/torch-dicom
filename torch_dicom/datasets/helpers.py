from typing import Any, Callable, Dict, Optional, Protocol, TypeVar, runtime_checkable

from torch import Tensor


E = TypeVar("E", bound=Dict[str, Any])

Transform = Callable[[E], E]


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


@runtime_checkable
class SupportsTransform(Protocol):
    """
    Protocol for classes that support applying a transform to an example.
    """

    transform: Optional[Transform]

    def on_before_transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Method to be called before the transform is applied.
        By default, it returns the example unchanged.

        Args:
            example: The example to be transformed.

        Returns:
            The example after any preprocessing.
        """
        return example

    def on_after_transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Method to be called after the transform is applied.
        By default, it returns the example unchanged.

        Args:
            example: The example that was transformed.

        Returns:
            The example after any postprocessing.
        """
        return example

    def apply_transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the transform to the example, with optional preprocessing and postprocessing.

        Args:
            example: The example to be transformed.

        Returns:
            The example after the transform and any preprocessing and postprocessing have been applied.
        """
        if self.transform is not None:
            example = self.on_before_transform(example)
            example = self.transform(example)
            example = self.on_after_transform(example)
        return example
