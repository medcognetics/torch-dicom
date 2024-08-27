import logging
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, TypedDict, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from dicom_utils.volume import SliceAtLocation, VolumeHandler
from PIL import Image as PILImage
from torch import Tensor
from torch.utils.data import IterableDataset
from torchvision.tv_tensors import Image as TVImage

from .dicom import filter_collatable_types, slice_iterable_for_multiprocessing
from .helpers import SupportsTransform, Transform, normalize_pixels
from .path import PathDataset, PathInput


class ImageExample(TypedDict):
    img: TVImage
    img_size: Tensor
    path: Optional[Path]


def save_image(
    img: Tensor, path: Path, dtype: np.dtype = cast(np.dtype, np.uint16), compression: str | None = None
) -> None:
    """
    Saves an image tensor to a file using PIL. Supports PNG and TIFF formats.
    Floating point inputs are expected to be in the range [0, 1].
    Floating point inputs will be converted to ``dtype`` before saving.

    Args:
        img: Image tensor.
        path: Path to the file.
        dtype: Data type to convert to before saving. Only used for floating point inputs.
        compression: Compression argument passed to ``PIL.Image.save``.

    Shapes:
        * ``img``: :math:`(C, H, W)` or :math:`(H, W)`
    """
    # Convert to channels last, squeeze C=1 if necessary
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.squeeze(0)
    if img.ndim == 3:
        img = img.movedim(0, -1)

    # Convert floating point inputs to the specified dtype
    if torch.is_floating_point(img):
        if not torch.all((0 <= img) & (img <= 1)):
            raise ValueError("Floating point inputs are expected to be in the range [0, 1].")  # pragma: no cover
        dtype = cast(np.dtype, dtype or np.uint16)
        dtype_max = np.iinfo(dtype).max
        img_np = (img * dtype_max).numpy().astype(dtype)
    else:
        img_np = img.numpy()

    if dtype == np.uint16 and img.ndim == 3:
        raise ValueError("Saving 3-channel uint16 images is not supported")  # pragma: no cover

    pil_mode = "I;16" if dtype == np.uint16 else None
    pil_img = PILImage.fromarray(img_np, mode=pil_mode)
    pil_img.save(str(path), compression=compression)


def load_image(inp: Union[PILImage.Image, Path]) -> TVImage:
    """
    Loads an image file using PIL. The loaded image will be min max normalized
    based on the dtype of the image.

    Args:
        inp: PIL Image or path to an image.

    Returns:
        Tensor: An image tensor.
    """
    if isinstance(inp, Path):
        if not inp.is_file():
            raise FileNotFoundError(f"{inp} does not exist")  # pragma: no cover
        img = PILImage.open(str(inp))
    else:
        img = inp

    # PIL doesn't seem to distinguish uint16 from int32. We will assume anything loaded
    # with mode "I" is uint16.
    if img.mode == "I":
        img = img.convert("I;16")

    img_tensor = np.array(img)
    dtype_max = np.iinfo(img_tensor.dtype).max
    img_tensor = (img_tensor / dtype_max).astype(np.float32)
    img_tensor = torch.from_numpy(img_tensor)

    # Ensure image is channels-first
    if len(img_tensor.shape) == 2:
        img_tensor = img_tensor.unsqueeze_(0)
    else:
        img_tensor = img_tensor.movedim(-1, 0)

    return TVImage(img_tensor)


class ImageInput(IterableDataset, SupportsTransform):
    r"""Dataset that iterates over PIL image objects and yields a metadata dictionary.

    Args:
        images: Iterable of PIL Image objects.
        img_size: Size of the image to be returned. If None, the original image size is returned.
        transform: Optional transform to be applied to the image.
        skip_errors: If True, errors are ignored and the next image is loaded. If False, the error is raised.
        volume_handler: Volume handler to be used to load the image.
        normalize: If True, the tensor is normalized to [0, 1].
    """

    def __init__(
        self,
        images: Iterable[PILImage.Image],
        img_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Transform] = None,
        skip_errors: bool = False,
        volume_handler: VolumeHandler = SliceAtLocation(),
        normalize: bool = False,
    ):
        self.images = images
        self.img_size = img_size
        self.transform = transform
        self.skip_errors = skip_errors
        self.volume_handler = volume_handler
        self.normalize = normalize

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(img_size={self.img_size}, normalize={self.normalize})"

    def __iter__(self) -> Iterator[ImageExample]:
        iterable = slice_iterable_for_multiprocessing(self.images)
        for t in iterable:
            try:
                yield self.load_example(t, self.img_size, self.apply_transform, self.normalize)
            except Exception as ex:
                if not self.skip_errors:
                    raise
                else:
                    logging.warn(
                        "Encountered error while loading Tensor but skip_errors is True, skipping", exc_info=ex
                    )

    @classmethod
    def load_raw_example(
        cls,
        img: PILImage.Image,
        img_size: Optional[Tuple[int, int]] = None,
        normalize: bool = False,
    ) -> ImageExample:
        r"""Loads an example, but does not perform any transforms.

        Args:
            pixels: PIL Image object.
            img_size: Size of the image to be returned. If None, the original image size is returned.
            normalize: If True, the tensor is normalized to [0, 1].

        Returns:
            An ImageExample without transforms applied
        """
        if not isinstance(img, PILImage.Image):
            raise TypeError(f"Expected PIL Image, got {type(img)}")  # pragma: no cover

        pixels = load_image(img)

        if normalize:
            pixels = normalize_pixels(pixels)

        img_size_tensor = torch.tensor(pixels.shape[-2:], dtype=torch.long)
        if img_size is not None:
            pixels = F.interpolate(pixels.unsqueeze_(0), img_size, mode="nearest").squeeze_(0)

        result = {
            "img": pixels,
            "img_size": img_size_tensor,
            "path": None,
        }
        return cast(ImageExample, result)

    @classmethod
    def load_example(
        cls,
        img: PILImage.Image,
        img_size: Optional[Tuple[int, int]],
        transform: Optional[Transform] = None,
        normalize: bool = False,
    ) -> ImageExample:
        r"""Loads a single Tensor example.

        Args:
            pixels: Image object.
            img_size: Size of the image to be returned. If None, the original image size is returned.
            transform: Optional transform to be applied to the image.
            normalize: If True, the tensor is normalized to [0, 1].

        Returns:
            A ImageExample
        """
        example = cls.load_raw_example(img, img_size, normalize)
        result = filter_collatable_types(example)

        if transform is not None:
            result = transform(result)

        return cast(ImageExample, result)


class ImagePathInput(ImageInput, PathInput, SupportsTransform):
    r"""Dataset that iterates over paths to image files and yields a metadata dictionary.

    Args:
        paths: Iterable of paths to image files.
        img_size: Size of the image to be returned. If None, the original image size is returned.
        transform: Optional transform to be applied to the image.
        skip_errors: If True, errors are ignored and the next Tensor is loaded. If False, the error is raised.
        normalize: If True, the tensor is normalized to [0, 1].
    """

    def __init__(
        self,
        paths: Iterable[Path],
        img_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Transform] = None,
        skip_errors: bool = False,
        normalize: bool = False,
    ):
        self.images = paths
        self.img_size = img_size
        self.transform = transform
        self.skip_errors = skip_errors
        self.normalize = normalize

    @classmethod
    def load_example(
        cls,
        path: Path,
        img_size: Optional[Tuple[int, int]],
        transform: Optional[Transform] = None,
        normalize: bool = False,
    ) -> ImageExample:
        if not isinstance(path, Path):
            raise TypeError(f"Expected Path, got {type(path)}")

        img = PILImage.open(str(path))
        example = super().load_example(img, img_size, transform, normalize)
        example["path"] = path
        return cast(ImageExample, example)


class ImagePathDataset(PathDataset, SupportsTransform):
    r"""Dataset that reads image files and returns a metadata dictionary. This dataset class scans over all input
    paths during instantiation. This takes time, but allows a dataset length to be determined.
    If you want to avoid this, use :class:`ImagePathInput` instead. This class is best suited for training.

    Args:
        paths: Iterable of paths to image files.
        img_size: Size of the image to be returned. If None, the original image size is returned.
        transform: Optional transform to be applied to the image.
        skip_errors: If True, errors are ignored and the next Tensor is loaded. If False, the error is raised.
        normalize: If True, the tensor is normalized to [0, 1].
    """

    def __init__(
        self,
        paths: Iterator[Path],
        img_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Transform] = None,
        normalize: bool = False,
    ):
        super().__init__(paths)
        self.img_size = img_size
        self.transform = transform
        self.normalize = normalize

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)}, img_size={self.img_size}, normalize={self.normalize})"

    def __getitem__(self, idx: int) -> ImageExample:
        if not 0 <= idx <= len(self):
            raise IndexError(f"Index {idx} is invalid for dataset length {len(self)}")
        path = self.files[idx]
        return ImagePathInput.load_example(path, self.img_size, self.apply_transform, self.normalize)

    def __iter__(self) -> Iterator[ImageExample]:
        for path in self.files:
            yield ImagePathInput.load_example(path, self.img_size, self.apply_transform, self.normalize)
