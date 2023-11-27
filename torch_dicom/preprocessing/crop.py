#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Final, Optional, Tuple, TypeVar, Union, cast, overload

import pandas as pd
import torch
from dicom_utils.container import SOPUID, DicomImageFileRecord
from einops import reduce, repeat
from torch import Tensor


if TYPE_CHECKING:
    from ..datasets import DicomExample, TensorExample
else:
    DicomExample = Any
    TensorExample = Any


E = TypeVar("E", bound=Union[DicomExample, TensorExample, Dict[str, Any]])
DE = TypeVar("DE", bound=DicomExample)
COORDS_PER_BOX: Final = 4


@overload
def _validate_coords(
    coords: Tensor,
    bounds: Tensor,
    img_size: Union[Tensor, Tuple[int, int]],
) -> Tuple[Tensor, Tensor, Tensor]:
    ...


@overload
def _validate_coords(
    coords: Tensor,
    bounds: Tensor,
    img_size: None,
) -> Tuple[Tensor, Tensor, None]:
    ...


def _validate_coords(
    coords: Tensor,
    bounds: Tensor,
    img_size: Optional[Union[Tensor, Tuple[int, int]]],
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    # Validate inputs
    if coords.shape[-1] != 2:
        raise ValueError(f"Expected coords to have shape (..., 2), got {coords.shape}")
    if bounds.shape[-1] != 4:
        raise ValueError(f"Expected bounds to have shape (..., 4), got {bounds.shape}")
    if coords.shape[:-1] != bounds.shape[:-1]:
        raise ValueError(
            f"Expected coords and bounds to have the same shape except for the last dimension, "
            f"got {coords.shape} and {bounds.shape}"
        )

    # Validate img_size and convert to Tensor if necessary
    if isinstance(img_size, Tensor):
        if img_size.shape[-1] != 1:
            raise ValueError(f"Expected img_size to have shape (..., 2), got {img_size.shape}")
        elif img_size.shape[:-1] != bounds.shape[:-1]:
            raise ValueError(
                f"Expected img_size and bounds to have the same shape except for the last dimension, "
                f"got {img_size.shape} and {bounds.shape}"
            )
    elif isinstance(img_size, tuple):
        if len(img_size) != 2:
            raise ValueError(f"Expected img_size to have length 2, got {len(img_size)}")
        else:
            img_size = bounds.new_tensor(img_size)
            img_size = repeat(img_size, "d -> b d", b=bounds.numel() // 4)
    elif img_size is None:
        pass
    else:
        raise TypeError(f"Expected img_size to be a Tensor, tuple, or None, got {type(img_size)}")

    return coords, bounds, img_size


@dataclass
class Crop:
    r"""Performs cropping of an image.

    Args:
        img_key: Key for the image in the example dict.

        bounds_key: Key for the crop bounds in the example dict. The determined
            crop bounds will be stored in this location.

        img_dest_key: Key for the cropped image in the example dict. If None,
            the cropped image will be stored in the same location as the original image.

    Shape:
        - Input: :math:`(N, *, H, W)`
        - Output: :math:`(N, *, H', W')`
    """
    img_key: str = "img"
    bounds_key: str = "crop_bounds"
    img_dest_key: Optional[str] = None

    def __post_init__(self):
        if self.img_dest_key is None:
            self.img_dest_key = self.img_key

    def __call__(self, example: E, xyxy_coords: Tensor) -> E:
        """
        Applies the cropping operation to the image in the example dict.

        Args:
            example: A dictionary containing the image to be cropped.
            xyxy_coords: A tensor containing the coordinates for cropping in the format (x1, y1, x2, y2).

        Returns:
            The example dict with the cropped image and updated metadata.
        """
        if not isinstance(example, dict):
            raise TypeError(f"Expected example to be a dict, got {type(example)}")
        if not xyxy_coords.numel() == COORDS_PER_BOX:
            raise ValueError(f"Expected xyxy_coords to have exactly 4 elements, got {xyxy_coords.shape}")

        # Crop image
        img = example[self.img_key]
        x1, y1, x2, y2 = xyxy_coords.unbind(dim=-1)
        img = img[..., y1:y2, x1:x2]

        # Update metadata
        cast(dict, example)[self.img_dest_key] = img
        cast(dict, example)[self.bounds_key] = xyxy_coords
        return cast(E, example)

    @staticmethod
    def apply_to_coords(coords: Tensor, bounds: Tensor, clip: bool = True) -> Tensor:
        r"""Applies the crop bounds to a set of coordinates. The coordinates are expected to be in
        absolute coordinates and in the format :math:`(x_1, y_1, x_2, y_2)`. Operations are performed
        in-place.

        Args:
            coords: A tensor of shape :math:`(*, 2)` containing the coordinates to be cropped.
                Coordinates are expected to be in the format :math:`(x_1, y_1, x_2, y_2)` in absolute
                coordinates.

            bounds: A tensor of shape :math:`(*, 4)` containing the crop bounds.
                Bounds are expected to be in the format :math:`(x_1, y_1, x_2, y_2)` in absolute
                coordinates.

            clip: If True, the cropped coordinates will be clipped to lie within `bounds` before adjustment.

        Shape:
            - `coords`: :math:`(*, 2)`
            - `bounds`: :math:`(*, 4)`
            - Output: Same as `coords`

        Returns:
            A tensor of shape :math:`(N, 2)` containing the cropped coordinates.
        """
        coords, bounds, _ = _validate_coords(coords, bounds, None)
        assert coords.shape[:-1] == bounds.shape[:-1]

        # Apply crop bounds
        lower_bound = bounds[..., :2]
        upper_bound = bounds[..., 2:]
        if clip:
            coords.clamp_(min=lower_bound, max=upper_bound)
        coords.sub_(lower_bound)

        return coords

    @staticmethod
    def unapply_to_coords(coords: Tensor, bounds: Tensor, clip: bool = True) -> Tensor:
        r"""Unapplies the crop bounds to a set of coordinates. The coordinates are expected to be in
        absolute coordinates and in the format :math:`(x_1, y_1, x_2, y_2)`. Operations are performed
        in-place.

        Args:
            coords: A tensor of shape :math:`(*, 2)` containing the coordinates to be cropped.
                Coordinates are expected to be in the format :math:`(x_1, y_1, x_2, y_2)` in absolute
                coordinates.

            bounds: A tensor of shape :math:`(*, 4)` containing the crop bounds.
                Bounds are expected to be in the format :math:`(x_1, y_1, x_2, y_2)` in absolute
                coordinates.

            img_size: A tuple of (H, W) containing the image size, or a tensor of shape :math:`(*, 2)`.

            clip: If True, the cropped coordinates will be clipped to lie within `bounds` after adjustment.

        Shape:
            - `coords`: :math:`(*, 2)`
            - `bounds`: :math:`(*, 4)`
            - `img_size`: :math:`(*, 2)` or a tuple of (H, W)
            - Output: Same as `coords`

        Returns:
            A tensor of shape :math:`(N, 2)` containing the cropped coordinates.
        """
        coords, bounds, _ = _validate_coords(coords, bounds, None)
        assert coords.shape[:-1] == bounds.shape[:-1]

        # Apply crop bounds
        lower_bound = bounds[..., :2]
        upper_bound = bounds[..., 2:]
        coords.add_(lower_bound)
        if clip:
            coords.clamp_(min=lower_bound, max=upper_bound)

        return coords


@dataclass
class MinMaxCrop(Crop):
    r"""Performs cropping of an image fitted to the minimum and maximum
    nonzero pixel coordinates along the height and width axes. If additional
    dimensions are present, they are first reduced by taking a max along the
    additional dimensions.

    Args:
        img_key: Key for the image in the example dict.

        bounds_key: Key for the crop bounds in the example dict. The determined
            crop bounds will be stored in this location.

        img_dest_key: Key for the cropped image in the example dict. If None,
            the cropped image will be stored in the same location as the original image.

    Shape:
        - Input: :math:`(N, *, H, W)`
        - Output: :math:`(N, *, H', W')`
    """

    def __call__(self, example: E) -> E:
        """
        Applies the MinMaxCrop operation to the given example.

        Args:
            example: The example to be processed.

        Returns:
            The processed example.
        """
        img = example[self.img_key]
        bounds = self.get_bounds(img)
        return super().__call__(example, bounds)

    @staticmethod
    @torch.no_grad()
    def get_bounds(x: Tensor) -> Tensor:
        r"""Returns the crop bounds for the given image.

        Args:
            x: Image tensor.

        Shape:
            - Input: :math:`(N, *, H, W)`
            - Output: :math:`(N, 4)`

        Returns:
            Absolute crop bounds in the format :math:`(x_1, y_1, x_2, y_2)`.
        """
        # Validate / process shape
        if x.ndim < 3:
            raise ValueError(f"Expected input to have at least 3 dimensions, got {x.ndim}")
        elif x.ndim > 3:
            # Reduce * dimension by taking a max
            x = reduce(x, "b ... h w -> b h w", "max")
        assert x.ndim == 3, f"Expected input to have 3 dimensions, got {x.ndim}"

        # Create initial result where bounds are the full image
        N, H, W = x.shape
        bounds = torch.tensor([[0, 0, H, W]], device=x.device, dtype=torch.long)
        bounds = repeat(bounds, "b d -> (b c) d", c=N, d=4)

        # Find the first and last nonzero pixel in each row and column
        nonzero_b, nonzero_h, nonzero_w = x.nonzero(as_tuple=True)

        if nonzero_b.numel():
            # Ignore "in beta" warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Update x1 and y1 based on minimum nonzero coordinate in each row and column
                bounds[..., 0].scatter_reduce_(0, nonzero_b, nonzero_w, reduce="amin", include_self=False)
                bounds[..., 1].scatter_reduce_(0, nonzero_b, nonzero_h, reduce="amin", include_self=False)
                # Update x2 and y2 based on maximum nonzero coordinate in each row and column
                # Add 1 to the max coordinate to get the correct crop
                bounds[..., 2].scatter_reduce_(0, nonzero_b, nonzero_w, reduce="amax", include_self=False).add_(1)
                bounds[..., 3].scatter_reduce_(0, nonzero_b, nonzero_h, reduce="amax", include_self=False).add_(1)

        assert bounds.shape == (N, 4), f"Expected bounds to have shape ({N}, 4), got {bounds.shape}"
        return bounds


@dataclass(kw_only=True)
class ROICrop(Crop):
    r"""Crops an image according to ROI coordinates from a CSV file.

    The CSV file should have at least the following columns:
        SOPInstanceUID,x1,y1,x2,y2

    If multiple ROIs are present for a given image, a random ROI will be selected.

    Args:
        path:  Path to CSV file with ROIs

        img_key: Key for the image in the example dict.

        bounds_key: Key for the crop bounds in the example dict. The determined
            crop bounds will be stored in this location.

        img_dest_key: Key for the cropped image in the example dict. If None,
            the cropped image will be stored in the same location as the original image.

        min_size: Minimum size of the crop in the format (H, W). If the ROI is smaller than this,
            the crop will be expanded to this size.

    Shape:
        - Input: :math:`(C, H, W)`
        - Output: :math:`(C, H', W')`
    """
    path: Path
    min_size: Tuple[int, int] = (256, 256)

    def __post_init__(self):
        super().__post_init__()
        # Trigger cached property
        assert isinstance(self.df, pd.DataFrame)

    @cached_property
    def df(self) -> pd.DataFrame:
        return pd.read_csv(self.path, index_col="SOPInstanceUID")

    def __call__(self, example: DE) -> DE:
        """
        Applies the MinMaxCrop operation to the given example.

        Args:
            example: The example to be processed.

        Returns:
            The processed example.
        """
        is_dicom_example = (
            "img" in example
            and isinstance(example["img"], Tensor)
            and "record" in example
            and isinstance(example["record"], DicomImageFileRecord)
        )
        if not is_dicom_example:
            raise TypeError(
                f"Expected example to be a DicomExample, got {example.keys() if isinstance(example, dict) else type(example)}"
            )

        img = example[self.img_key]
        sopuid = example["record"].SOPInstanceUID

        if sopuid is not None:
            bounds = self.get_bounds(img, sopuid)
            example = super().__call__(example, bounds)

        return example

    @torch.no_grad()
    def get_bounds(self, x: Tensor, sopuid: SOPUID) -> Tensor:
        r"""Returns the crop bounds for the given image.

        Args:
            x: Image tensor.
            sopuid: SOPInstanceUID of the image.

        Shape:
            - Input: :math:`(*, H, W)`
            - Output: :math:`(4,)`

        Returns:
            Absolute crop bounds in the format :math:`(x_1, y_1, x_2, y_2)`.
        """
        # Validate / process shape
        if x.ndim != 3:
            raise ValueError(f"Expected input to have at exactly 3 dimensions, got {x.ndim}")

        H, W = x.shape[-2:]
        H_min, W_min = self.min_size

        if sopuid not in self.df.index:
            non_zero_indices = torch.nonzero(x)
            if non_zero_indices.numel():
                # Try to find a crop that contains a nonzero pixel
                center = non_zero_indices[torch.randint(0, len(non_zero_indices), (1,))]
                center_y, center_x = tuple(int(t.item()) for t in center[0, -2:])
                h_start = max(0, center_y - H_min // 2)
                h_end = min(H, h_start + H_min)
                w_start = max(0, center_x - W_min // 2)
                w_end = min(W, w_start + W_min)
            else:
                # Choose a random crop
                h_start = torch.randint(0, H - H_min + 1, (1,))
                h_end = h_start + H_min
                w_start = torch.randint(0, W - W_min + 1, (1,))
                w_end = w_start + W_min

            # Adjust the start and end if necessary
            if h_end - h_start < H_min:
                h_start = H - H_min
            if w_end - w_start < W_min:
                w_start = W - W_min

            assert h_end <= H
            assert w_end <= W
            # In xyxy format
            bounds = torch.tensor([w_start, h_start, w_end, h_end])

        else:
            # Get matches for sopuid in self.df
            matches = self.df.loc[sopuid]
            matches = matches[["x1", "y1", "x2", "y2"]]

            # If there are multiple matches, choose one ROI at random
            if isinstance(matches, pd.DataFrame) and len(matches) > 1:
                match = matches.sample(n=1)
                x1, y1, x2, y2 = match.values[0]
            else:
                match = matches
                x1, y1, x2, y2 = match.values
            # If the ROI is smaller than self.min_size, expand the bounds
            if (x2 - x1) < H_min or (y2 - y1) < W_min:
                # Calculate the amount of expansion needed in each direction
                expand_x = H_min - (x2 - x1)
                expand_y = W_min - (y2 - y1)

                # Generate random jitter for x and y within the expansion bounds
                jitter_x = torch.randint(0, expand_x + 1, (1,))
                jitter_y = torch.randint(0, expand_y + 1, (1,))

                # Apply the jitter to the ROI bounds, ensuring they stay within the image bounds
                x1 = max(0, x1 - jitter_x)
                y1 = max(0, y1 - jitter_y)
                x2 = min(W, x1 + H_min)
                y2 = min(H, y1 + W_min)

            # In xyxy format
            bounds = torch.tensor([x1, y1, x2, y2])

        return bounds
