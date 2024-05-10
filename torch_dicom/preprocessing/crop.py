#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import warnings
from copy import copy
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Final, List, Optional, Tuple, TypeVar, Union, cast, overload

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
) -> Tuple[Tensor, Tensor, Tensor]: ...


@overload
def _validate_coords(
    coords: Tensor,
    bounds: Tensor,
    img_size: None,
) -> Tuple[Tensor, Tensor, None]: ...


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

        roi_expansion: Amount to expand the ROI by. This ensures that adequte context is included in the crop.
            The default increases size by 50%

    Shape:
        - Input: :math:`(C, H, W)`
        - Output: :math:`(C, H', W')`
    """

    path: Path
    min_size: Tuple[int, int] = (256, 256)
    roi_expansion: float = 1.5

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
            # Choose a random height and width for the crop
            MAX_RANDOM_CROP_RATIO = 0.5
            max_H = max(H_min, int(H * MAX_RANDOM_CROP_RATIO))
            max_W = max(W_min, int(W * MAX_RANDOM_CROP_RATIO))
            crop_H = int(torch.randint(H_min, max_H + 1, (1,)))
            crop_W = int(torch.randint(W_min, max_W + 1, (1,)))

            non_zero_indices = torch.nonzero(x)
            if non_zero_indices.numel():
                # Try to find a crop that contains a nonzero pixel
                center = non_zero_indices[torch.randint(0, len(non_zero_indices), (1,))]
                center_y, center_x = tuple(int(t.item()) for t in center[0, -2:])
                y1 = max(0, center_y - crop_H // 2)
                y2 = min(H, y1 + crop_H)
                x1 = max(0, center_x - crop_W // 2)
                x2 = min(W, x1 + crop_W)
            else:
                # Choose a random crop
                y1 = int(torch.randint(0, H - crop_H + 1, (1,)))
                y2 = y1 + crop_H
                x1 = int(torch.randint(0, W - crop_W + 1, (1,)))
                x2 = x1 + crop_W

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

            # Expand the ROI in each direction to increase context
            x1, y1, x2, y2 = self.resize_roi(x1, y1, x2, y2, self.roi_expansion)

        # Change the aspect ratio to match that of self.min_size
        x1, y1, x2, y2 = self.apply_aspect_ratio(x1, y1, x2, y2, H_min / W_min)

        # If the ROI is smaller than self.min_size, expand the bounds
        if (x2 - x1) < W_min or (y2 - y1) < H_min:
            x1, y1, x2, y2 = self.resize_roi(x1, y1, x2, y2, (H_min, W_min))

        # If the ROI is larger than the frame, contract the bounds
        if (x2 - x1) > W or (y2 - y1) > H:
            target_h = min(H, (y2 - y1))
            target_w = min(W, (x2 - x1))
            x2 = x1 + target_w
            y2 = y1 + target_h
            assert x2 - x1 <= W, f"Expected x2 - x1 <= W, got {x2 - x1}, {W}"
            assert y2 - y1 <= H, f"Expected y2 - y1 <= H, got {y2 - y1}, {H}"

        # Shift the ROI to be in the frame
        x1, y1, x2, y2 = self.shift_roi_in_frame(x1, y1, x2, y2, H, W)

        # In xyxy format
        bounds = torch.tensor([x1, y1, x2, y2])
        assert 0 <= x1 < x2 <= W, f"Expected x1 < x2 < W, got {x1}, {x2}, {W}"
        assert 0 <= y1 < y2 <= H, f"Expected y1 < y2 < H, got {y1}, {y2}, {H}"
        bounds[0].clamp_(min=0, max=W)
        bounds[1].clamp_(min=0, max=H)
        bounds[2].clamp_(min=int(bounds[0]) + 1, max=W)
        bounds[3].clamp_(min=int(bounds[1]) + 1, max=H)

        return bounds

    @staticmethod
    def shift_roi_in_frame(x1: int, y1: int, x2: int, y2: int, height: int, width: int) -> Tuple[int, int, int, int]:
        """
        Shifts the region of interest (ROI) within the frame. If the ROI is outside of the frame, it is shifted
        by the minimum amount necessary to be within the frame. Does not change the size of the ROI. If the ROI
        is larger than the frame, there will still be a portion of the ROI outside of the frame.

        Args:
            x1: The x-coordinate of the top left corner of the ROI.
            y1: The y-coordinate of the top left corner of the ROI.
            x2: The x-coordinate of the bottom right corner of the ROI.
            y2: The y-coordinate of the bottom right corner of the ROI.
            height: The height of the frame.
            width: The width of the frame.

        Returns:
            The coordinates of the shifted ROI in the format (x1, y1, x2, y2).
        """
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > width:
            x1 = width - (x2 - x1)
            x2 = width
        if y2 > height:
            y1 = height - (y2 - y1)
            y2 = height
        return x1, y1, x2, y2

    @staticmethod
    def resize_roi(
        x1: int, y1: int, x2: int, y2: int, amount: Union[Tuple[int, int], float]
    ) -> Tuple[int, int, int, int]:
        """
        Resizes the region of interest (ROI) by a specified amount. The center of the ROI is unchanged. Resized ROI coordinates
        may lie outside of the frame.

        Args:
            x1: The x-coordinate of the top left corner of the ROI.
            y1: The y-coordinate of the top left corner of the ROI.
            x2: The x-coordinate of the bottom right corner of the ROI.
            y2: The y-coordinate of the bottom right corner of the ROI.
            amount: The amount to resize the ROI. If a float is provided, the ROI is resized by this factor.
                If a tuple is provided, the ROI is resized to the specified width and height.

        Returns:
            The coordinates of the resized ROI in the format (x1, y1, x2, y2).
        """
        if isinstance(amount, float):
            new_width = (x2 - x1) * amount
            new_height = (y2 - y1) * amount
        else:
            new_height, new_width = amount

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        x1 = int(center_x - new_width / 2)
        y1 = int(center_y - new_height / 2)
        x2 = int(center_x + new_width / 2)
        y2 = int(center_y + new_height / 2)
        return x1, y1, x2, y2

    @staticmethod
    def apply_aspect_ratio(x1: int, y1: int, x2: int, y2: int, aspect_ratio: float) -> Tuple[int, int, int, int]:
        """
        Adjusts the aspect ratio of the region of interest (ROI). The ROI is expanded or contracted about its center.
        Only the width or height of the ROI is changed, depending on which dimension is further from the desired
        aspect ratio.

        Args:
            x1: The x-coordinate of the top left corner of the ROI.
            y1: The y-coordinate of the top left corner of the ROI.
            x2: The x-coordinate of the bottom right corner of the ROI.
            y2: The y-coordinate of the bottom right corner of the ROI.
            aspect_ratio: The desired aspect ratio for the ROI (height / width)

        Returns:
            The coordinates of the ROI with the adjusted aspect ratio in the format (x1, y1, x2, y2).
        """
        current_aspect_ratio = (y2 - y1) / (x2 - x1)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        if current_aspect_ratio > aspect_ratio:
            # Increase width
            new_width = (y2 - y1) / aspect_ratio
            x1 = int(x_center - new_width / 2)
            x2 = int(x_center + new_width / 2)
        else:
            # Increase height
            new_height = (x2 - x1) * aspect_ratio
            y1 = int(y_center - new_height / 2)
            y2 = int(y_center + new_height / 2)

        return x1, y1, x2, y2


@dataclass
class TileCrop(Crop):
    r"""Performs cropping of an image into tiles.

    Note:
        Because this transform creates multiple tiles, it should not be followed by other image transforms.
        Additionally, :func:`apply_to_coords` and :func:`unapply_to_coords` will not work.

    Args:
        img_key: Key for the image in the example dict.

        bounds_key: Key for the crop bounds in the example dict. The determined
            crop bounds will be stored in this location.

        img_dest_key: Key for the cropped image in the example dict. If None,
            the cropped image will be stored in the same location as the original image.

        size: Size of the tiles in the format (H, W).

        overlap: Amount of overlap between tiles (in fractional units) in the format (H, W).

    Shape:
        - Input: :math:`(N, *, H, W)`
        - Output: :math:`(N, *, D, H', W')`
    """

    size: Tuple[int, int] = (256, 256)
    overlap: Tuple[float, float] = (0.2, 0.2)

    def __call__(self, example: E) -> E:
        # Get the tile bounds
        img: Tensor = example["img"]
        bounds = self.get_bounds(img)

        # TODO: Consider adding crop metadata and fixing apply_to_coords / unapply_to_coords
        # if it becomes necessary to invert the crops
        # Apply each crop
        images: List[Tensor] = [super(TileCrop, self).__call__(copy(example), b)["img"] for b in bounds]

        # Stack into a new "depth" dimension
        example["img"] = torch.stack(images, dim=-3)

        return example

    @torch.no_grad()
    def get_bounds(self, x: Tensor) -> Tensor:
        r"""Returns the crop bounds for the given image. Bounds are determined by
        splitting the image into tiles of size `self.size` with overlap `self.overlap`.

        Args:
            x: Image tensor.

        Shape:
            - Input: :math:`(*, H, W)`
            - Output: :math:`(N, 4)` where :math:`N` is the number of tiles.

        Returns:
            Absolute crop bounds in the format :math:`(x_1, y_1, x_2, y_2)`.
        """
        if x.ndim < 2:
            raise ValueError(f"Expected input to have at least 2 dimensions, got {x.ndim}")
        H, W = x.shape[-2:]
        Hc, Wc = self.size

        # Determine overlap in pixels
        overlap_h = int(self.overlap[0] * Hc)
        overlap_w = int(self.overlap[1] * Wc)

        # Calculate the number of tiles in each dimension
        num_tiles_h = math.ceil(H / (Hc - overlap_h))
        num_tiles_w = math.ceil(W / (Wc - overlap_w))

        # Calculate the starting points for each tile
        start_h = (torch.arange(num_tiles_h) * (Hc - overlap_h)).clip(max=H - Hc)
        start_w = (torch.arange(num_tiles_w) * (Wc - overlap_w)).clip(max=W - Wc)

        # Create a grid of starting points
        start_grid_h, start_grid_w = torch.meshgrid(start_h, start_w, indexing="ij")

        # Calculate the ending points for each tile
        end_grid_h = start_grid_h + Hc
        end_grid_w = start_grid_w + Wc

        # Validate
        assert torch.all((end_grid_h - start_grid_h) == Hc), "Height of all tiles should be equal to target crop height"
        assert torch.all((end_grid_w - start_grid_w) == Wc), "Width of all tiles should be equal to target crop width"

        # Stack the starting and ending points to create the bounds
        bounds = torch.stack(
            [start_grid_w.flatten(), start_grid_h.flatten(), end_grid_w.flatten(), end_grid_h.flatten()], dim=-1
        )

        return bounds

    @staticmethod
    def apply_to_coords(coords: Tensor, bounds: Tensor, clip: bool = True) -> Tensor:
        raise NotImplementedError("TileCrop does not support apply_to_coords")

    @staticmethod
    def unapply_to_coords(coords: Tensor, bounds: Tensor, clip: bool = True) -> Tensor:
        raise NotImplementedError("TileCrop does not support unapply_to_coords")
