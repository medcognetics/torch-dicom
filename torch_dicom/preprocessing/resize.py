#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Final, Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .crop import E


EPS: Final = torch.finfo(torch.float32).eps


def make_tensor_like(proto: Tensor, value: Any, expand: bool = False, **kwargs) -> Tensor:
    result = proto.new_tensor(value, **kwargs) if not isinstance(value, Tensor) else value.type_as(proto)
    if expand:
        result = result.expand_as(proto)
    return result


@dataclass
class Resize:
    r"""Performs resizing on the image and updates the metadata. If `preserve_aspect_ratio` is True,`
    the image will be resized to fit within the given size while preserving the aspect ratio. If
    `preserve_aspect_ratio` is False, the image will be resized to exactly the given size.
    When `preserve_aspect_ratio` is True, the image will be resized to the largest possible size
    that fits within the given size while preserving the aspect ratio. The remaining space will
    be filled with the given `fill` value.

    Args:
        size: Size to resize the image to.

        mode: Interpolation mode to use. See :func:`torch.nn.functional.interpolate` for more details.
            Can also be ``"max"`` to use maximum pooling instead of interpolation.

        align_corners: See :func:`torch.nn.functional.interpolate` for more details.

        preserve_aspect_ratio: Whether to preserve the aspect ratio when resizing.

        img_key: Key for the image in the example dict.

        img_size_key: Key for the image size in the example dict.

        bounds_key: Key for the crop bounds in the example dict.

        img_dest_key: Key for the cropped image in the example dict. If None,
            the cropped image will be stored in the same location as the original image.

    Shape:
        - Input: :math:`(N, *, H, W)`
        - Output: :math:`(N, *, H', W')`
    """

    size: Tuple[int, int]
    mode: str = "bilinear"
    align_corners: Optional[bool] = None
    preserve_aspect_ratio: bool = True
    smart_pad: bool = True
    fill: Union[int, float] = 0

    img_key: str = "img"
    img_size_key: str = "img_size"
    resize_config_key: str = "resize_config"
    img_dest_key: Optional[str] = None

    def __post_init__(self):
        if self.img_dest_key is None:
            self.img_dest_key = self.img_key

    def __call__(self, example: E) -> E:
        if not isinstance(example, dict):
            raise TypeError(f"Expected example to be a dict, got {type(example)}")

        # This should be present, but it's not an error if it's missing.
        # We won't try to inject it ourselves because other augmentations may have
        # been applied.
        if self.img_size_key not in example:
            warnings.warn(
                f"Expected example to have '{self.img_size_key}' key, but it was not found. "
                "This may cause issues when trying to reconstruct the image."
            )

        # Resize image
        resize_result = self.resize(
            example[self.img_key],
            self.size,
            preserve_aspect_ratio=self.preserve_aspect_ratio,
            smart_pad=self.smart_pad,
            fill_value=self.fill,
            mode=self.mode,
        )

        # Update metadata
        cast(dict, example)[self.img_dest_key] = resize_result.pop("img")
        cast(dict, example)[self.resize_config_key] = resize_result
        return cast(E, example)

    @staticmethod
    @torch.no_grad()
    def resize(
        x: Tensor,
        size: Tuple[int, int],
        preserve_aspect_ratio: bool = True,
        smart_pad: bool = True,
        fill_value: Union[int, float] = 0,
        mode: str = "bilinear",
    ) -> Dict[str, Any]:
        r"""Resizes the given image to the given size.

        Args:
            x: Image tensor.

            size: Size to resize the image to.

            preserve_aspect_ratio: Whether to preserve the aspect ratio when resizing.

        Shape:
            - Input: :math:`(N, *, H, W)`
        """
        # Validate / process shape
        if x.ndim < 3:
            raise ValueError(f"Expected input to have at least 3 dimensions, got {x.ndim}")

        orig_shape = x.shape
        orig_dtype = x.dtype
        if x.ndim > 3:
            x = rearrange(x, "b ... h w -> (b ...) h w")
        assert x.ndim == 3, f"Expected input to have 3 dimensions, got {x.ndim}"

        # Get image size and target size
        N, H, W = x.shape
        H_target, W_target = size
        result_dict: Dict[str, Any] = {"orig_h": H, "orig_w": W}

        # Get scale factor
        scale_h, scale_w = (
            (min(H_target / H, W_target / W),) * 2 if preserve_aspect_ratio else (H_target / H, W_target / W)
        )

        # Get new size, accounting for floating point precision
        H_new = int(H * scale_h + EPS)
        W_new = int(W * scale_w + EPS)
        assert (preserve_aspect_ratio and (H_new == H_target or W_new == W_target)) or (
            not preserve_aspect_ratio and (H_new == H_target and W_new == W_target)
        ), "Resized image size does not match target size"
        result_dict["resized_h"] = H_new
        result_dict["resized_w"] = W_new

        # Resize image, converting to float32 if necessary
        x = x.float().unsqueeze_(1)
        if mode == "max":
            x = F.adaptive_max_pool2d(x, (H_new, W_new))
        else:
            x = F.interpolate(
                x,
                size=(H_new, W_new),
                mode=mode,
                align_corners=None,
            )

        # Pad image
        if preserve_aspect_ratio and (H_new != H_target or W_new != W_target):
            # Choose vertical padding to be centered
            pad_top = (H_target - H_new) // 2

            if smart_pad:
                # Choose horizontal padding based on what half of the image has a higher mean pixel value
                mean_left = x[..., : W_new // 2].mean(dim=(-1, -2)).flatten()
                mean_right = x[..., W_new // 2 :].mean(dim=(-1, -2)).flatten()
                pad_left = torch.where(mean_left > mean_right, 0, W_target - W_new)

            else:
                # Choose horizontal padding to be centered
                pad_left = x.new_tensor((W_target - W_new) // 2, dtype=torch.long)

            result_dict["padding_top"] = pad_top.cpu() if isinstance(pad_top, Tensor) else pad_top
            result_dict["padding_left"] = pad_left.cpu() if isinstance(pad_left, Tensor) else pad_left

            # Separate out the bounds and convert to tensor
            lb_H = make_tensor_like(x, pad_top).view(-1).expand(N)
            ub_H = make_tensor_like(x, pad_top + H_new).view(-1).expand(N)
            lb_W = make_tensor_like(x, pad_left).view(-1).expand(N)
            ub_W = make_tensor_like(x, pad_left + W_new).view(-1).expand(N)
            lb_H, ub_H, lb_W, ub_W = torch.broadcast_tensors(lb_H, ub_H, lb_W, ub_W)

            # Build indices where we want to copy to
            idx_N = torch.arange(N).view(N, 1, 1, 1)
            idx_C = torch.arange(1).view(1, 1, 1, 1)
            idx_H = torch.stack([torch.arange(lb_H[i], ub_H[i], dtype=torch.long) for i in range(N)], dim=0).view(
                N, 1, -1, 1
            )
            idx_W = torch.stack([torch.arange(lb_W[i], ub_W[i], dtype=torch.long) for i in range(N)], dim=0).view(
                N, 1, 1, -1
            )
            idx_N, idx_C, idx_H, idx_W = torch.broadcast_tensors(idx_N, idx_C, idx_H, idx_W)

            # Allocate result tensor and copy from original image
            result = x.new_full((N, 1, H_target, W_target), fill_value=fill_value)
            result[
                idx_N.flatten(),
                idx_C.view(-1),
                idx_H.flatten(),
                idx_W.flatten(),
            ] = x.view(-1)
            x = result

        # restore original shape but with resized dimensions
        x = x.view(*orig_shape[:-2], H_target, W_target)
        result_dict["img"] = x.to(orig_dtype)

        return result_dict

    @staticmethod
    def apply_to_coords(coords: Tensor, resize_config: Dict[str, Any]) -> Tensor:
        r"""Applies the resize to a set of coordinates. The coordinates are expected to be in
        absolute coordinates and in the format :math:`(x_1, y_1, x_2, y_2)`. Operations are performed
        in-place.

        Args:
            coords: A tensor of shape :math:`(*, 2)` containing the coordinates to be adjusted.
                Coordinates are expected to be in the format :math:`(x_1, y_1, x_2, y_2)` in absolute
                coordinates.

            resize_config: A dict containing the resize configuration. This is the same dict that
                is returned by :meth:`resize`.

        Shape:
            - `coords`: :math:`(*, 2)`
            - Output: Same as `coords`

        Returns:
            A tensor of shape :math:`(N, 2)` containing the adjusted coordinates.
        """
        H = torch.as_tensor(resize_config["orig_h"], dtype=torch.long, device=coords.device)
        W = torch.as_tensor(resize_config["orig_w"], dtype=torch.long, device=coords.device)
        H_new = torch.as_tensor(resize_config["resized_h"], dtype=torch.long, device=coords.device)
        W_new = torch.as_tensor(resize_config["resized_w"], dtype=torch.long, device=coords.device)

        # these keys are optional
        pad_top = resize_config.get("padding_top", None)
        pad_left = resize_config.get("padding_left", None)

        # Rescale coordinates to new size based on the ratio of new to old
        orig_dtype = coords.dtype
        coords = coords.float()
        coords[..., 0].mul_(make_tensor_like(coords, W_new / W))
        coords[..., 1].mul_(make_tensor_like(coords, H_new / H))

        # Adjust coordinates based on padding
        if pad_left is not None:
            coords[..., 0].add_(make_tensor_like(coords, pad_left))
        if pad_top is not None:
            coords[..., 1].add_(make_tensor_like(coords, pad_top))

        if not orig_dtype.is_floating_point:
            coords = coords.round().to(orig_dtype)

        return coords

    @staticmethod
    def unapply_to_coords(coords: Tensor, resize_config: Dict[str, Any]) -> Tensor:
        r"""Unapplies the resize to a set of coordinates. The coordinates are expected to be in
        absolute coordinates and in the format :math:`(x_1, y_1, x_2, y_2)`. Operations are performed
        in-place.

        Args:
            coords: A tensor of shape :math:`(*, 2)` containing the coordinates to be adjusted.
                Coordinates are expected to be in the format :math:`(x_1, y_1, x_2, y_2)` in absolute
                coordinates.

            resize_config: A dict containing the resize configuration. This is the same dict that
                is returned by :meth:`resize`.

        Shape:
            - `coords`: :math:`(*, 2)`
            - Output: Same as `coords`

        Returns:
            A tensor of shape :math:`(N, 2)` containing the adjusted coordinates.
        """
        H, W = resize_config["orig_h"], resize_config["orig_w"]
        H_new, W_new = resize_config["resized_h"], resize_config["resized_w"]

        # these keys are optional
        pad_top = resize_config.get("padding_top", None)
        pad_left = resize_config.get("padding_left", None)

        # NOTE: The order of the operations is reversed compared to apply_to_coords
        # Adjust coordinates based on padding.
        orig_dtype = coords.dtype
        coords = coords.float()
        if pad_left is not None:
            coords[..., 0].sub_(make_tensor_like(coords, pad_left))
        if pad_top is not None:
            coords[..., 1].sub_(make_tensor_like(coords, pad_top))

        # Rescale coordinates to original size based on the ratio of old to new
        coords[..., 0].mul_(make_tensor_like(coords, W / W_new))
        coords[..., 1].mul_(make_tensor_like(coords, H / H_new))

        if not orig_dtype.is_floating_point:
            coords = coords.round().to(orig_dtype)

        return coords
