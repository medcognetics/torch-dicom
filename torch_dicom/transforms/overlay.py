from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torch import Tensor
from torchvision.transforms.v2 import ConvertImageDtype, ToImage, Transform
from torchvision.tv_tensors import Image as TVImage
from tqdm import tqdm


class AddOverlay(Transform):
    r"""Add an overlay to the image at a random position. Currently, only `.png` images with a luminance
    and alpha channel are supported.

    Args:
        path: Path to directory containing overlay `.png` images.
        p: Probability of applying the transform.
        scale: Range of scales to apply to the overlay.
        force_in_bounds: If `True`, the overlay will be translated to ensure it is fully contained within the image.
        use_bar: If `True`, a progress bar will be displayed when loading the overlay images.
    """

    def __init__(
        self,
        path: Path,
        p: float = 0.2,
        scale: Tuple[float, float] = (0.75, 1.25),
        force_in_bounds: bool = True,
        use_bar: bool = True,
    ):
        super().__init__()
        if not path.is_dir():
            raise NotADirectoryError(path)  # pragma: no cover
        self.paths = list(
            tqdm(
                (p for p in path.rglob("*.png")),
                desc="Loading overlay images",
                unit="image",
                disable=not use_bar,
            )
        )
        self.p = p
        self.scale_range = scale
        self.force_in_bounds = force_in_bounds

        # TODO: Implement this
        if not self.force_in_bounds:
            raise NotImplementedError("`force_in_bounds=False` is not yet implemented")  # pragma: no cover

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params["augment"] or not isinstance(inpt, TVImage):
            return inpt
        if inpt.ndim > 3:
            raise ValueError("Batched inputs are not supported")

        # Load inputs
        idx = params["watermark_idx"]
        scale = params["scale"]
        center = params["center"]
        H, W = inpt.shape[-2:]
        watermark = self._load_image(idx).to(inpt.device)

        # Rescale the overlay according to random scale factor.
        # Ensure that the overlay is not larger than the input image.
        Hw, Ww = watermark.shape[-2:]
        Hw_scale, Ww_scale = int(Hw * scale), int(Ww * scale)
        Hw_scale = min(Hw_scale, H)
        Ww_scale = min(Ww_scale, W)
        watermark = F.interpolate(
            watermark.view(1, 2, Hw, Ww),
            size=(Hw_scale, Ww_scale),
            mode="nearest",
        ).view(2, Hw_scale, Ww_scale)

        # Determine bounds for indexing into the input image
        Hw, Ww = watermark.shape[-2:]
        lower_bounds = torch.stack(
            [
                center[..., 0] - Hw // 2,
                center[..., 1] - Ww // 2,
            ],
            dim=-1,
        )
        # If we require the entire overlay to be in bounds, we need to check
        # that the randomly chosen center is not too close to the edge. If it
        # is, we need to choose a new center.
        if self.force_in_bounds:
            lower_bounds.clip_(min=0)
            lower_bounds = torch.min(lower_bounds, lower_bounds.new_tensor([H - Hw, W - Ww]))

        # Otherwise, if we are out of bounds we need to slice the overlay
        else:
            # TODO: Implement this
            raise NotImplementedError  # pragma: no cover

        # Determine the upper bounds based on watermark size and lower bounds
        upper_bounds = lower_bounds + lower_bounds.new_tensor([Hw, Ww])
        bounds = torch.cat([lower_bounds, upper_bounds], -1)
        assert (
            (bounds[..., 0] >= 0).all()
            and (bounds[..., 1] >= 0).all()
            and (bounds[..., 2] <= H).all()
            and (bounds[..., 3] <= W).all()
        ), f"Bounds {bounds} are out of range for image of size {H}x{W}"

        # Apply the watermark
        target_for_watermark = inpt[..., bounds[0] : bounds[2], bounds[1] : bounds[3]]
        watermark, alpha = watermark.split(1)
        target_for_watermark = torch.where(alpha > 0, watermark, target_for_watermark)
        inpt[..., bounds[0] : bounds[2], bounds[1] : bounds[3]] = target_for_watermark
        return inpt

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        img = next(iter(i for i in flat_inputs if isinstance(i, TVImage)))
        H, W = img.shape[-2:]
        watermark_idx = int(torch.randint(len(self.paths), ()))
        augment = bool(torch.rand(1) < self.p)
        scale = float(torch.rand(1) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0])
        center = torch.rand(2, device=img.device) * torch.tensor([H, W], device=img.device)
        center = center.round().long()
        return {
            "augment": augment,
            "watermark_idx": watermark_idx,
            "scale": scale,
            "center": center,
        }

    def _load_image(self, idx: int) -> Tensor:
        """
        Load an image from the list of paths based on the provided index.

        Args:
            idx: The index of the image to load.

        Returns:
            The loaded image as a Tensor.
        """
        if not 0 <= idx < len(self.paths):
            raise IndexError(f"Index {idx} out of range")  # pragma: no cover
        path = self.paths[idx]

        img = Image.open(path)
        if img.mode != "LA":
            img = img.convert("LA")

        img = ToImage()(img)
        img = ConvertImageDtype(torch.float32)(img)
        return img


class AddWatermark(AddOverlay):
    r"""Add a machine watermark to the image. This transform will attempt to place the watermark
    in an empty region of the image. Currently, only `.png` images with a luminance and alpha channel
    are supported.

    Args:
        path: Path to directory containing overlay `.png` images.
        p: Probability of applying the transform.
        scale: Range of scales to apply to the overlay.
        force_in_bounds: If `True`, the overlay will be translated to ensure it is fully contained within the image.
        grid_size: Size of the grid to use when searching for an empty region of the image.
        use_bar: If `True`, a progress bar will be displayed when loading the watermark images.
    """

    def __init__(
        self,
        path: Path,
        p: float = 0.2,
        scale: Tuple[float, float] = (0.75, 1.25),
        force_in_bounds: bool = True,
        grid_size: int = 4,
        use_bar: bool = True,
    ):
        super().__init__(path, p, scale, force_in_bounds, use_bar)
        self.grid_size = grid_size

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        img = next(iter(i for i in flat_inputs if isinstance(i, TVImage)))
        params = super()._get_params(flat_inputs)

        # Select the grid cell where the watermark will be placed
        quadrant = self._select_grid_cell(img, grid_size=self.grid_size)
        H, W = img.shape[-2:]
        Hq, Wq = H // self.grid_size, W // self.grid_size

        # Convert this to a center coordinate
        quadrant_center = torch.cat(
            [H * (quadrant // self.grid_size) + Hq // 2, W * (quadrant % self.grid_size) + Wq // 2], dim=-1
        )

        params["center"] = quadrant_center.view(2)
        return params

    @staticmethod
    def _select_grid_cell(img: Tensor, grid_size: int = 2) -> Tensor:
        """
        Selects a grid cell from the image tensor.

        This method divides the image into a grid of size `grid_size` x `grid_size` and selects the cell
        with the lowest pixel sum. If multiple cells have the same sum, one is selected at random.

        Args:
            img: The image tensor.
            grid_size: The size of the grid. Defaults to 2 (meaning 2x2 grid).

        Returns:
            The selected grid cell.
        """
        # Reduce channel dim if multichannel
        if img.ndim > 3 and img.shape[-3] > 1:
            img = img.amax(dim=-3, keepdim=True)

        # Split the image into grid_size x grid_size cells
        img = rearrange(img, "... c (nh hq) (nw wq) -> ... c (hq wq) (nh nw)", nh=grid_size, nw=grid_size)

        # Sum pixels within each quadrant
        pixel_sum = img.sum(dim=-2)

        # Select the cell with the lowest pixel sum.
        # If multiple cells have the same sum, select one at random.
        min_pixel_count = pixel_sum.amin(dim=-1, keepdim=True)
        min_pixel_count = (pixel_sum == min_pixel_count).float()
        min_pixel_count = min_pixel_count / min_pixel_count.sum(dim=-1, keepdim=True)
        return torch.multinomial(min_pixel_count.view(-1, grid_size * grid_size), num_samples=1).view(-1, 1)
