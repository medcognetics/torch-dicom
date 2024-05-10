#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Sequence, Set, cast

import numpy as np
import torch
from dicom_utils.dicom import Dicom, set_pixels
from dicom_utils.volume import KeepVolume, VolumeHandler
from pydicom.dataset import FileDataset
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm_multiprocessing import ConcurrentMapper

from .. import __version__
from ..datasets import AggregateInput, collate_fn, uncollate
from ..datasets.image import save_image


class OutputFormat(StrEnum):
    PNG = "png"
    DICOM = "dcm"


def update_dicom(dicom: Dicom, img: Tensor) -> Dicom:
    if img.ndim > 4:
        raise ValueError(f"img.ndim = {img.ndim}, expected <= 4")
    if img.shape[0] != 1:
        raise ValueError(f"img.shape[0] = {img.shape[0]}, expected 1")
    if img.is_floating_point():
        raise ValueError(f"img.dtype = {img.dtype}, expected integer")

    H, W = img.shape[-2:]
    if img.ndim == 4:
        img = img.view(1, -1, H, W)
    else:
        img = img.view(1, H, W)

    # Copy the dicom
    dicom = dicom.copy()

    # Update pixel description tags
    dicom.Rows = H
    dicom.Columns = W
    dicom.NumberOfFrames = img.shape[-3]

    # Set TransferSyntaxUID to ExplicitVRLittleEndian if compressed.
    # This matches what dicom_utils does when decompressing.
    new_syntax = ImplicitVRLittleEndian if dicom.is_implicit_VR else ExplicitVRLittleEndian
    dtype = np.uint16 if dicom.BitsAllocated == 16 else np.uint8
    dicom = set_pixels(cast(FileDataset, dicom), img.cpu().numpy().astype(dtype), syntax=new_syntax)

    # Disabled for performance
    # assert (dicom.pixel_array == img.cpu().numpy().astype(np.uint16)).all()

    return dicom


def recursive_tensor_to_list(x: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for k, v in x.items():
        if isinstance(v, dict):
            result[k] = recursive_tensor_to_list(v)
        elif isinstance(v, Tensor):
            result[k] = v.tolist()
        else:
            result[k] = v
    return result


@dataclass
class PreprocessingPipeline:
    r"""Preprocessing pipeline for DICOM images.

    Args:
        dicom_paths: Iterable of paths to DICOM files.
        dicoms: Iterable of DICOM objects.
        transforms: Iterable of transforms to apply to the image and DICOM object.
        batch_size: Batch size for the DataLoader.
        num_workers: Number of workers for the DataLoader and output threads.
        prefetch_factor: Prefetch factor for the DataLoader.
        device: Device to use for augmentations.
        volume_handler: VolumeHandler to use for handling volumes.
        dataloader_kwargs: Additional keyword arguments to pass to the DataLoader.
        use_bar: Whether to use a progress bar for the output threads.

    """

    dicom_paths: Iterable[Path] = field(default_factory=list)
    dicoms: Iterable[Dicom] = field(default_factory=list)

    # TODO: add transforms
    transforms: Sequence[Callable[[Dict[str, Any]], Dict[str, Any]]] = field(default_factory=list)

    batch_size: int = 1
    num_workers: int = 1
    prefetch_factor: int = 4
    device: torch.device = torch.device("cpu")
    volume_handler: VolumeHandler = KeepVolume()
    dataloader_kwargs: dict = field(default_factory=dict)
    use_bar: bool = True
    output_format: OutputFormat = OutputFormat.PNG

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for batch in self.dataloader:
            for example in uncollate(batch):
                # Update the DICOM object
                dicom = example["dicom"]
                img = example["img"]
                assert isinstance(dicom, Dicom)
                assert isinstance(img, Tensor)
                if self.output_format == "dcm":
                    dicom = update_dicom(dicom, img)
                    dicom["PixelData"].VR = "OB"

                # Propagate to dict and yield
                example["dicom"] = dicom
                yield example

    def __call__(self, dest: Path) -> Set[Path]:
        if not dest.is_dir():
            raise NotADirectoryError(f"{dest} is not a directory")

        # Write the config
        config_path = dest / "preprocess_config.json"
        with config_path.open("w") as f:
            json.dump(self.to_dict(), f, indent=4)

        # Use multiple threads to speed up file output.
        # Input is already accelerated by DataLoader multi-processing.
        with ConcurrentMapper(threads=True, jobs=self.num_workers, ignore_exceptions=False) as mapper:
            mapper.create_bar(desc="Preprocessing", disable=not self.use_bar)
            result: Set[Path] = set(
                mapper(lambda x: PreprocessingPipeline._save_as(x, dest, self.output_format), iter(self))
            )
        return result

    def apply_transforms(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        for transform in self.transforms:
            inp = transform(inp)
        return inp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": __version__,
            "transforms": [str(x) for x in self.transforms],
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "device": str(self.device),
            "volume_handler": str(self.volume_handler),
            "dataloader_kwargs": self.dataloader_kwargs,
            "use_bar": self.use_bar,
        }

    @cached_property
    def dataloader(self) -> DataLoader:
        # NOTE: We don't want to normalize the images here for DICOM output because we want them to be int dtype.
        # We don't want to apply the voi_lut here because it will be applied when loading the
        # preprocessed image and applying the voi_lut again will cause the image to be incorrect.
        # For other output formats we will apply these transformations.
        should_adjust_image = self.output_format != "dcm"
        ds = AggregateInput(
            dicom_paths=self.dicom_paths,
            dicoms=self.dicoms,
            normalize=should_adjust_image,
            voi_lut=should_adjust_image,
            inversion=should_adjust_image,
            rescale=should_adjust_image,
            volume_handler=self.volume_handler,
            skip_errors=True,
            transform=self.apply_transforms,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn,
            **self.dataloader_kwargs,
        )

    @staticmethod
    def _save_as(result: Dict[str, Any], dest: Path, output_format: OutputFormat) -> Path:
        dicom = result["dicom"]
        assert isinstance(dicom, Dicom)

        relative_dest = Path(f"{dicom.StudyInstanceUID}") / f"{dicom.SOPInstanceUID}.{str(output_format)}"

        # Save image as DICOM or PNG
        dest_path = dest / "images" / relative_dest
        dest_path.parent.mkdir(exist_ok=True, parents=True)
        if output_format == OutputFormat.DICOM:
            dicom.save_as(dest_path, write_like_original=False)
        elif output_format == OutputFormat.PNG:
            img = result["img"]
            img.squeeze_(0)
            dtype = cast(np.dtype, np.uint16 if dicom.BitsAllocated == 16 else np.uint8)
            save_image(img, dest_path, dtype)
        else:
            raise ValueError(f"Unknown output format {output_format}")

        # Convert record to dict for JSON serialization
        result["record"] = result["record"].to_dict()

        # Pop things we don't want to serialize in metadata
        result.pop("dicom")
        result.pop("img")
        result.pop("record")

        # Ensure any tensors are converted to lists
        result = recursive_tensor_to_list(result)

        # Save metadata JSON
        metadata_path = dest / "metadata" / relative_dest.with_suffix(".json")
        metadata_path.parent.mkdir(exist_ok=True, parents=True)
        with metadata_path.open("w") as f:
            json.dump(result, f)

        return dest_path
