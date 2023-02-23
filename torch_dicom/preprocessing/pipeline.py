#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Sequence, Set, cast

import numpy as np
import torch
from dicom_utils.dicom import ALGORITHM_PRESENTATION_TYPE, Dicom, set_pixels
from dicom_utils.volume import KeepVolume, VolumeHandler
from pydicom.dataset import FileDataset
from pydicom.uid import ImplicitVRLittleEndian
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm_multiprocessing import ConcurrentMapper

from ..datasets import AggregateInput, collate_fn, uncollate


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

    tsuid = dicom.file_meta.TransferSyntaxUID
    new_syntax = tsuid if not tsuid.is_compressed else ImplicitVRLittleEndian
    dicom = set_pixels(cast(FileDataset, dicom), img.cpu().numpy().astype(np.uint16), syntax=new_syntax)

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

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for batch in self.dataloader:
            batch = self.apply_transforms(batch)

            for example in uncollate(batch):
                # Update the DICOM object
                dicom = example["dicom"]
                img = example["img"]
                assert isinstance(dicom, Dicom)
                assert isinstance(img, Tensor)
                dicom = update_dicom(dicom, img)
                dicom["PixelData"].VR = "OB"
                dicom.PresentationIntentType = ALGORITHM_PRESENTATION_TYPE

                # Propagate to dict and yield
                example["dicom"] = dicom
                yield example

    def __call__(self, dest: Path) -> Set[Path]:
        if not dest.is_dir():
            raise NotADirectoryError(f"{dest} is not a directory")

        # Use multiple threads to speed up file output.
        # Input is already accelerated by DataLoader multi-processing.
        with ConcurrentMapper(threads=True, jobs=self.num_workers, ignore_exceptions=False) as mapper:
            mapper.create_bar(desc="Preprocessing", disable=not self.use_bar)
            result: Set[Path] = set(mapper(lambda x: PreprocessingPipeline._save_as(x, dest), iter(self)))
        return result

    def apply_transforms(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        for transform in self.transforms:
            inp = transform(inp)
        return inp

    @cached_property
    def dataloader(self) -> DataLoader:
        ds = AggregateInput(
            dicom_paths=self.dicom_paths,
            dicoms=self.dicoms,
            normalize=False,
            volume_handler=self.volume_handler,
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
    def _save_as(result: Dict[str, Any], dest: Path) -> Path:
        dicom = result["dicom"]
        assert isinstance(dicom, Dicom)

        # Save the DICOM
        dest_path = dest / "images" / f"{dicom.SOPInstanceUID}.dcm"
        dest_path.parent.mkdir(exist_ok=True)
        dicom.save_as(dest_path)

        # Pop big objects
        result.pop("dicom")
        result.pop("img")
        result.pop("record")

        # Ensure any tensors are converted to lists
        result = recursive_tensor_to_list(result)

        metadata_path = dest / "metadata" / f"{dicom.SOPInstanceUID}.json"
        metadata_path.parent.mkdir(exist_ok=True)
        with metadata_path.open("w") as f:
            json.dump(result, f)

        return dest_path
