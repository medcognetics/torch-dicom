#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Iterable, Iterator, Set, Tuple, cast

import numpy as np
import torch
from dicom_utils.container import DicomImageFileRecord
from dicom_utils.dicom import ALGORITHM_PRESENTATION_TYPE, Dicom, set_pixels
from dicom_utils.volume import KeepVolume, VolumeHandler
from pydicom.dataset import FileDataset
from pydicom.uid import ImplicitVRLittleEndian
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm_multiprocessing import ConcurrentMapper

from ..datasets import AggregateInput, DicomExample, collate_fn


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

    batch_size: int = 1
    num_workers: int = 1
    prefetch_factor: int = 4
    device: torch.device = torch.device("cpu")
    volume_handler: VolumeHandler = KeepVolume()
    dataloader_kwargs: dict = field(default_factory=dict)
    use_bar: bool = True

    def __iter__(self) -> Iterator[Tuple[DicomImageFileRecord, Dicom]]:
        for batch in self.dataloader:
            batch: DicomExample
            for img, dicom, record in zip(batch["img"], batch["dicom"], batch["record"]):
                assert isinstance(record, DicomImageFileRecord)
                assert isinstance(dicom, Dicom)
                img, dicom = self.apply_transforms(img, dicom)
                dicom = update_dicom(dicom, img)
                dicom["PixelData"].VR = "OB"
                dicom.PresentationIntentType = ALGORITHM_PRESENTATION_TYPE
                yield record, dicom

    def __call__(self, dest: Path) -> Set[Path]:
        if not dest.is_dir():
            raise NotADirectoryError(f"{dest} is not a directory")

        # Use multiple threads to speed up file output.
        # Input is already accelerated by DataLoader multi-processing.
        with ConcurrentMapper(threads=True, jobs=self.num_workers, ignore_exceptions=False) as mapper:
            mapper.create_bar(desc="Preprocessing", disable=not self.use_bar)
            result: Set[Path] = set(mapper(lambda x: PreprocessingPipeline._save_as(x[1], dest), iter(self)))
        return result

    def apply_transforms(self, img: Tensor, dicom: Dicom) -> Tuple[Tensor, Dicom]:
        # FIXME
        return img, dicom

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
    def _save_as(dicom: Dicom, dest: Path) -> Path:
        dest_path = dest / f"{dicom.SOPInstanceUID}.dcm"
        dicom.save_as(dest_path)
        return dest_path
