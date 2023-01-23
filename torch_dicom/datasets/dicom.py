#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import warnings
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from dicom_utils.container import DicomImageFileRecord, FileRecord
from dicom_utils.dicom import Dicom, read_dicom_image
from dicom_utils.volume import SliceAtLocation, VolumeHandler
from torch import Tensor
from torch.utils.data import IterableDataset, default_collate, get_worker_info

from .path import PathDataset, PathInput


DUMMY_PATH: Final[Path] = Path("dummy.dcm")
D = TypeVar("D", bound=Union[Dict[str, Any], TypedDict])


class DicomExample(TypedDict):
    img: Tensor
    img_size: Tensor
    record: DicomImageFileRecord


def collate_fn(batch: Sequence[D], default_fallback: bool = True) -> D:
    r"""Collate function that supports Paths and FileRecords.
    All inputs should be dictionaries with the same keys. Any key that is a
    Path or FileRecord will be joined as a list. If all inputs are not dictionaries
    or there is an error, and default_fallback is True, the default collate function
    will be used.

    Args:
        batch: The batch of inputs to collate.
        default_fallback: If True, the default collate function will be used if
            there is an error or all inputs are not dictionaries.

    Returns:
        The collated batch.
    """
    # use default collate unless all batch elements are dicts
    if not all(isinstance(b, dict) for b in batch):
        if default_fallback:
            return default_collate(cast(List[Any], batch))
        else:
            raise TypeError("All inputs must be dictionaries.")

    try:
        # manually collate Paths and FileRecords
        paths: Dict[str, List[Path]] = {}
        records: Dict[str, List[FileRecord]] = {}
        proto = batch[0]
        # ensure keys are copied since we will be mutating
        for key in set(proto.keys()):
            for elem in batch:
                assert isinstance(elem, dict)
                elem = cast(Dict[str, Any], elem)
                if key not in elem:
                    raise KeyError(f"Key {key} not found in {elem}")
                value = elem[key]
                if isinstance(value, Path):
                    paths.setdefault(key, []).append(value)
                    elem.pop(key)
                elif isinstance(value, FileRecord):
                    records.setdefault(key, []).append(value)
                    elem.pop(key)

        # we should have removed all batch elem values that are Paths or FileRecords
        assert not any(isinstance(v, Path) for elem in batch for v in elem.values())
        assert not any(isinstance(v, FileRecord) for elem in batch for v in elem.values())

        # call default collate and merge with the manually collated Paths and FileRecords
        result = default_collate(cast(List[Any], batch))
        result.update(records)
        result.update(paths)

        return result

    except Exception as e:
        if default_fallback:
            warnings.warn(f"Collating batch raised {e}, falling back to default collate")
            return default_collate(cast(List[Any], batch))
        else:
            raise e


def filter_collatable_types(example: D) -> D:
    r"""Filters out non-collatable types from a dictionary."""
    result = {k: v for k, v in example.items() if isinstance(v, (Tensor, list, str, Path, FileRecord))}
    return cast(D, result)


def slice_iterable_for_multiprocessing(iterable: Iterable[Any]) -> Iterable[Any]:
    r"""Slices an iterable based on the current worker index and number of workers.
    Use this to propertly slice an iterable when using multiprocessing in a dataloader.

    Args:
        iterable: The iterable to slice.

    Returns:
        The sliced iterable for the current worker.
    """
    worker_info = get_worker_info()
    if worker_info is not None:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        return islice(iterable, worker_id, None, num_workers)
    else:
        return iterable


class DicomInput(IterableDataset):
    r"""Dataset that iterates over DICOM objects and yields a metadata dictionary.

    Args:
        dicoms: Iterable of DICOM objects.
        img_size: Size of the image to be returned. If None, the original image size is returned.
        transform: Optional transform to be applied to the image.
        skip_errors: If True, errors are ignored and the next DICOM is loaded. If False, the error is raised.
        volume_handler: Volume handler to be used to load the DICOM image.
    """

    def __init__(
        self,
        dicoms: Iterable[Dicom],
        img_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        skip_errors: bool = False,
        volume_handler: VolumeHandler = SliceAtLocation(),
    ):
        self.dicoms = dicoms
        self.img_size = img_size
        self.transform = transform
        self.skip_errors = skip_errors
        self.volume_handler = volume_handler

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(img_size={self.img_size})"

    def __iter__(self) -> Iterator[DicomExample]:
        iterable = slice_iterable_for_multiprocessing(self.dicoms)
        for dcm in iterable:
            try:
                yield self.load_example(dcm, self.img_size, self.transform, self.volume_handler)
            except Exception as ex:
                if not self.skip_errors:
                    raise
                else:
                    logging.warn("Encountered error while loading DICOM but skip_errors is True, skipping", ex)

    @classmethod
    def load_example(
        cls,
        dcm: Dicom,
        img_size: Optional[Tuple[int, int]],
        transform: Optional[Callable] = None,
        volume_handler: VolumeHandler = SliceAtLocation(),
    ) -> DicomExample:
        r"""Loads a single DICOM example.

        Args:
            dcm: DICOM object.
            img_size: Size of the image to be returned. If None, the original image size is returned.
            transform: Optional transform to be applied to the image.
            volume_handler: Volume handler to be used to load the DICOM image.

        Returns:
            A DicomExample
        """
        example = DicomInput.load_raw_example(dcm, img_size, volume_handler)
        result = filter_collatable_types(example)

        if transform is not None:
            result = transform(result)

        return cast(DicomExample, result)

    @classmethod
    def load_pixels(cls, dcm: Dicom, volume_handler: VolumeHandler = SliceAtLocation()) -> Tensor:
        pixels = torch.from_numpy(read_dicom_image(dcm, volume_handler=volume_handler).astype(np.int64))
        pixels = cls.normalize_pixels(pixels)
        return pixels

    @classmethod
    def normalize_pixels(cls, pixels: Tensor, eps: float = 1e-6) -> Tensor:
        pmin, pmax = pixels.aminmax()
        delta = (pmax - pmin).clip(min=eps)
        pixels = (pixels.float() - pmin).div_(delta)
        return pixels

    @classmethod
    def load_raw_example(
        cls,
        dcm: Dicom,
        img_size: Optional[Tuple[int, int]] = None,
        volume_handler: VolumeHandler = SliceAtLocation(),
    ) -> DicomExample:
        r"""Loads an example, but does not perform any transforms.

        Args:
            dcm: DICOM object.
            img_size: Size of the image to be returned. If None, the original image size is returned.
            volume_handler: Volume handler to be used to load the DICOM image.

        Returns:
            A DicomExample without transforms applied
        """
        if not isinstance(dcm, Dicom):
            raise TypeError(f"Expected Dicom object, got {type(dcm)}")

        pixels = cls.load_pixels(dcm, volume_handler)

        img_size_tensor = torch.tensor(pixels.shape[-2:], dtype=torch.long)
        if img_size is not None:
            pixels = F.interpolate(pixels.unsqueeze_(0), img_size, mode="nearest").squeeze_(0)

        rec = DicomImageFileRecord.from_dicom(DUMMY_PATH, dcm)
        # from_dicom will make DUMMY_PATH absolute, but we want it relative
        rec = replace(rec, path=DUMMY_PATH)
        result = {
            "img": pixels,
            "img_size": img_size_tensor,
            "record": rec,
        }
        return cast(DicomExample, result)


class DicomPathInput(DicomInput, PathInput):
    r"""Dataset that iterates over paths to DICOM files and yields a metadata dictionary.

    Args:
        paths: Iterable of paths to DICOM files.
        img_size: Size of the image to be returned. If None, the original image size is returned.
        transform: Optional transform to be applied to the image.
        skip_errors: If True, errors are ignored and the next DICOM is loaded. If False, the error is raised.
        volume_handler: Volume handler to be used to load the DICOM image.
    """

    def __init__(
        self,
        paths: Iterable[Path],
        img_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        skip_errors: bool = False,
        volume_handler: VolumeHandler = SliceAtLocation(),
    ):
        self.dicoms = paths
        self.img_size = img_size
        self.transform = transform
        self.skip_errors = skip_errors
        self.volume_handler = volume_handler

    @classmethod
    def load_example(
        cls,
        path: Path,
        img_size: Optional[Tuple[int, int]],
        transform: Optional[Callable] = None,
        volume_handler: VolumeHandler = SliceAtLocation(),
    ) -> DicomExample:
        with pydicom.dcmread(path) as dcm:
            example = super().load_example(dcm, img_size, transform, volume_handler)
        example["record"] = replace(example["record"], path=path)
        return cast(DicomExample, example)


class DicomPathDataset(PathDataset):
    r"""Dataset that reads DICOM files and returns a metadata dictionary. This dataset class scans over all input
    paths during instantiation. This takes time, but allows a dataset length to be determined.
    If you want to avoid this, use :class:`DicomPathInput` instead. This class is best suited for training.

    Args:
        paths: Iterable of paths to DICOM files.
        img_size: Size of the image to be returned. If None, the original image size is returned.
        transform: Optional transform to be applied to the image.
        skip_errors: If True, errors are ignored and the next DICOM is loaded. If False, the error is raised.
        volume_handler: Volume handler to be used to load the DICOM image.
    """

    def __init__(
        self,
        paths: Iterator[Path],
        img_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        volume_handler: VolumeHandler = SliceAtLocation(),
    ):
        super().__init__(paths)
        self.img_size = img_size
        self.transform = transform
        self.volume_handler = volume_handler

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)}, img_size={self.img_size})"

    def __getitem__(self, idx: int) -> DicomExample:
        if not 0 <= idx <= len(self):
            raise IndexError(f"Index {idx} is invalid for dataset length {len(self)}")
        path = self.files[idx]
        return DicomPathInput.load_example(path, self.img_size, self.transform, self.volume_handler)

    def __iter__(self) -> Iterator[DicomExample]:
        for path in self.files:
            yield DicomPathInput.load_example(path, self.img_size, self.transform, self.volume_handler)
