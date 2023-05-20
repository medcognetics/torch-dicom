#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import warnings
from copy import copy, deepcopy
from dataclasses import is_dataclass, replace
from itertools import islice
from multiprocessing.managers import SyncManager
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
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from dicom_utils.container import DicomImageFileRecord, FileRecord, RecordCreator
from dicom_utils.dicom import Dicom, read_dicom_image
from dicom_utils.volume import ReduceVolume, VolumeHandler
from torch import Tensor
from torch.utils.data import IterableDataset, default_collate, get_worker_info
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from .path import PathDataset, PathInput


DUMMY_PATH: Final[Path] = Path("dummy.dcm")
D = TypeVar("D", bound=Union[Dict[str, Any], TypedDict])
LIST_COLLATE_TYPES: Final = (Path, FileRecord, Dicom)


class DicomExample(TypedDict):
    img: Tensor
    img_size: Tensor
    record: DicomImageFileRecord
    dicom: Dicom


def _collate_tensor_with_broadcast(batch: Sequence[Tensor], collate_fn_map: Dict) -> Tensor:
    batch = torch.broadcast_tensors(*batch)
    return cast(Tensor, collate(batch, collate_fn_map=default_collate_fn_map))


def collate_fn(
    batch: Sequence[D],
    default_fallback: bool = True,
    list_types: Tuple[Type, ...] = LIST_COLLATE_TYPES,
    dataclasses_as_lists: bool = True,
    broadcast_tensors: bool = False,
) -> D:
    r"""Collate function that supports Paths and FileRecords.
    All inputs should be dictionaries with the same keys. Any key that is a
    Path, DICOM, or FileRecord will be joined as a list. If all inputs are not dictionaries
    or there is an error, and default_fallback is True, the default collate function
    will be used.

    Args:
        batch: The batch of inputs to collate.
        default_fallback: If True, the default collate function will be used if
            there is an error or all inputs are not dictionaries.
        list_types: The custom types to collate into a list.
        dataclasses_as_lists: If True, dataclasses will be collated as lists.
        broadcast_tensors: If True, tensors will be broadcasted to the largest
            size in the batch.

    Returns:
        The collated batch.
    """
    collate_fn_map = copy(default_collate_fn_map)
    if broadcast_tensors:
        collate_fn_map[Tensor] = _collate_tensor_with_broadcast

    # use default collate unless all batch elements are dicts
    if not all(isinstance(b, dict) for b in batch):
        if default_fallback:
            return cast(D, collate(cast(List[Any], batch), collate_fn_map=collate_fn_map))
        else:
            raise TypeError("All inputs must be dictionaries.")

    # Copy because we will be mutating the batch
    batch = [copy(b) for b in batch]

    try:
        manual: Dict[Type, Dict[str, List[Any]]] = {k: {} for k in list_types}
        proto = batch[0]

        # ensure keys are copied since we will be mutating
        for key in set(proto.keys()):
            # apply to every element in the batch
            for elem in batch:
                assert isinstance(elem, dict)
                elem = cast(Dict[str, Any], elem)

                # read the value of the current key for this batch element
                if key not in elem:
                    raise KeyError(f"Key {key} not found in {elem}")
                value = elem[key]

                # check if the value needs manual collation.
                # if so, add it to the manual collation container and remove it from the batch element
                for dtype, container in manual.items():
                    if isinstance(value, dtype):
                        container.setdefault(key, []).append(value)
                        elem.pop(key)
                        break
                else:
                    if is_dataclass(value) and dataclasses_as_lists:
                        manual.setdefault(type(value), {}).setdefault(key, []).append(value)
                        elem.pop(key)

        # we should have removed all batch elem values that are being handled manually
        assert not any(isinstance(v, list_types) for elem in batch for v in elem.values())

        # call default collate and merge with the manually collated dtypes
        result = default_collate(cast(List[Any], batch))
        for v in manual.values():
            result.update(v)

        return result

    except Exception as e:
        if default_fallback:
            warnings.warn(f"Collating batch raised '{e}', falling back to default collate")
            return cast(D, collate(cast(List[Any], batch), collate_fn_map=collate_fn_map))
        else:
            raise e


def uncollate(batch: D) -> Iterator[D]:
    r"""Uncollates a batch dictionary into an iterator of example dictionaries.
    This is the inverse of :func:`collate_fn`. Non-sequence elements are repeated
    for each example in the batch. If examples in the batch have different
    sequence lengths, the iterator will be truncated to the shortest sequence.

    Args:
        batch: The batch dictionary to uncollate.

    Returns:
        An iterator of example dictionaries.
    """
    # separate out sequence-like elements and compute a batch size
    sequences = {k: v for k, v in batch.items() if isinstance(v, (Sequence, Tensor))}
    batch_size = min((len(v) for v in sequences.values()), default=0)

    # repeat non-sequence elements
    non_sequences = {k: [v] * batch_size for k, v in batch.items() if not isinstance(v, (Sequence, Tensor))}

    for idx in range(batch_size):
        result = {k: v[idx] for container in (sequences, non_sequences) for k, v in container.items()}
        yield cast(D, result)


def filter_collatable_types(example: D) -> D:
    r"""Filters out non-collatable types from a dictionary."""
    result = {k: v for k, v in example.items() if isinstance(v, (Tensor, list, str, *LIST_COLLATE_TYPES))}
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
        normalize: If True, the image is normalized to [0, 1].
        voi_lut: If True, the VOI LUT is applied to the image.
        inversion: If True, apply PhotometricInterpretation inversion.

    Shapes:
        - ``'img'``: :math:`(C, H, W)` for 2D images, :math:`(C, D, H, W)` for 3D volumes.
    """

    def __init__(
        self,
        dicoms: Iterable[Dicom],
        img_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        skip_errors: bool = False,
        volume_handler: VolumeHandler = ReduceVolume(),
        normalize: bool = True,
        voi_lut: bool = True,
        inversion: bool = True,
    ):
        self.dicoms = dicoms
        self.img_size = img_size
        self.transform = transform
        self.skip_errors = skip_errors
        self.volume_handler = volume_handler
        self.normalize = normalize
        self.voi_lut = voi_lut
        self.inversion = inversion

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(img_size={self.img_size})"

    def __iter__(self) -> Iterator[DicomExample]:
        iterable = slice_iterable_for_multiprocessing(self.dicoms)
        for dcm in iterable:
            try:
                yield self.load_example(
                    dcm,
                    self.img_size,
                    self.transform,
                    self.volume_handler,
                    self.normalize,
                    self.voi_lut,
                    self.inversion,
                )
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
        volume_handler: VolumeHandler = ReduceVolume(),
        normalize: bool = True,
        voi_lut: bool = True,
        inversion: bool = True,
    ) -> DicomExample:
        r"""Loads a single DICOM example.

        Args:
            dcm: DICOM object.
            img_size: Size of the image to be returned. If None, the original image size is returned.
            transform: Optional transform to be applied to the image.
            volume_handler: Volume handler to be used to load the DICOM image.
            normalize: If True, the image is normalized to [0, 1].
            voi_lut: If True, the VOI LUT is applied to the image.
            inversion: If True, apply PhotometricInterpretation inversion.

        Returns:
            A DicomExample
        """
        example = DicomInput.load_raw_example(
            dcm, img_size, volume_handler, normalize, voi_lut=voi_lut, inversion=inversion
        )
        result = filter_collatable_types(example)

        if transform is not None:
            result = transform(result)

        return cast(DicomExample, result)

    @classmethod
    def load_pixels(
        cls,
        dcm: Dicom,
        volume_handler: VolumeHandler = ReduceVolume(),
        normalize: bool = True,
        voi_lut: bool = True,
        inversion: bool = True,
    ) -> Tensor:
        pixels = torch.from_numpy(
            read_dicom_image(dcm, volume_handler=volume_handler, voi_lut=voi_lut, inversion=inversion).astype(np.int32)
        )
        if normalize:
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
        volume_handler: VolumeHandler = ReduceVolume(),
        normalize: bool = True,
        resize_mode: str = "bilinear",
        voi_lut: bool = True,
        inversion: bool = True,
    ) -> DicomExample:
        r"""Loads an example, but does not perform any transforms.

        Args:
            dcm: DICOM object.
            img_size: Size of the image to be returned. If None, the original image size is returned.
            volume_handler: Volume handler to be used to load the DICOM image.
            normalize: If True, the image is normalized to [0, 1].
            resize_mode: Interpolation mode to use when resizing the image.
            voi_lut: If True, the VOI LUT is applied to the image.
            inversion: If True, apply PhotometricInterpretation inversion.

        Returns:
            A DicomExample without transforms applied
        """
        if not isinstance(dcm, Dicom):
            raise TypeError(f"Expected Dicom object, got {type(dcm)}")

        pixels = cls.load_pixels(dcm, volume_handler, normalize, voi_lut, inversion)

        img_size_tensor = torch.tensor(pixels.shape[-2:], dtype=torch.long)
        if img_size is not None:
            H, W = pixels.shape[-2:]
            Ht, Wt = img_size
            if H != Ht or W != Wt:
                # Ensure that the pixel type is preserved
                pixel_type = pixels.dtype
                pixels = pixels if pixels.is_floating_point() else pixels.float()

                # Pixel shape may be (C H W) or (C D H W) for volumes. Ensure we have 4 dims for interpolate
                is_volume = pixels.ndim == 4
                pixels = pixels if is_volume else pixels.unsqueeze_(0)

                # Run resize
                pixels = F.interpolate(pixels, img_size, mode=resize_mode)

                # Restore shape and dtype
                pixels = pixels if is_volume else pixels.squeeze_(0)
                pixels = pixels.to(dtype=pixel_type)

        creator = RecordCreator()
        rec = creator(DUMMY_PATH, dcm)

        # Ensure that the path is set to DUMMY_PATH
        rec = replace(rec, path=DUMMY_PATH)

        # Copy the dicom object to avoid modifying the original and remove the pixel data.
        # We first do a shallow copy to remove pixel data followed by a deep copy.
        dcm = dcm.copy()
        delattr(dcm, "PixelData")
        delattr(dcm, "_pixel_array")
        dcm = deepcopy(dcm)

        result = {
            "img": pixels,
            "img_size": img_size_tensor,
            "record": rec,
            "dicom": dcm,
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
        normalize: If True, the image is normalized to [0, 1].
        voi_lut: If True, the VOI LUT is applied to the image.
        inversion: If True, apply PhotometricInterpretation inversion.

    Shapes:
        - ``'img'``: :math:`(C, H, W)` for 2D images, :math:`(C, D, H, W)` for 3D volumes.
    """

    def __init__(
        self,
        paths: Iterable[Path],
        img_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        skip_errors: bool = False,
        volume_handler: VolumeHandler = ReduceVolume(),
        normalize: bool = True,
        voi_lut: bool = True,
        inversion: bool = True,
    ):
        self.dicoms = paths
        self.img_size = img_size
        self.transform = transform
        self.skip_errors = skip_errors
        self.volume_handler = volume_handler
        self.normalize = normalize
        self.voi_lut = voi_lut
        self.inversion = inversion

    @classmethod
    def load_example(
        cls,
        path: Path,
        img_size: Optional[Tuple[int, int]],
        transform: Optional[Callable] = None,
        volume_handler: VolumeHandler = ReduceVolume(),
        normalize: bool = True,
        voi_lut: bool = True,
        inversion: bool = True,
    ) -> DicomExample:
        with pydicom.dcmread(path) as dcm:
            example = super().load_example(dcm, img_size, transform, volume_handler, normalize, voi_lut, inversion)
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
        normalize: If True, the image is normalized to [0, 1].
        voi_lut: If True, the VOI LUT is applied to the image.
        inversion: If True, apply PhotometricInterpretation inversion.
        manager: Optional multiprocessing manager to be used for sharing data between processes.

    Shapes:
        - ``'img'``: :math:`(C, H, W)` for 2D images, :math:`(C, D, H, W)` for 3D volumes.
    """

    def __init__(
        self,
        paths: Iterator[Path],
        img_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        volume_handler: VolumeHandler = ReduceVolume(),
        normalize: bool = True,
        voi_lut: bool = True,
        inversion: bool = True,
        manager: Optional[SyncManager] = None,
    ):
        super().__init__(paths, manager=manager)
        self.img_size = img_size
        self.transform = transform
        self.volume_handler = volume_handler
        self.normalize = normalize
        self.voi_lut = voi_lut
        self.inversion = inversion

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)}, img_size={self.img_size})"

    def __getitem__(self, idx: int) -> DicomExample:
        if not 0 <= idx <= len(self):
            raise IndexError(f"Index {idx} is invalid for dataset length {len(self)}")
        path = self.files[idx]
        return self.load_example(
            path, self.img_size, self.transform, self.volume_handler, self.normalize, self.voi_lut, self.inversion
        )

    def __iter__(self) -> Iterator[DicomExample]:
        for path in self.files:
            yield self.load_example(
                path,
                self.img_size,
                self.transform,
                self.volume_handler,
                self.normalize,
                self.voi_lut,
                self.inversion,
            )

    @classmethod
    def load_example(
        cls,
        path: Path,
        img_size: Optional[Tuple[int, int]],
        transform: Optional[Callable] = None,
        volume_handler: VolumeHandler = ReduceVolume(),
        normalize: bool = True,
        voi_lut: bool = True,
        inversion: bool = True,
    ) -> DicomExample:
        return DicomPathInput.load_example(path, img_size, transform, volume_handler, normalize, voi_lut, inversion)
