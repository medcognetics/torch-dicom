#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Tuple, TypedDict, cast

import torch
import torch.nn.functional as F
from dicom_utils.volume import SliceAtLocation, VolumeHandler
from torch import Tensor
from torch.utils.data import IterableDataset

from .dicom import filter_collatable_types, slice_iterable_for_multiprocessing
from .path import PathDataset, PathInput


class TensorExample(TypedDict):
    img: Tensor
    img_size: Tensor
    path: Optional[Path]


class TensorInput(IterableDataset):
    r"""Dataset that iterates over Tensor objects and yields a metadata dictionary.

    Args:
        tensors: Iterable of Tensor objects.
        img_size: Size of the image to be returned. If None, the original image size is returned.
        transform: Optional transform to be applied to the image.
        skip_errors: If True, errors are ignored and the next Tensor is loaded. If False, the error is raised.
        volume_handler: Volume handler to be used to load the Tensor image.
    """

    def __init__(
        self,
        tensors: Iterable[Tensor],
        img_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        skip_errors: bool = False,
        volume_handler: VolumeHandler = SliceAtLocation(),
    ):
        self.tensors = tensors
        self.img_size = img_size
        self.transform = transform
        self.skip_errors = skip_errors
        self.volume_handler = volume_handler

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(img_size={self.img_size})"

    def __iter__(self) -> Iterator[TensorExample]:
        iterable = slice_iterable_for_multiprocessing(self.tensors)
        for t in iterable:
            try:
                yield self.load_example(t, self.img_size, self.transform)
            except Exception as ex:
                if not self.skip_errors:
                    raise
                else:
                    logging.warn("Encountered error while loading Tensor but skip_errors is True, skipping", ex)

    @classmethod
    def load_raw_example(
        cls,
        pixels: Tensor,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> TensorExample:
        r"""Loads an example, but does not perform any transforms.

        Args:
            pixels: Tensor object.
            img_size: Size of the image to be returned. If None, the original image size is returned.

        Returns:
            A TensorExample without transforms applied
        """
        if not isinstance(pixels, Tensor):
            raise TypeError(f"Expected tensor, got {type(pixels)}")

        img_size_tensor = torch.tensor(pixels.shape[-2:], dtype=torch.long)
        if img_size is not None:
            pixels = F.interpolate(pixels.unsqueeze_(0), img_size, mode="nearest").squeeze_(0)

        result = {
            "img": pixels,
            "img_size": img_size_tensor,
            "path": None,
        }
        return cast(TensorExample, result)

    @classmethod
    def load_example(
        cls,
        pixels: Tensor,
        img_size: Optional[Tuple[int, int]],
        transform: Optional[Callable] = None,
    ) -> TensorExample:
        r"""Loads a single Tensor example.

        Args:
            pixels: Tensor object.
            img_size: Size of the image to be returned. If None, the original image size is returned.
            transform: Optional transform to be applied to the image.

        Returns:
            A TensorExample
        """
        example = cls.load_raw_example(pixels, img_size)
        result = filter_collatable_types(example)

        if transform is not None:
            result = transform(result)

        return cast(TensorExample, result)


class TensorPathInput(TensorInput, PathInput):
    r"""Dataset that iterates over paths to Tensor files and yields a metadata dictionary.

    Args:
        paths: Iterable of paths to Tensor files.
        img_size: Size of the image to be returned. If None, the original image size is returned.
        transform: Optional transform to be applied to the image.
        skip_errors: If True, errors are ignored and the next Tensor is loaded. If False, the error is raised.
    """

    def __init__(
        self,
        paths: Iterable[Path],
        img_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        skip_errors: bool = False,
    ):
        self.tensors = paths
        self.img_size = img_size
        self.transform = transform
        self.skip_errors = skip_errors

    @classmethod
    def load_example(
        cls,
        path: Path,
        img_size: Optional[Tuple[int, int]],
        transform: Optional[Callable] = None,
    ) -> TensorExample:
        if not isinstance(path, Path):
            raise TypeError(f"Expected Path, got {type(path)}")
        pixels = torch.load(path)
        example = super().load_example(pixels, img_size, transform)
        example["path"] = path
        return cast(TensorExample, example)


class TensorPathDataset(PathDataset):
    r"""Dataset that reads Tensor files and returns a metadata dictionary. This dataset class scans over all input
    paths during instantiation. This takes time, but allows a dataset length to be determined.
    If you want to avoid this, use :class:`TensorPathInput` instead. This class is best suited for training.

    Args:
        paths: Iterable of paths to Tensor files.
        img_size: Size of the image to be returned. If None, the original image size is returned.
        transform: Optional transform to be applied to the image.
        skip_errors: If True, errors are ignored and the next Tensor is loaded. If False, the error is raised.
        manager: Optional multiprocessing manager to be used for sharing data between processes.
    """

    def __init__(
        self,
        paths: Iterator[Path],
        img_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        manager: Optional[SyncManager] = None,
    ):
        super().__init__(paths, manager=manager)
        self.img_size = img_size
        self.transform = transform

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)}, img_size={self.img_size})"

    def __getitem__(self, idx: int) -> TensorExample:
        if not 0 <= idx <= len(self):
            raise IndexError(f"Index {idx} is invalid for dataset length {len(self)}")
        path = self.files[idx]
        return TensorPathInput.load_example(path, self.img_size, self.transform)

    def __iter__(self) -> Iterator[TensorExample]:
        for path in self.files:
            yield TensorPathInput.load_example(path, self.img_size, self.transform)
