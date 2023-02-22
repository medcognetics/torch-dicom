#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Iterable, Iterator, Union

from dicom_utils.dicom import Dicom
from torch import Tensor
from torch.utils.data import ChainDataset, ConcatDataset

from .dicom import DicomExample, DicomInput, DicomPathDataset, DicomPathInput
from .tensor import TensorExample, TensorInput, TensorPathDataset, TensorPathInput


class AggregateInput(ChainDataset):
    r"""Chain dataset that aggregates multiple input sources using the respective dataset types.

    Args:
        dicom_paths: Iterable of paths to DICOM files.
        dicoms: Iterable of DICOM objects.
        tensor_paths: Iterable of paths to tensor files.
        tensors: Iterable of tensors.
    """

    def __init__(
        self,
        dicom_paths: Iterable[Path] = [],
        dicoms: Iterable[Dicom] = [],
        tensor_paths: Iterable[Path] = [],
        tensors: Iterable[Tensor] = [],
        **kwargs,
    ):
        dicom_file_ds = DicomPathInput(dicom_paths, **kwargs)
        dicom_object_ds = DicomInput(dicoms, **kwargs)
        tensor_file_ds = TensorPathInput(tensor_paths, **kwargs)
        tensor_object_ds = TensorInput(tensors, **kwargs)
        super().__init__(
            [
                dicom_file_ds,
                dicom_object_ds,
                tensor_file_ds,
                tensor_object_ds,
            ]
        )

    def __iter__(self) -> Iterator[Union[DicomExample, TensorExample]]:
        return super().__iter__()


class AggregateDataset(ConcatDataset):
    r"""Concat dataset that aggregates multiple input sources using the respective dataset types.
    Use this for training or when a dataset size or getitem support is needed.

    Args:
        dicom_paths: Iterable of paths to DICOM files.
        tensor_paths: Iterable of paths to tensor files.
    """

    def __init__(
        self,
        dicom_paths: Iterable[Path] = [],
        tensor_paths: Iterable[Path] = [],
        **kwargs,
    ):
        dicom_file_ds = DicomPathDataset(iter(dicom_paths), **kwargs)
        tensor_file_ds = TensorPathDataset(iter(tensor_paths), **kwargs)
        super().__init__(
            [
                dicom_file_ds,
                tensor_file_ds,
            ]
        )

    def __getitem__(self, idx: int) -> Union[DicomExample, TensorExample]:
        return super().__getitem__(idx)

    def __iter__(self) -> Iterator[Union[DicomExample, TensorExample]]:
        for i in range(len(self)):
            yield self[i]
