#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import cast

from torch.utils.data import DataLoader

from torch_dicom.datasets import DicomExample, TensorExample, collate_fn
from torch_dicom.datasets.chain import AggregateDataset, AggregateInput


class TestAggregateInput:
    def test_iter_dicoms(self, dicoms, file_iterator):
        ds = AggregateInput(
            dicom_paths=file_iterator,
            dicoms=iter(dicoms),
        )
        contents = list(ds)
        actual_dicoms = [cast(DicomExample, item)["dicom"] for item in contents]
        expected_dicoms = dicoms * 2
        assert len(actual_dicoms) == len(expected_dicoms)
        assert all(
            [
                dicom.SOPInstanceUID == expected_dicom.SOPInstanceUID
                for dicom, expected_dicom in zip(actual_dicoms, expected_dicoms)
            ]
        )

    def test_iter_tensors(self, tensors, tensor_files):
        ds = AggregateInput(
            tensors=iter(tensors),
            tensor_paths=iter(tensor_files),
        )
        contents = list(ds)
        actual = [cast(TensorExample, item)["img"] for item in contents]
        expected = tensors * 2
        assert len(actual) == len(expected)
        assert all([(act == exp).all() for act, exp in zip(actual, expected)])

    def test_dataloader(self, dicoms, file_iterator):
        ds = AggregateInput(
            dicom_paths=file_iterator,
            dicoms=iter(dicoms),
        )
        dl = DataLoader(ds, batch_size=1, collate_fn=collate_fn)
        expected_dicoms = dicoms * 2
        for i, batch in enumerate(dl):
            assert batch["dicom"][0].SOPInstanceUID == expected_dicoms[i].SOPInstanceUID


class TestAggregateDataset:
    def test_iter_dicoms(self, dicoms, file_iterator):
        ds = AggregateDataset(
            dicom_paths=file_iterator,
        )
        contents = list(ds)
        actual_dicoms = [cast(DicomExample, item)["dicom"] for item in contents]
        expected_dicoms = dicoms
        assert len(actual_dicoms) == len(expected_dicoms)
        assert all(
            [
                dicom.SOPInstanceUID == expected_dicom.SOPInstanceUID
                for dicom, expected_dicom in zip(actual_dicoms, expected_dicoms)
            ]
        )

    def test_iter_tensors(self, tensors, tensor_files):
        ds = AggregateDataset(
            tensor_paths=iter(tensor_files),
        )
        contents = list(ds)
        actual = [cast(TensorExample, item)["img"] for item in contents]
        expected = tensors
        assert len(actual) == len(expected)
        assert all([(act == exp).all() for act, exp in zip(actual, expected)])

    def test_dataloader(self, dicoms, file_iterator):
        ds = AggregateDataset(
            dicom_paths=file_iterator,
        )
        dl = DataLoader(ds, batch_size=1, collate_fn=collate_fn)
        expected_dicoms = dicoms
        for i, batch in enumerate(dl):
            assert batch["dicom"][0].SOPInstanceUID == expected_dicoms[i].SOPInstanceUID
