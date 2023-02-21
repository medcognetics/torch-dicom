#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

import pytest
import torch
from dicom_utils.dicom_factory import CompleteMammographyStudyFactory


@pytest.fixture
def dicoms():
    fact = CompleteMammographyStudyFactory(Rows=2048, Columns=1536, seed=42)
    return fact()


@pytest.fixture
def dicom_iterator(dicoms):
    return iter(dicoms)


@pytest.fixture
def file_iterator(tmp_path, dicoms):
    files = []
    for i, dcm in enumerate(dicoms):
        dest = Path(tmp_path, f"file_{i}.dcm")
        dcm.save_as(dest)
        files.append(dest)
    return iter(files)


@pytest.fixture
def file_list(tmp_path, file_iterator):
    file_list = Path(tmp_path, "file_list.txt")
    with open(file_list, "w") as f:
        for path in file_iterator:
            f.write(f"{path}\n")
    return file_list


@pytest.fixture
def tensors():
    torch.random.manual_seed(0)
    return [torch.rand(1, 2048, 1536) for _ in range(12)]


@pytest.fixture
def tensor_input(tensors):
    return iter(tensors)


@pytest.fixture
def tensor_files(tmp_path, tensors):
    paths = []
    for i, t in enumerate(tensors):
        path = tmp_path / f"tensor_{i}.pt"
        torch.save(t, path)
        paths.append(path)
    return iter(paths)
