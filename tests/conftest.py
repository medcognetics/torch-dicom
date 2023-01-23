#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

import pytest
from dicom_utils.dicom_factory import CompleteMammographyStudyFactory


@pytest.fixture
def dicom_iterator():
    fact = CompleteMammographyStudyFactory(Rows=2048, Columns=1536, seed=42)
    return iter(fact())


@pytest.fixture
def file_iterator(tmp_path, dicom_iterator):
    files = []
    for i, dcm in enumerate(dicom_iterator):
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
