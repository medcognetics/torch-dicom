#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pytest
import torch
from dicom_utils.dicom import set_pixels
from dicom_utils.dicom_factory import CompleteMammographyStudyFactory
from PIL import Image
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, RLELossless


@pytest.fixture(scope="session")
def dicom_size():
    return (64, 32)


@pytest.fixture(scope="session")
def dicoms(dicom_size):
    H, W = dicom_size
    fact = CompleteMammographyStudyFactory(
        Rows=H,
        Columns=W,
        seed=42,
        BitsAllocated=16,
        WindowCenter=4096,
        WindowWidth=8192,
    )
    dicoms = fact()

    # Ensure we have a mix of inversions
    dicoms[0].PhotometricInterpretation = "MONOCHROME1"
    dicoms[1].PhotometricInterpretation = "MONOCHROME2"
    dicoms[0].LUTDescriptor = b"LUT Descriptor"

    # Ensure we have a mix of TSUIDs
    tsuids = (ImplicitVRLittleEndian, ExplicitVRLittleEndian, RLELossless)
    for dcm, tsuid in zip(dicoms[: len(tsuids)], tsuids):
        set_pixels(dcm, dcm.pixel_array, tsuid)
        dcm.file_meta.TransferSyntaxUID = tsuid

    return dicoms


@pytest.fixture
def dicom_iterator(dicoms):
    return iter(dicoms)


@pytest.fixture(scope="session")
def files(tmpdir_factory, dicoms):
    tmp_path = tmpdir_factory.mktemp("dicoms")
    files = []
    for i, dcm in enumerate(dicoms):
        dest = Path(tmp_path, f"file_{i}.dcm")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dcm.save_as(dest)
        files.append(dest)
    return files


@pytest.fixture
def file_iterator(files):
    return iter(files)


@pytest.fixture(scope="session")
def file_list(tmpdir_factory, file_iterator):
    tmp_path = tmpdir_factory.mktemp("data")
    file_list = Path(tmp_path, "file_list.txt")
    with open(file_list, "w") as f:
        for path in file_iterator:
            f.write(f"{path}\n")
    return file_list


@pytest.fixture(scope="session")
def tensors(dicom_size):
    H, W = dicom_size
    torch.random.manual_seed(0)
    return [torch.rand(1, H, W) for _ in range(12)]


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


@pytest.fixture(scope="session")
def images(tensors):
    return [Image.fromarray((t * np.iinfo(np.uint16).max).squeeze().numpy().astype(np.uint16)) for t in tensors]


@pytest.fixture
def image_input(images):
    return iter(images)


@pytest.fixture(params=["png", "tiff"])
def image_files(tmpdir_factory, images, request):
    tmp_path = tmpdir_factory.mktemp("img_data")
    paths = []
    for i, img in enumerate(images):
        path = Path(tmp_path, f"image_{i}.{request.param}")
        img.save(path, format=request.param.upper())
        paths.append(path)
    return iter(paths)
