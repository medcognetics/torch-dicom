#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pytest
import torch
from dicom_utils.dicom import set_pixels
from dicom_utils.dicom_factory import CompleteMammographyStudyFactory
from dicom_utils.tags import Tag
from PIL import Image
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, RLELossless


@pytest.fixture(scope="session")
def dicoms():
    fact = CompleteMammographyStudyFactory(Rows=2048, Columns=1536, seed=42, WindowCenter=5, WindowWidth=4)
    dicoms = fact()

    # Ensure we have a mix of inversions
    dicoms[0].PhotometricInterpretation = "MONOCHROME1"
    dicoms[1].PhotometricInterpretation = "MONOCHROME2"

    # Ensure we have a mix of TSUIDs
    tsuids = (ImplicitVRLittleEndian, ExplicitVRLittleEndian, RLELossless)
    for dcm, tsuid in zip(dicoms[: len(tsuids)], tsuids):
        set_pixels(dcm, dcm.pixel_array, tsuid)
        dcm.file_meta.TransferSyntaxUID = tsuid

    for dcm in dicoms:
        dcm[Tag.PixelData].VR = "OW"
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


@pytest.fixture(scope="session")
def images(tensors):
    return [Image.fromarray((t * np.iinfo(np.uint16).max).squeeze().numpy().astype(np.uint16)) for t in tensors]


@pytest.fixture
def image_input(images):
    return iter(images)


@pytest.fixture(scope="session")
def image_files(tmpdir_factory, images):
    tmp_path = tmpdir_factory.mktemp("img_data")
    paths = [Path(tmp_path / f"image_{i}.png") for i, img in enumerate(images)]
    [img.save(path) for img, path in zip(images, paths)]
    return paths
