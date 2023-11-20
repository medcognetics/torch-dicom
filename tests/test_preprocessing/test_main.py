#!/usr/bin/env python
# -*- coding: utf-8 -*-

import runpy
import sys

import pytest

from torch_dicom.preprocessing import PreprocessingPipeline


@pytest.mark.usefixtures("file_iterator")
def test_main(tmp_path, files, mocker):
    dicom_dir = files[0].parent
    m = mocker.MagicMock(spec_set=PreprocessingPipeline)
    mocker.patch("torch_dicom.preprocessing.PreprocessingPipeline", new=m)
    src = dicom_dir
    dest = tmp_path / "dest"
    dest.mkdir()

    sys.argv = [
        sys.argv[0],
        str(src),
        str(dest),
        "-f",
        "dcm",
    ]

    try:
        runpy.run_module("torch_dicom.preprocessing", run_name="__main__", alter_sys=True)
    except SystemExit as e:
        raise e.__context__ if e.__context__ is not None else e
