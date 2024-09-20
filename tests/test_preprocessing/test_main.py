#!/usr/bin/env python
# -*- coding: utf-8 -*-

import runpy
import sys
import warnings

import pytest

from torch_dicom.preprocessing import PreprocessingPipeline
from torch_dicom.preprocessing.__main__ import entrypoint


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


@pytest.mark.parametrize(
    "pattern",
    [
        "ERROR: Invalid value for VR: ",
        "The Bits Stored Value does not match",
    ],
)
def test_main_with_warnings(mocker, pattern, capsys):
    def side_effect_function(_):
        warnings.warn(pattern)
        print("Warning issued")

    mocker.patch("torch_dicom.preprocessing.__main__.main", side_effect=side_effect_function)
    mocker.patch("torch_dicom.preprocessing.__main__.parse_args")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        entrypoint()

    captured = capsys.readouterr()
    assert not any(pattern in str(warning.message) for warning in w)
    assert pattern not in captured.out
    assert pattern not in captured.err
