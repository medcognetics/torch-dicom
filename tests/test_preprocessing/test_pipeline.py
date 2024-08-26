#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
from copy import deepcopy
from pathlib import Path

import pydicom
import pytest
import torch
from dicom_utils.dicom import read_dicom_image
from dicom_utils.volume import ReduceVolume
from torch import Tensor

from torch_dicom.datasets import ImagePathDataset
from torch_dicom.preprocessing.pipeline import OutputFormat, PreprocessingPipeline


class TestPreprocessingPipeline:
    @pytest.mark.parametrize("voi_lut", [False, True])
    @pytest.mark.parametrize("inversion", [False, True])
    @pytest.mark.parametrize("rescale", [False, True])
    def test_preprocess_dicom(self, tmp_path, dicoms, dicom_iterator, file_iterator, voi_lut, inversion, rescale):
        dicoms = deepcopy(dicoms)
        pipeline = PreprocessingPipeline(file_iterator, dicom_iterator, output_format=OutputFormat.DICOM)
        dest = Path(tmp_path, "output")
        dest.mkdir()

        output_files = pipeline(dest)

        # Check that we wrote the config file
        assert (dest / "preprocess_config.json").is_file()

        # We expected 2x the number of files because we are passing file and dicom iterators.
        # However, outputs are named by SOPInstanceUID, so we should only get 1x the number of files.
        assert len(output_files) == len(dicoms)

        for path in output_files:
            assert path.exists()
            actual_dcm = pydicom.dcmread(path)
            for expected_dcm in dicoms:
                if expected_dcm.SOPInstanceUID == actual_dcm.SOPInstanceUID:
                    expected = torch.from_numpy(
                        read_dicom_image(expected_dcm, voi_lut=voi_lut, inversion=inversion, rescale=rescale)
                    )
                    actual = torch.from_numpy(
                        read_dicom_image(actual_dcm, voi_lut=voi_lut, inversion=inversion, rescale=rescale)
                    )
                    assert torch.allclose(expected, actual)
                    break
            else:
                raise AssertionError("No matching SOPInstanceUID found")

    @pytest.mark.parametrize("output_format", [OutputFormat.PNG, OutputFormat.TIFF])
    def test_preprocess_image(self, tmp_path, dicoms, dicom_iterator, file_iterator, output_format):
        dicoms = deepcopy(dicoms)
        pipeline = PreprocessingPipeline(
            file_iterator, dicom_iterator, output_format=output_format, volume_handler=ReduceVolume()
        )
        dest = Path(tmp_path, "output")
        dest.mkdir()

        output_files = pipeline(dest)

        # Check that we wrote the config file
        assert (dest / "preprocess_config.json").is_file()

        # We expected 2x the number of files because we are passing file and dicom iterators.
        # However, outputs are named by SOPInstanceUID, so we should only get 1x the number of files.
        assert len(output_files) == len(dicoms)

        dataset = ImagePathDataset(iter(output_files))
        assert len(dataset) == len(output_files)
        for example in dataset:
            assert isinstance(example["img"], Tensor)

    def test_pixel_vr(self, tmp_path, dicoms, dicom_iterator, file_iterator):
        dicoms = deepcopy(dicoms)
        pipeline = PreprocessingPipeline(file_iterator, dicom_iterator, output_format=OutputFormat.DICOM)
        dest = Path(tmp_path, "output")
        dest.mkdir()

        output_files = pipeline(dest)
        assert output_files

        for path in output_files:
            assert path.exists()
            with warnings.catch_warnings(record=True) as warning_list:
                pydicom.dcmread(path)

                # Check if any warning was raised
                if warning_list:
                    pytest.fail(str(warning_list[0].message))
