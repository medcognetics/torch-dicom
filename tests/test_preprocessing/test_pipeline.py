#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
from copy import deepcopy
from pathlib import Path

import pydicom
import pytest
import torch
from dicom_utils.dicom import read_dicom_image

from torch_dicom.preprocessing.pipeline import PreprocessingPipeline


class TestPreprocessingPipeline:
    @pytest.mark.parametrize("voi_lut", [False, True])
    def test_preprocess(self, tmp_path, dicoms, dicom_iterator, file_iterator, voi_lut):
        dicoms = deepcopy(dicoms)
        pipeline = PreprocessingPipeline(file_iterator, dicom_iterator)
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

            # The pipeline sets this tag to 'FOR ALGORITHM' but this breaks VOILUT application.
            # For now we just remove it.
            del actual_dcm.PresentationIntentType  # type: ignore

            for expected_dcm in dicoms:
                if expected_dcm.SOPInstanceUID == actual_dcm.SOPInstanceUID:
                    expected = torch.from_numpy(read_dicom_image(expected_dcm, voi_lut=voi_lut))
                    actual = torch.from_numpy(read_dicom_image(actual_dcm, voi_lut=voi_lut))
                    assert torch.allclose(expected, actual)
                    break
            else:
                raise AssertionError("No matching SOPInstanceUID found")

    def test_pixel_vr(self, tmp_path, dicoms, dicom_iterator, file_iterator):
        dicoms = deepcopy(dicoms)
        pipeline = PreprocessingPipeline(file_iterator, dicom_iterator)
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
