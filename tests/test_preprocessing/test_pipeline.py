#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

import pydicom
import torch
from dicom_utils.dicom import read_dicom_image

from torch_dicom.preprocessing.pipeline import PreprocessingPipeline


class TestPreprocessingPipeline:
    def test_preprocess(self, tmp_path, dicoms, dicom_iterator, file_iterator):
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
            for expected_dcm in dicoms:
                if expected_dcm.SOPInstanceUID == actual_dcm.SOPInstanceUID:
                    expected = torch.from_numpy(read_dicom_image(expected_dcm))
                    actual = torch.from_numpy(read_dicom_image(pydicom.dcmread(path)))
                    assert torch.allclose(expected, actual)
                    break
            else:
                raise AssertionError("No matching SOPInstanceUID found")
