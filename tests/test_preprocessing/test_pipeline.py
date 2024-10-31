#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy
from pathlib import Path

import pytest
from dicom_utils.volume import ReduceVolume
from torch import Tensor

from torch_dicom.datasets import ImagePathDataset
from torch_dicom.preprocessing.pipeline import OutputFormat, PreprocessingPipeline


class TestPreprocessingPipeline:

    @pytest.mark.parametrize(
        "output_format, compression",
        [
            (OutputFormat.PNG, None),
            (OutputFormat.TIFF, None),
            (OutputFormat.TIFF, "tiff_packbits"),
            (OutputFormat.TIFF, "tiff_lzw"),
            (OutputFormat.TIFF, "tiff_deflate"),
            (OutputFormat.TIFF, "tiff_adobe_deflate"),
        ],
    )
    def test_preprocess_image(self, tmp_path, dicoms, dicom_iterator, file_iterator, output_format, compression):
        dicoms = deepcopy(dicoms)
        pipeline = PreprocessingPipeline(
            file_iterator,
            dicom_iterator,
            output_format=output_format,
            volume_handler=ReduceVolume(),
            compression=compression,
            num_workers=2,
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
