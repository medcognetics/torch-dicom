#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .crop import MinMaxCrop, ROICrop, TileCrop
from .pipeline import PreprocessingPipeline
from .resize import Resize


def datamodule_available() -> bool:
    try:
        pass
    except ImportError:
        return False
    return True


def require_datamodule() -> None:
    if not datamodule_available():
        raise ImportError(
            "PreprocessedPNGDataModule is not available." "Please install the `torch-dicom[datamodule]` extra."
        )


__all__ = [
    "MinMaxCrop",
    "PreprocessingPipeline",
    "Resize",
    "ROICrop",
    "TileCrop",
    "datamodule_available",
    "require_datamodule",
]
