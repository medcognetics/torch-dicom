#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import cast

import numpy as np
import torch
from dicom_utils.container.collection import iterate_input_path
from dicom_utils.dicom import nvjpeg2k_is_available
from dicom_utils.volume import KeepVolume, ReduceVolume, SliceAtLocation, VolumeHandler
from registry import Registry

from .crop import MinMaxCrop
from .pipeline import OUTPUT_FORMATS, PreprocessingPipeline
from .resize import Resize


VOLUME_HANDLERS = Registry("volume handlers")

VOLUME_HANDLERS(name="keep")(KeepVolume)
VOLUME_HANDLERS(name="max")(ReduceVolume)
VOLUME_HANDLERS(name="mean", reduction=np.mean)(ReduceVolume)
VOLUME_HANDLERS(name="slice")(SliceAtLocation)

# Multi-frame reductions
for output_frames in (1, 8, 10, 16):
    for skip_edge_frames in (0, 5, 10):
        VOLUME_HANDLERS(
            name=f"max-{output_frames}-{skip_edge_frames}",
            output_frames=output_frames,
            skip_edge_frames=skip_edge_frames,
        )(ReduceVolume)


def parse_args() -> Namespace:
    parser = ArgumentParser(prog="dicom-preprocess")
    parser.add_argument(
        "input", type=Path, help="Path to input. Can be a DICOM file, directory, or text file of paths to DICOM files."
    )
    parser.add_argument("output", type=Path, help="Path to output dir")
    parser.add_argument("-n", "--num-workers", type=int, default=8, help="Number of workers for dataloader")
    parser.add_argument(
        "-d", "--device", type=torch.device, default=torch.device("cpu"), help="Device to use for augmentations"
    )
    parser.add_argument("-p", "--prefetch-factor", type=int, default=4, help="Prefetch factor for dataloader")
    parser.add_argument("-s", "--size", nargs=2, type=int, default=None, help="Output image size")
    parser.add_argument("-v", "--volume-handler", default="keep", help="Volume handler")
    parser.add_argument("-m", "--resize-mode", default="bilinear", help="Resize mode")
    parser.add_argument(
        "-f", "--output-format", default="png", choices=OUTPUT_FORMATS, help="Preprocessing output format"
    )
    return parser.parse_args()


def main(args: Namespace):
    # Build transform list
    if args.size:
        H, W = tuple(args.size)
        crop = MinMaxCrop()
        resize = Resize(size=(H, W), mode=args.resize_mode)
        transforms = [
            crop,
            resize,
        ]
    else:
        transforms = []

    inp = Path(args.input)
    dest_dir = Path(args.output)
    if not inp.exists():
        raise FileNotFoundError(f"Input {inp} does not exist")  # pragma: no cover
    if not dest_dir.is_dir():
        raise NotADirectoryError(f"Output directory {dest_dir} does not exist")  # pragma: no cover

    # `iterate_input_path` will recurse into the output directory, so we need to
    # check that the output directory is not a subdirectory of the input directory.
    if inp.is_dir() and dest_dir.is_relative_to(inp):
        raise ValueError(f"Output directory {dest_dir} cannot be a subdirectory of input directory {inp}")

    # TODO: Batch size should be configurable. It is hard-coded to 1 for now because we
    # cannot collate inputs of different sizes. We should either pad inputs or disable
    # batching.
    print(f"NVJPEG Available: {nvjpeg2k_is_available()}")

    volume_handler = cast(VolumeHandler, VOLUME_HANDLERS.get(args.volume_handler).instantiate_with_metadata().fn)
    pipeline = PreprocessingPipeline(
        iterate_input_path(inp),
        num_workers=args.num_workers,
        batch_size=1,
        device=args.device,
        prefetch_factor=args.prefetch_factor,
        transforms=transforms,
        volume_handler=volume_handler,
        output_format=args.output_format,
    )
    pipeline(args.output)


def entrypoint():
    main(parse_args())


if __name__ == "__main__":
    entrypoint()
