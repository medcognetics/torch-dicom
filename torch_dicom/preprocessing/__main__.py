#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, cast, Final

import numpy as np
import torch
from dicom_utils.container.collection import iterate_input_path
from dicom_utils.dicom import nvjpeg2k_is_available
from dicom_utils.volume import KeepVolume, ReduceVolume, SliceAtLocation, VolumeHandler
from registry import Registry

from .crop import MinMaxCrop, ROICrop
from .pipeline import OutputFormat, PreprocessingPipeline
from .resize import Resize


WARNING_IGNORE_SUBSTRINGS: Final = ["Bits Stored value", "Invalid value for VR"]

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
    parser.add_argument(
        "-v", "--volume-handler", default="keep", choices=VOLUME_HANDLERS.available_keys(), help="Volume handler"
    )
    parser.add_argument("-m", "--resize-mode", default="bilinear", help="Resize mode")
    parser.add_argument(
        "-f",
        "--output-format",
        default="png",
        choices=[str(x) for x in OutputFormat],
        help="Preprocessing output format",
    )
    parser.add_argument("-co", "--compression", type=str, default=None, help="Compression passed to PIL.Image.save")
    parser.add_argument("-c", "--crop", default=None, choices=["minmax", "roi"], help="Cropping method.")
    parser.add_argument(
        "-r", "--rois", type=Path, default=None, help="Path to ROI metadata to be used for ROI cropping."
    )
    return parser.parse_args()


def main(args: Namespace):
    # Cropping
    if args.crop == "roi":
        if args.rois is None:
            raise ValueError("ROI cropping selected but no ROI metadata provided. " "Please see the --rois argument.")
        elif not args.rois.is_file():
            raise FileNotFoundError(f"ROI metadata file {args.rois} does not exist")
        elif not args.size:
            # TODO: Allow ROICrop to accept None for min_size, in which case we use the ROI size
            raise ValueError("ROI cropping selected but no output size provided. Please see the --size argument.")
        crop = ROICrop(path=args.rois, min_size=args.size)
    elif args.crop == "minmax":
        crop = MinMaxCrop()
    elif args.crop is None:
        crop = None
    else:
        raise ValueError(f"Invalid crop value: {args.crop}")

    # Resize
    if args.size:
        H, W = tuple(args.size)
        resize = Resize(size=(H, W), mode=args.resize_mode)
    else:
        resize = None

    # Build transform list
    transforms = [cast(Any, t) for t in (crop, resize) if t is not None]

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
        output_format=OutputFormat(args.output_format),
        compression=args.compression,
    )
    pipeline(args.output)


def entrypoint():
    """Entry point for the preprocessing script."""
    for substring in WARNING_IGNORE_SUBSTRINGS:
        warnings.filterwarnings("ignore", message=f".*{substring}.*")
    main(parse_args())


if __name__ == "__main__":
    entrypoint()
