#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from dicom_utils.container.collection import iterate_input_path
from dicom_utils.dicom import nvjpeg2k_is_available

from .crop import MinMaxCrop
from .pipeline import PreprocessingPipeline
from .resize import Resize


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
    return parser.parse_args()


def main(args: Namespace):
    # Build transform list
    if args.size:
        H, W = tuple(args.size)
        crop = MinMaxCrop()
        resize = Resize(size=(H, W))
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

    pipeline = PreprocessingPipeline(
        iterate_input_path(inp),
        num_workers=args.num_workers,
        batch_size=1,
        device=args.device,
        prefetch_factor=args.prefetch_factor,
        transforms=transforms,
    )
    pipeline(args.output)


def entrypoint():
    main(parse_args())


if __name__ == "__main__":
    entrypoint()
