#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from dicom_utils.container.collection import iterate_input_path

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
    parser.add_argument("-s", "--size", nargs="+", type=int, default=None, help="Output image size")
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

    # TODO: Batch size should be configurable. It is hard-coded to 1 for now because we
    # cannot collate inputs of different sizes. We should either pad inputs or disable
    # batching.
    pipeline = PreprocessingPipeline(
        iterate_input_path(args.input),
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
