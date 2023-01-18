#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, Namespace


def main(args: Namespace) -> None:
    ...


def parse_args() -> Namespace:
    parser = ArgumentParser(prog="project")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
