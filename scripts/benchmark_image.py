import argparse
import tempfile
import timeit
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def generate_uint16_array(size: Tuple[int, int], random: bool = True) -> np.ndarray:
    """Generate a uint16 array with a single channel, either random or all zeros."""
    if random:
        return np.random.randint(0, 65536, size=(1, *size), dtype=np.uint16)
    return np.zeros((1, *size), dtype=np.uint16)


def save_image(arr: np.ndarray, path: Path) -> None:
    """Save a uint16 array as an image."""
    Image.fromarray(arr.squeeze(), mode="I;16").save(path)


def load_image(path: Path) -> np.ndarray:
    """Load an image as a uint16 array."""
    return np.array(Image.open(path)).astype(np.uint16)[np.newaxis, ...]


def benchmark_format(
    arr: np.ndarray, tmp_path: Path, extension: str, repetitions: int = 10
) -> Tuple[float, float, float]:
    """Benchmark saving and loading for a specific format."""
    file_path = tmp_path / f"test.{extension}"

    def save_operation():
        save_image(arr, file_path)

    def load_operation():
        return load_image(file_path)

    save_time = timeit.timeit(save_operation, number=repetitions) * 1000 / repetitions
    load_time = timeit.timeit(load_operation, number=repetitions) * 1000 / repetitions

    loaded_arr = load_operation()
    assert np.array_equal(arr, loaded_arr), f"Loaded array does not match original for {extension}"

    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    return save_time, load_time, file_size_mb


def main(image_size: Tuple[int, int], random: bool = False) -> None:
    """Main function to run benchmarks."""
    arr = generate_uint16_array(image_size, random=random)

    formats = {
        "png": "PNG",
        "tiff": "TIFF",
    }

    theoretical_size_mb = (image_size[0] * image_size[1] * 2) / (1024 * 1024)
    print(f"Benchmarking image I/O operations for size: {image_size[0]}x{image_size[1]}")
    print(f"Theoretical size (2 bytes per pixel): {theoretical_size_mb:.2f}MB")
    print(f"Random data: {random}")
    print("-" * 60)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for ext, format_name in formats.items():
            save_time, load_time, file_size_mb = benchmark_format(arr, tmp_path, ext)
            print(f"{format_name} - Save: {save_time:.2f}ms, Load: {load_time:.2f}ms, Size: {file_size_mb:.2f}MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark image I/O operations.")
    parser.add_argument("-s", "--image-size", nargs=2, type=int, default=(2048, 1536), help="Image size (width height)")
    parser.add_argument("-r", "--random", action="store_true", help="Generate random image data")
    args = parser.parse_args()

    main(tuple(args.image_size))
