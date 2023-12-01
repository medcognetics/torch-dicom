import json
from abc import ABC, abstractclassmethod, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sized, Tuple, Union, cast

import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from ..preprocessing.crop import MinMaxCrop
from ..preprocessing.resize import Resize
from .helpers import SupportsTransform, Transform


def get_sopuid_key_from_example(example: Dict[str, Any]) -> Optional[Any]:
    """
    Extracts the SOPInstanceUID from the given example.

    The key is extracted from the 'record' field of the example if it exists.
    If not, it is extracted from the 'path' field if it exists and is not None, assuming
    that the path stem is the SOPInstanceUID.
    If neither conditions are met, None is returned.

    Args:
        example: The example from which to extract the key.

    Returns:
        The extracted key if it exists, None otherwise.
    """
    key = (
        example["record"].SOPInstanceUID
        if "record" in example
        else example["path"].stem
        if "path" in example and example["path"] is not None
        else None
    )
    return key


def _disable_wrapped_transform(wrapper: Union["MetadataDatasetWrapper", "MetadataInputWrapper"]) -> None:
    r"""Disables the transform in the wrapped dataset and attaches the transform to the wrapper."""
    if isinstance(wrapper.dataset, SupportsTransform):
        wrapper.transform = wrapper.dataset.transform
        wrapper.dataset.transform = None
    else:
        wrapper.transform = None


def _assert_wrapped_transform_disabled(wrapper: Union["MetadataDatasetWrapper", "MetadataInputWrapper"]) -> None:
    assert (
        not isinstance(wrapper.dataset, SupportsTransform) or wrapper.dataset.transform is None
    ), "Transform in wrapped dataset must be disabled"


class MetadataInputWrapper(IterableDataset, ABC, SupportsTransform):
    r"""Wraps an existing input that returns a dictionary of items and adds metadata from a file.

    Args:
        dataset: Iterable dataset to wrap.
        metadata: Path to a file with metadata.
    """
    transform: Optional[Transform]

    def __init__(self, dataset: Union[Dataset, IterableDataset], metadata: Path):
        self.dataset = dataset
        if not metadata.is_file():
            raise FileNotFoundError(f"Metadata file {metadata} does not exist.")  # pragma: no cover
        self.metadata_path = Path(metadata)
        # Trigger loading of metadata
        self.metadata
        _disable_wrapped_transform(self)

    @abstractclassmethod
    def load_metadata(cls, metadata: Path) -> Any:
        r"""Loads metadata from a file."""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get_metadata(self, example: Dict[str, Any]) -> Dict[str, Any]:
        r"""Gets metadata for a given example.

        Returns:
            Dictionary of metadata to be added to the example.
        """
        raise NotImplementedError  # pragma: no cover

    @cached_property
    def metadata(self) -> Any:
        return self.load_metadata(self.metadata_path)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        _assert_wrapped_transform_disabled(self)
        for example in self.dataset:
            if not isinstance(example, dict):
                raise TypeError(
                    f"{self.__class__.__name__} expects examples to be of type dict, got {type(example)}"
                )  # pragma: no cover
            metadata = self.get_metadata(example)
            example.update(metadata)

            # Apply transform
            if self.transform is not None:
                example = self.apply_transform(example)

            yield example

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset}, metadata={self.metadata_path})"


class MetadataDatasetWrapper(Dataset, ABC, SupportsTransform):
    r"""Wraps an existing dataset that returns a dictionary of items and adds metadata from a file.

    Args:
        dataset: Dataset to wrap.
        metadata: Path to a file with metadata.
    """
    transform: Optional[Transform]

    def __init__(self, dataset: Dataset, metadata: Path):
        self.dataset = dataset
        if not metadata.is_file():
            raise FileNotFoundError(f"Metadata file {metadata} does not exist.")  # pragma: no cover
        self.metadata_path = Path(metadata)
        # Trigger loading of metadata
        self.metadata
        _disable_wrapped_transform(self)

    @abstractclassmethod
    def load_metadata(cls, metadata: Path) -> Any:
        r"""Loads metadata from a file."""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get_metadata(self, example: Dict[str, Any]) -> Dict[str, Any]:
        r"""Gets metadata for a given example.

        Returns:
            Dictionary of metadata to be added to the example.
        """
        raise NotImplementedError  # pragma: no cover

    @cached_property
    def metadata(self) -> Any:
        return self.load_metadata(self.metadata_path)

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        _assert_wrapped_transform_disabled(self)
        example = self.dataset[idx]
        if not isinstance(example, dict):
            raise TypeError(
                f"{self.__class__.__name__} expects examples to be of type dict, got {type(example)}"
            )  # pragma: no cover
        metadata = self.get_metadata(example)
        example.update(metadata)

        # Apply transform
        if self.transform is not None:
            example = self.apply_transform(example)

        return example

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset}, metadata={self.metadata_path})"


class PreprocessingConfigMetadata(MetadataDatasetWrapper):
    r"""Adds preprocessing configuration metadata to a dataset that was preprocessed
    using this library. It is assumed that the preprocessed data contains "images"
    and "metadata" subdirectories. The respective metadata JSON file is loaded
    by replacing the "images" directory with "metadata" and changing the file
    extension to ".json".

    The loaded metadata is added to the example under the key ``dest_key``.

    Args:
        dataset: Dataset to wrap.
        metadata: Path to directory of metadata JSON files.
        dest_key: Key under which to add the metadata to the example.
    """

    def __init__(self, dataset: Dataset, dest_key: str = "preprocessing"):
        self.dataset = dataset
        self.dest_key = dest_key
        _disable_wrapped_transform(self)

    @classmethod
    def load_metadata(cls, metadata: Path) -> Any:
        r"""Loads metadata from a file."""
        if not metadata.is_file():
            raise FileNotFoundError(metadata)  # pragma: no cover
        with open(metadata, "r") as f:
            return json.load(f)

    def get_metadata(self, example: Dict[str, Any]) -> Dict[str, Any]:
        r"""Gets preprocessing metadata for a given example.

        Returns:
            Dictionary of metadata to be added to the example.
        """
        if not isinstance(example, dict):
            raise TypeError(
                f"{self.__class__.__name__} expects examples to be of type dict, got {type(example)}"
            )  # pragma: no cover

        path = (
            example["path"]
            if "path" in example and example["path"] is not None
            else example["record"].path
            if "record" in example
            else None
        )
        if path is None:
            raise KeyError(f"Unable to find path in example {example}")  # pragma: no cover

        path = Path(str(path).replace("images", "metadata")).with_suffix(".json")
        return {self.dest_key: self.load_metadata(path)}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset})"


class BoundingBoxMetadata(MetadataDatasetWrapper):
    r"""Wraps an existing dataset that returns a dictionary of items and adds bounding box metadata from a file.

    The metadata file is expected to be a CSV file with the following columns:
        * SOPInstanceUID: SOPInstanceUID of the image
        * x1: x coordinate of the top left corner of the bounding box (absolute coordinates)
        * y1: y coordinate of the top left corner of the bounding box (absolute coordinates)
        * x2: x coordinate of the bottom right corner of the bounding box (absolute coordinates)
        * y2: y coordinate of the bottom right corner of the bounding box (absolute coordinates)

    Bounding boxes are added to the example under the key ``dest_key``. Subkeys for "boxes", and any ``extra_keys``
    will be present. The bounding boxes will be :class:`torchvision.tv_tensors.BoundingBoxes` with format
    :class:`~torchvision.tv_tensors.BoundingBoxFormat.XYXY`. If preprocessing metadata is available, the box coordinates
    are transformed accordingly.

    Args:
        dataset: Dataset to wrap.
        metadata: Path to a file with metadata.
        extra_keys: List of extra keys to add to the example.
        dest_key: Key under which to add the bounding boxes to the example.

    Shapes:
        * ``boxes`` - :math:`(N, 4)` where :math:`N` is the number of bounding boxes.
        * Extra keys will be lists of length :math:`N`.
    """
    metadata: pd.DataFrame

    def __init__(
        self, dataset: Dataset, metadata: Path, extra_keys: Iterable[str] = [], dest_key: str = "bounding_boxes"
    ):
        super().__init__(dataset, metadata)
        self.extra_keys = extra_keys
        self.dest_key = dest_key

    @classmethod
    def load_metadata(cls, metadata: Path) -> pd.DataFrame:
        r"""Loads metadata from a file."""
        df = pd.read_csv(metadata, index_col="SOPInstanceUID")
        return df

    def get_key_from_example(self, example: Dict[str, Any]) -> Optional[Any]:
        """
        Extracts the key from the given example.

        The key is extracted from the 'record' field of the example if it exists.
        If not, it is extracted from the 'path' field if it exists and is not None.
        If neither conditions are met, None is returned.

        Args:
            example: The example from which to extract the key.

        Returns:
            The extracted key if it exists, None otherwise.
        """
        return get_sopuid_key_from_example(example)

    def get_metadata(self, example: Dict[str, Any]) -> Dict[str, Any]:
        key = self.get_key_from_example(example)
        if key is None:
            raise KeyError(f"Unable to find key in example {example}")  # pragma: no cover
        elif key not in self.metadata.index:
            return {self.dest_key: {}}

        # Loop through boxes and apply preprocessing
        bboxes, extra_keys = [], {}
        for box in self._iterate_boxes(key, example.get("preprocessing", {})):
            bboxes.append(box["bbox"])
            for k in self.extra_keys:
                extra_keys.setdefault(k, []).append(box.get(k, None))

        # Convert bboxes to BoundingBoxes. Since preprocessing is applied to the bounding boxes,
        # the canvas size is the same as the image size.
        if "img_size" in example:
            img_size = tuple(example["img_size"].tolist())
        elif "img" in example:
            img_size = tuple(example["img"].shape[-2:])
        else:
            raise KeyError(f"Unable to find img_size in example {example}")  # pragma: no cover
        assert len(img_size) == 2, f"Expected 2D image size, got {img_size}"
        bboxes = BoundingBoxes(
            torch.stack(bboxes),
            format=BoundingBoxFormat.XYXY,
            canvas_size=cast(Tuple[int, int], img_size),
        )

        return {
            self.dest_key: {
                "boxes": bboxes,
                **extra_keys,
            }
        }

    def _iterate_boxes(self, key: str, preprocessing_config: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        rows = self.metadata.loc[key]
        if isinstance(rows, pd.Series):
            rows = pd.DataFrame(rows).transpose()

        for _, row in rows.iterrows():
            bbox = torch.tensor(
                [row["x1"], row["y1"], row["x2"], row["y2"]],
                dtype=torch.long,
            )

            # Apply preprocessing to bounding box in the same order as the image
            if "crop_bounds" in preprocessing_config:
                bounds = torch.tensor(preprocessing_config["crop_bounds"]).view(1, -1).expand(2, -1)
                bbox = MinMaxCrop.apply_to_coords(bbox.view(-1, 2), bounds)
            if "resize_config" in preprocessing_config:
                bbox = Resize.apply_to_coords(bbox.view(-1, 2), preprocessing_config["resize_config"])

            yield {"bbox": bbox.view(4), **{k: row[k] for k in self.extra_keys}}


class DataFrameMetadata(MetadataDatasetWrapper):
    r"""Wraps an existing dataset that returns a dictionary of items and adds metadata from a file / DataFrame.
    The default implementation assumes that the metadata file is a CSV file indexed by a "SOPInstanceUID" column.
    All columns are added to the example under the key ``dest_key``.

    Args:
        dataset: Dataset to wrap.
        metadata: Path to a file with metadata.
        dest_key: Key under which to add the metadata to the example.
    """
    metadata: pd.DataFrame

    def __init__(self, dataset: Dataset, metadata: Path, dest_key: str = "metadata"):
        super().__init__(dataset, metadata)
        self.dest_key = dest_key

    @classmethod
    def load_metadata(cls, metadata: Path) -> pd.DataFrame:
        r"""Loads metadata from a file."""
        return pd.read_csv(metadata, index_col="SOPInstanceUID")

    def get_key_from_example(self, example: Dict[str, Any]) -> Optional[Any]:
        """
        Extracts the key from the given example.

        The key is extracted from the 'record' field of the example if it exists.
        If not, it is extracted from the 'path' field if it exists and is not None.
        If neither conditions are met, None is returned.

        Args:
            example: The example from which to extract the key.

        Returns:
            The extracted key if it exists, None otherwise.
        """
        return get_sopuid_key_from_example(example)

    def get_metadata(self, example: Dict[str, Any]) -> Dict[str, Any]:
        r"""Gets metadata for a given example.

        Returns:
            Dictionary of metadata to be added to the example.
        """
        key = self.get_key_from_example(example)
        if key is None:
            raise KeyError(f"Unable to find key in example {example}")  # pragma: no cover
        elif key not in self.metadata.index:
            return {self.dest_key: {}}

        return {self.dest_key: self.metadata.loc[key].to_dict()}
