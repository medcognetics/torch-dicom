import json
from abc import ABC, abstractclassmethod, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterator, Sized, Union, cast

from torch.utils.data import Dataset, IterableDataset


class MetadataInputWrapper(IterableDataset, ABC):
    r"""Wraps an existing input that returns a dictionary of items and adds metadata from a file.

    Args:
        dataset: Iterable dataset to wrap.
        metadata: Path to a file with metadata.
    """

    def __init__(self, dataset: Union[Dataset, IterableDataset], metadata: Path):
        self.dataset = dataset
        if not metadata.is_file():
            raise FileNotFoundError(f"Metadata file {metadata} does not exist.")  # pragma: no cover
        self.metadata_path = Path(metadata)
        # Trigger loading of metadata
        self.metadata

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
        for example in self.dataset:
            if not isinstance(example, dict):
                raise TypeError(
                    f"{self.__class__.__name__} expects examples to be of type dict, got {type(example)}"
                )  # pragma: no cover
            metadata = self.get_metadata(example)
            example.update(metadata)
            yield example

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset}, metadata={self.metadata_path})"


class MetadataDatasetWrapper(Dataset, ABC):
    r"""Wraps an existing dataset that returns a dictionary of items and adds metadata from a file.

    Args:
        dataset: Dataset to wrap.
        metadata: Path to a file with metadata.
    """

    def __init__(self, dataset: Dataset, metadata: Path):
        self.dataset = dataset
        if not metadata.is_file():
            raise FileNotFoundError(f"Metadata file {metadata} does not exist.")  # pragma: no cover
        self.metadata_path = Path(metadata)
        # Trigger loading of metadata
        self.metadata

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
        example = self.dataset[idx]
        if not isinstance(example, dict):
            raise TypeError(
                f"{self.__class__.__name__} expects examples to be of type dict, got {type(example)}"
            )  # pragma: no cover
        metadata = self.get_metadata(example)
        example.update(metadata)
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

    The loaded metadata is added to the example under the key "preprocessing".

    Args:
        dataset: Dataset to wrap.
        metadata: Path to directory of metadata JSON files.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

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
        return {"preprocessing": self.load_metadata(path)}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={self.dataset})"
