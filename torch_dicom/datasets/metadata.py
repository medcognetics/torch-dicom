from abc import ABC, abstractclassmethod, abstractmethod
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
        self.metadata = self.load_metadata(Path(metadata))

    @abstractclassmethod
    def load_metadata(cls, metadata: Path) -> Any:
        r"""Loads metadata from a file."""
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self, example: Dict[str, Any]) -> Dict[str, Any]:
        r"""Gets metadata for a given example.

        Returns:
            Dictionary of metadata to be added to the example.
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for example in self.dataset:
            if not isinstance(example, dict):
                raise TypeError(
                    f"{self.__class__.__name__} expects examples to be of type dict, got {type(example)}"
                )  # pragma: no cover
            metadata = self.get_metadata(example)
            example.update(metadata)
            yield example


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
        self.metadata = self.load_metadata(Path(metadata))

    @abstractclassmethod
    def load_metadata(cls, metadata: Path) -> Any:
        r"""Loads metadata from a file."""
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self, example: Dict[str, Any]) -> Dict[str, Any]:
        r"""Gets metadata for a given example.

        Returns:
            Dictionary of metadata to be added to the example.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.dataset[idx]
        if not isinstance(example, dict):
            raise TypeError(f"{self.__class__.__name__} expects examples to be of type dict, got {type(example)}")
        metadata = self.get_metadata(example)
        example.update(metadata)
        return example

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]
