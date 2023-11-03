import os
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Final, Iterator, List, Optional, Sequence, Sized, Tuple, Union, cast

import torch
from data_organizer.core.database.manifest import AnnotationManifest, Manifest
from dicom_utils.volume import VOLUME_HANDLERS, VolumeHandler
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from .dataset import AnnotatedDicomPathDataset, AnnotatedDicomStudyPathDataset, collate_fn
from .manager import ManifestManager, make_manager
from .sampler import SAMPLER_REGISTRY
from .transform import Compose


TRAIN_VOLUME_HANDLER: Final = cast(VolumeHandler, VOLUME_HANDLERS.get("max-1-5").instantiate_with_metadata().fn)
INFER_VOLUME_HANDLER: Final = cast(VolumeHandler, VOLUME_HANDLERS.get("max-8-5").instantiate_with_metadata().fn)


def _prepare_inputs(inputs: Union[str, Sequence[str], os.PathLike]) -> List[Path]:
    if isinstance(inputs, (str, os.PathLike, Path)):
        return [Path(inputs)]
    return [Path(i) for i in inputs]


# Adapted from
# https://github.com/Lightning-AI/lightning-bolts VisionDataModule
class MammogramDataModule(LightningDataModule):
    r"""DataModule for mammographic train/test data."""

    def __init__(
        self,
        train_inputs: Union[str, Sequence[str]],
        val_inputs: Union[str, Sequence[str], int, float] = 0.1,
        test_inputs: Union[str, Sequence[str]] = [],
        img_size: Optional[Tuple[int, int]] = None,
        batch_size: int = 4,
        seed: int = 42,
        shuffle: bool = True,
        train_transforms: Optional[Union[Callable, Compose]] = None,
        val_transforms: Optional[Union[Callable, Compose]] = None,
        test_transforms: Optional[Union[Callable, Compose]] = None,
        dataloader_config: Dict[str, Any] = {},
        train_dataset_config: Dict[str, Any] = {},
        dataset_config: Dict[str, Any] = {},
        sampler_name: Optional[str] = None,
        sampler_config: Dict[str, Any] = {},
        include_study_images: bool = False,
        train_volume_handler: VolumeHandler = TRAIN_VOLUME_HANDLER,
        infer_volume_handler: VolumeHandler = INFER_VOLUME_HANDLER,
    ) -> None:
        super().__init__()
        self.train_inputs = _prepare_inputs(train_inputs)
        self.val_inputs = val_inputs if isinstance(val_inputs, (int, float)) else _prepare_inputs(val_inputs)
        self.test_inputs = _prepare_inputs(test_inputs)
        self.img_size = img_size
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.include_study_images = include_study_images

        dataloader_config.setdefault("num_workers", 0)
        dataloader_config.setdefault("pin_memory", True)
        self.dataloader_config = dataloader_config
        self.train_dataset_config = train_dataset_config
        self.dataset_config = dataset_config
        self._dims = None
        self.sampler = (
            SAMPLER_REGISTRY.get(sampler_name).bind_metadata(**sampler_config) if sampler_name is not None else None
        )
        self.train_volume_handler = train_volume_handler
        self.infer_volume_handler = infer_volume_handler

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        # Prepare configs
        train_dataset_config = self.train_dataset_config
        train_dataset_config["volume_handler"] = deepcopy(self.train_volume_handler)
        infer_dataset_config = self.dataset_config
        infer_dataset_config["volume_handler"] = deepcopy(self.infer_volume_handler)

        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

            # prepare training dataset
            train_datasets = [
                self.create_dataset(inp, transform=train_transforms, **train_dataset_config)
                for inp in self.train_inputs
            ]

            # prepare validation dataset
            # if self.val_inputs is a float, split the training dataset
            if isinstance(self.val_inputs, (int, float)):
                val_datasets = [
                    self.create_dataset(
                        inp,
                        manifest=train_ds.manifest,
                        annotations=train_ds.annotations,
                        transform=val_transforms,
                        **infer_dataset_config,
                    )
                    for inp, train_ds in zip(self.train_inputs, train_datasets)
                ]
                self.dataset_train = self._split_dataset(ConcatDataset(train_datasets), train=True)
                self.dataset_val = self._split_dataset(ConcatDataset(val_datasets), train=False)

            # otherwise, use the validation dataset as is
            else:
                val_datasets = [
                    self.create_dataset(inp, transform=val_transforms, **infer_dataset_config)
                    for inp in self.val_inputs
                ]
                self.dataset_train = ConcatDataset(train_datasets)
                self.dataset_val = ConcatDataset(val_datasets)

            if self.sampler is not None:
                self.train_sampler = self.sampler(self.dataset_train)
                if (sample_len := len(self.train_sampler)) < self.batch_size:
                    raise RuntimeError(  # pragma: no cover
                        f"Sampler {self.train_sampler} length {sample_len} is smaller "
                        f"than batch size {self.batch_size}."
                    )
                print(f"Sampler:\n{self.train_sampler.membership_counts}")
            else:
                self.train_sampler = None

            # Validate what we have
            assert isinstance(self.dataset_train, Sized)
            assert isinstance(self.dataset_val, Sized)
            if not len(self.dataset_train):
                raise RuntimeError(f"Empty training dataset from inputs: {self.train_inputs}")  # pragma: no cover
            if not len(self.dataset_val):
                raise RuntimeError(f"Empty validation dataset from inputs: {self.val_inputs}")  # pragma: no cover
            self._dims = train_datasets[0].img_size

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            test_datasets = [
                self.create_dataset(inp, transform=test_transforms, **infer_dataset_config) for inp in self.test_inputs
            ]
            self.dataset_test = ConcatDataset(test_datasets)
            assert isinstance(self.dataset_test, Sized)
            if not len(self.dataset_test):
                raise RuntimeError(f"Empty test dataset from inputs: {self.test_inputs}")  # pragma: no cover
            self._dims = test_datasets[0].img_size

    def _split_dataset(self, dataset: Dataset, train: bool = True) -> Dataset:
        """Splits the dataset into train and validation set."""
        assert isinstance(dataset, Sized)
        len_dataset = len(dataset)
        splits = self._get_splits(len_dataset)
        dataset_train, dataset_val = random_split(dataset, splits, generator=torch.Generator().manual_seed(self.seed))

        return dataset_train if train else dataset_val

    def _get_splits(self, len_dataset: int) -> Tuple[int, int]:
        """Computes split lengths for train and validation set.

        Args:
            len_dataset: Length of the train dataset.

        Returns:
            Tuple of train and validation set lengths.
        """
        if isinstance(self.val_inputs, int):
            train_len = len_dataset - self.val_inputs
            splits = (train_len, self.val_inputs)
        elif isinstance(self.val_inputs, float):
            val_len = int(self.val_inputs * len_dataset)
            train_len = len_dataset - val_len
            splits = (train_len, val_len)
        else:
            raise TypeError(f"Unsupported type {type(self.val_inputs)}")  # pragma: no cover
        return splits

    def default_transforms(self) -> Optional[Callable]:
        """Default transform for the dataset."""
        return

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""
        return self._data_loader(self.dataset_train, train=True)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The val dataloader."""
        return self._data_loader(self.dataset_val, train=False)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader."""
        return self._data_loader(self.dataset_test, train=False)

    def _data_loader(self, dataset: Dataset, train: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train and self.train_sampler is None,
            drop_last=train,
            sampler=self.train_sampler if train else None,
            collate_fn=collate_fn,
            **self.dataloader_config,
        )