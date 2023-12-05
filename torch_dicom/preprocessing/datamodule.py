import numpy as np
import os
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Final, Iterator, List, Optional, Sequence, Sized, Tuple, Union, cast, Iterable, Sized
from functools import partial

import torch
from dicom_utils.volume import VOLUME_HANDLERS, VolumeHandler
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms.v2 import Compose
from torch_dicom.datasets import collate_fn, DicomPathDataset, ImagePathDataset, DataFrameMetadata, BoundingBoxMetadata, PreprocessingConfigMetadata, MetadataDatasetWrapper
from torch_dicom.datasets.sampler import WeightedCSVSampler, BatchComplementSampler
from torch.utils.data import Sampler, BatchSampler, RandomSampler, SequentialSampler
from torch_dicom.datasets.helpers import Transform
from deep_helpers.data.sampler import ConcatBatchSampler, ConcatSampler


PathLike = Union[str, os.PathLike, Path]


def _prepare_inputs(inputs: Union[PathLike, Sequence[PathLike]]) -> List[Path]:
    return [
        Path(i) 
        for i in ([inputs] if isinstance(inputs, (str, os.PathLike, Path)) else inputs)
    ]


class PreprocessedPNGDataModule(LightningDataModule):
    r"""Data module for preprocessed PNG images."""

    def __init__(
        self,
        train_inputs: Union[PathLike, Sequence[PathLike]],
        val_inputs: Union[PathLike, Sequence[PathLike]] = [],
        test_inputs: Union[PathLike, Sequence[PathLike]] = [],
        batch_size: int = 4,
        seed: int = 42,
        train_transforms: Optional[Transform] = None,
        train_gpu_transforms: Optional[Transform] = None,
        val_transforms: Optional[Transform] = None,
        test_transforms: Optional[Transform] = None,
        train_dataset_kwargs: Dict[str, Any] = {},
        dataset_kwargs: Dict[str, Any] = {},
        balance_malign: bool = False,
        image_complements: bool = False,
        metadata_filenames: Dict[str, str] = {},
        boxes_filename: str = "traces.csv",
        **kwargs,
    ) -> None:
        super().__init__()
        self.train_inputs = _prepare_inputs(train_inputs)
        self.val_inputs = _prepare_inputs(val_inputs)
        self.test_inputs = _prepare_inputs(test_inputs)
        self.batch_size = batch_size
        self.seed = seed
        self.train_transforms = train_transforms
        self.train_gpu_transforms = train_gpu_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.train_dataset_kwargs = train_dataset_kwargs
        self.dataset_kwargs = dataset_kwargs
        self.balance_malign = balance_malign
        self.image_complements = image_complements
        self.metadata_filenames = metadata_filenames
        self.boxes_filename = boxes_filename

        self.dataloader_config = kwargs
        self.dataloader_config.setdefault("num_workers", 0)
        self.dataloader_config.setdefault("pin_memory", True)

    def create_dataset(
        self,
        target: PathLike,
        normalize=False,
        **kwargs,
    ) -> Dataset:
        target = Path(target)
        if not target.is_dir():
            raise NotADirectoryError(target)  # pragma: no cover

        # Create a dataset of preprocessed images and apply the metadata wrapper
        images = target.rglob("*.png")
        dataset = ImagePathDataset(images, **kwargs, normalize=normalize)
        dataset = PreprocessingConfigMetadata(dataset)

        # Apply any additional metadata wrappers configured in self.metadata_filenames
        for key, filename in self.metadata_filenames.items():
            if (metadata_path := target / filename).exists():
                dataset = DataFrameMetadata(dataset, metadata_path, dest_key=key)

        # Apply a bounding box wrapper if the boxes CSV file exists
        if (traces_path := target / self.boxes_filename).exists():
            dataset = BoundingBoxMetadata(dataset, traces_path, extra_keys=("trait", "types"))

        return dataset

    def create_sampler(self, dataset: Union[ImagePathDataset, MetadataDatasetWrapper], root: Path) -> BatchSampler:
        # Get the file list, stepping through any wrappers if needed
        _dataset = dataset
        while isinstance(_dataset, MetadataDatasetWrapper):
            _dataset = _dataset.dataset
        assert isinstance(_dataset, ImagePathDataset)
        files = _dataset.files.to_list()

        # Build a weighted sampler for malignant / benign
        if self.balance_malign and (annotation_path := root / "annotation.csv").exists():
            weights = {"True": 0.25, "False": 0.75}
            sampler = WeightedCSVSampler(annotation_path, files, "malignant", weights)
        else:
            sampler = RandomSampler(dataset)

        # Build a batch sampler
        if self.image_complements and (manifest_path := root / "manifest.csv").exists():
            batch_sampler = PatientComplementSampler(sampler, self.batch_size, manifest_path, files)
        else:
            batch_sampler = BatchSampler(sampler, self.batch_size, drop_last=True)

        return batch_sampler

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""

        if stage == "fit" or stage is None:
            # prepare training dataset
            train_dataset_config = self.train_dataset_kwargs
            train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            train_datasets = [
                self.create_dataset(inp, transform=train_transforms, **train_dataset_config)
                for inp in self.train_inputs
            ]
            self.dataset_train = ConcatDataset(train_datasets)
            assert isinstance(self.dataset_train, Sized)
            if not len(self.dataset_train):
                raise RuntimeError(f"Empty training dataset from inputs: {self.train_inputs}")  # pragma: no cover

            # prepare training batch sampler
            train_batch_samplers: List[BatchSampler] = [
                self.create_sampler(dataset, root)
                for dataset, root in zip(train_datasets, self.train_inputs)
            ]
            train_samplers = cast(List[Sampler], [batch_sampler.sampler for batch_sampler in train_batch_samplers])
            self.train_sampler = ConcatBatchSampler(train_samplers, train_batch_samplers, method="zip")

            # prepare validation dataset
            infer_dataset_config = self.dataset_kwargs
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms
            val_datasets = [
                self.create_dataset(inp, transform=val_transforms, **infer_dataset_config)
                for inp in self.val_inputs
            ]
            self.dataset_val = ConcatDataset(val_datasets)
            assert isinstance(self.dataset_val, Sized)
            if not len(self.dataset_val):
                raise RuntimeError(f"Empty validation dataset from inputs: {self.val_inputs}")  # pragma: no cover

        if stage == "test" or stage is None:
            infer_dataset_config = self.dataset_kwargs
            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            test_datasets = [
                self.create_dataset(inp, transform=test_transforms, **infer_dataset_config) for inp in self.test_inputs
            ]
            self.dataset_test = ConcatDataset(test_datasets)
            assert isinstance(self.dataset_test, Sized)
            if not len(self.dataset_test):
                raise RuntimeError(f"Empty test dataset from inputs: {self.test_inputs}")  # pragma: no cover

    @abstractmethod
    def default_transforms(self) -> Optional[Callable]:
        """Default transform for the dataset."""
        return

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""
        config = {k: v for k, v in self.dataloader_config.items() if k != "batch_sampler"}
        return DataLoader(
            self.dataset_train,
            batch_sampler=self.train_sampler,
            collate_fn=partial(collate_fn, default_fallback=False),
            **config,
        )

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
            shuffle=train,
            drop_last=train,
            collate_fn=partial(collate_fn, default_fallback=False),
            **self.dataloader_config,
        )