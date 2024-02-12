import logging
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from dicom_utils.dicom import Dicom
from dicom_utils.volume import ReduceVolume, VolumeHandler
from PIL import Image
from torch import Tensor
from torch.utils.data import ChainDataset, ConcatDataset, DataLoader, Dataset, IterableDataset
from torchvision.transforms.v2 import Compose
from tqdm import tqdm

from torch_dicom.datasets import (
    DicomInput,
    DicomPathDataset,
    DicomPathInput,
    ImageInput,
    ImagePathDataset,
    ImagePathInput,
    TensorInput,
    TensorPathDataset,
    TensorPathInput,
    collate_fn,
    uncollate,
)
from torch_dicom.datasets.dicom import Transform
from torch_dicom.preprocessing import MinMaxCrop, Resize


B = TypeVar("B")


@dataclass(repr=False)
class InferencePipeline(ABC):
    """
    Inference pipeline for processing medical images.
    It supports multiple input sources including DICOM files, tensors, and images.
    The pipeline is designed to be used with PyTorch models and can handle batch processing.
    It also supports custom datasets and transformations. It is assumed that models will return predictions as a dictionary.

    The pipeline is an iterator that yields input examples and predictions for each example.

    Args:
        dicom_paths: Paths to DICOM files to be processed.
        dicoms: DICOM objects to be processed.
        tensor_paths: Paths to tensor files to be processed.
        tensors: Tensor objects to be processed.
        image_paths: Paths to image files to be processed.
        images: PIL images to be processed.
        custom_inputs: Custom datasets to be processed.
        device: Device to be used for processing.
        batch_size: Size of the batch for processing.
        transform: Optional transformation to be applied to the inputs.
        enumerate_inputs: Whether to enumerate the inputs.
            This enables progress bars to display a total and ETA but may be slower to initialize.
            Note that this will only work if all inputs are map-style datasets. When ``True``, ``skip_errors``
            will not be applied to any enumerated datasets.
        skip_errors: Whether to skip errors during processing.
        volume_handler: Handler for volume data.
        dataloader_kwargs: Additional arguments for the DataLoader.
    """

    # Known input sources
    dicom_paths: Iterable[Path] = field(default_factory=list)
    dicoms: Iterable[Dicom] = field(default_factory=list)
    tensor_paths: Iterable[Path] = field(default_factory=list)
    tensors: Iterable[Tensor] = field(default_factory=list)
    image_paths: Iterable[Path] = field(default_factory=list)
    images: Iterable[Image.Image] = field(default_factory=list)

    # Custom datasets
    custom_inputs: Sequence[Dataset] = field(default_factory=list)

    # Config
    device: torch.device = torch.device("cpu")
    batch_size: int = 1
    transform: Optional[Transform] = None
    enumerate_inputs: bool = False
    skip_errors: bool = True
    volume_handler: VolumeHandler = ReduceVolume()
    dataloader_kwargs: Dict[str, Any] = field(default_factory=dict)
    models: List[nn.Module] = field(default_factory=list)

    def __post_init__(self):
        if not self.models:
            raise ValueError("At least one model must be provided.")

        # Set all models to eval mode
        for model in self.models:
            model.eval()

    @abstractmethod
    def infer_with_model(self, model: nn.Module, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """
        Performs inference with the given model on the given batch of data.

        Args:
            model: The PyTorch model to use for inference.
            batch: The batch of data to be processed.
            batch_idx: The index of the batch.

        Returns:
            A dictionary containing the results of the inference.
        """
        raise NotImplementedError

    @abstractmethod
    def transfer_batch_to_device(self, models: List[nn.Module], batch: B, device: torch.device) -> B:
        """
        Transfers a batch of data to a specified device.

        Args:
            models: The PyTorch models in use.
            batch: The batch of data to be transferred.
            device: The device to which the data should be transferred.

        Returns:
            The batch of data after being transferred to the specified device.
        """
        raise NotImplementedError

    def prepare_dataset(self, **kwargs) -> Union[ChainDataset, ConcatDataset]:
        """
        Prepares the dataset for the inference pipeline.

        This method combines different types of datasets (DICOM, tensor, image, and custom inputs) into a single dataset.
        Depending on the configuration, it either returns a ConcatDataset (if enumerate_inputs is True and every dataset is a Sequence)
        or a ChainDataset (if enumerate_inputs is False or inputs were given that don't have an associated map-style Dataset).

        Keyword Args:
            Forwarded to the dataset constructors.

        Returns:
            The prepared dataset.
        """
        # Build the datasets
        # TODO: We have aggregate datasets in torch_dicom.datasets.chain, but they are not used here.
        # The current implementation gies more control over dataset creation, but it may be worth
        # considering using the aggregate datasets instead.
        datasets: List[Dataset] = [
            *self._prepare_dicom_datasets(**kwargs),
            *self._prepare_tensor_datasets(**kwargs),
            *self._prepare_image_datasets(**kwargs),
            *self.custom_inputs,
        ]

        # If enumerate_inputs is True and every dataset has a length, we will return a ConcatDataset
        #
        # If enumerate_inputs is False or inputs were given that don't have an associated map-style Dataset,
        # we will return a ChainDataset
        return (
            ConcatDataset(datasets)
            if self.enumerate_inputs and not any(isinstance(ds, IterableDataset) for ds in datasets)
            else ChainDataset(datasets)
        )

    def _prepare_dicom_datasets(self, **kwargs) -> Iterator[Dataset]:
        if self.dicom_paths:
            # TODO: Input style datasets support a skip_errors argument, but map-style datasets do not.
            # This is because an iterable style dataset can just skip the error and move on to the next item,
            # but a map-style dataset will raise an error when the item is accessed. This means that enumerate_inputs
            # cannot be used in conjunction with skip_errors for map-style datasets. Address this if possible.
            dataset_type = (
                DicomPathDataset if self.enumerate_inputs else partial(DicomPathInput, skip_errors=self.skip_errors)
            )
            ds = dataset_type(
                iter(self.dicom_paths),
                volume_handler=self.volume_handler,
                transform=self.transform,
                **kwargs,
            )
            yield ds

        if self.dicoms:
            ds = DicomInput(
                self.dicoms,
                skip_errors=self.skip_errors,
                volume_handler=self.volume_handler,
                transform=self.transform,
                **kwargs,
            )
            yield ds

    def _prepare_tensor_datasets(self, **kwargs) -> Iterator[Dataset]:
        if self.tensor_paths:
            dataset_type = (
                TensorPathDataset if self.enumerate_inputs else partial(TensorPathInput, skip_errors=self.skip_errors)
            )
            ds = dataset_type(
                iter(self.tensor_paths),
                transform=self.transform,
                **kwargs,
            )
            yield ds

        if self.tensors:
            ds = TensorInput(self.tensors, skip_errors=self.skip_errors, transform=self.transform, **kwargs)
            yield ds

    def _prepare_image_datasets(self, **kwargs) -> Iterator[Dataset]:
        if self.image_paths:
            dataset_type = (
                ImagePathDataset if self.enumerate_inputs else partial(ImagePathInput, skip_errors=self.skip_errors)
            )
            ds = dataset_type(
                iter(self.image_paths),
                transform=self.transform,
                **kwargs,
            )
            yield ds

        if self.images:
            ds = ImageInput(self.images, skip_errors=self.skip_errors, transform=self.transform, **kwargs)
            yield ds

    def prepare_dataloader(self, dataset: Dataset) -> DataLoader:
        """
        Prepares a DataLoader from the given dataset.

        Args:
            dataset: The dataset to be loaded.

        Returns:
            The DataLoader prepared from the dataset.
        """
        # Set defaults
        dataloader_kwargs = copy(self.dataloader_kwargs)
        dataloader_kwargs.setdefault("collate_fn", collate_fn)
        dataloader_kwargs.setdefault("drop_last", False)

        return DataLoader(dataset, batch_size=self.batch_size, **dataloader_kwargs)

    @classmethod
    def create_default_transform(cls, img_size: Tuple[int, int]) -> Transform:
        """
        Creates a default transformation pipeline for the input images. The default transformation
        first crops the image to nonzero pixels and then resizes the image to the desired size.

        Args:
            img_size: A tuple specifying the desired height and width of the images after transformation.

        Returns:
            Transform
        """
        crop = MinMaxCrop()
        resize = Resize(size=img_size)
        transform = Compose(
            [
                crop,
                resize,
            ]
        )
        return transform

    @torch.no_grad()
    def __call__(
        self,
        use_bar: bool = True,
        desc: str = "Processing",
        **kwargs,
    ) -> Iterator[Tuple[Any, Dict[str, Any]]]:
        """
        Returns an iterator that processes the dataset and yields predictions for each batch.
        It prepares the dataset and dataloader, and then iterates over the dataloader.
        For each batch, it transfers the batch to the device, generates predictions for each model,
        and yields the uncollated batch and predictions.

        Args:
            use_bar: Whether to use a progress bar. Default is True.
            desc: Description for the progress bar. Default is "Processing".

        Keyword Args:
            Forwarded to tqdm bar constructor.

        Yields:
            Tuple containing the uncollated batch and predictions.
        """
        dataset = self.prepare_dataset()
        dataloader = self.prepare_dataloader(dataset)

        is_enumerated = isinstance(dataset, ConcatDataset)
        bar = tqdm(
            total=len(dataset) if is_enumerated else None,
            disable=not use_bar,
            desc=desc,
            **kwargs,
        )

        for batch_idx, batch in enumerate(dataloader):
            # Check this every batch
            assert not any(model.training for model in self.models), "Models must be in eval mode during inference"

            try:
                batch = self.transfer_batch_to_device(self.models, batch, self.device)

                # Generate predictions for each model and merge them into a prediction dictionary
                predictions = {
                    k: v for model in self.models for k, v in self.infer_with_model(model, batch, batch_idx).items()
                }

                # Uncollate the batch and predictions and yield them
                for example, prediction in zip(uncollate(batch), uncollate(predictions)):
                    bar.update(1)
                    yield example, prediction

            except Exception as ex:
                if self.skip_errors:
                    logging.warning(f"Error processing batch {batch_idx}", exc_info=ex)
                    bar.update(self.batch_size)
                else:
                    raise

        bar.close()

    def __iter__(self) -> Iterator[Tuple[Any, Dict[str, Any]]]:
        r"""Alias for :meth:`__call__`."""
        return self.__call__()
