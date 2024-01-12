import torch
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union, cast, TYPE_CHECKING, ForwardRef, Union, Sequence
from pathlib import Path
from torch import Tensor
from torch.utils.data import DataLoader, ChainDataset, ConcatDataset, Dataset
from torch_dicom.datasets import (
    TensorInput, 
    TensorPathInput, 
    DicomInput, 
    DicomPathInput,
    DicomPathDataset,
    ImageInput,
    ImagePathInput,
    ImagePathDataset,
    collate_fn,
)
from torch_dicom.datasets.dicom import Transform
from dicom_utils.dicom import Dicom
from dicom_utils.volume import VolumeHandler, ReduceVolume
from PIL import Image

if TYPE_CHECKING:
    import pytorch_lightning as pl
    Model = pl.LightningModule
else:
    Model = FowardRef("pytorch_lightning.LightningModule")


@dataclass
class InferencePipeline:
    # Known input sources
    dicom_paths: Iterable[Path] = field(default_factory=list)
    dicoms: Iterable[Dicom] = field(default_factory=list)
    tensor_paths: Iterable[Path] = field(default_factory=list)
    tensors: Iterable[Tensor] = field(default_factory=list)
    image_paths: Iterable[Path] = field(default_factory=list)
    images: Iterable[Image.Image] = field(default_factory=list)

    # Custom datasets
    custom_inputs: Sequence[Dataset] = field(default_factory=list)

    batch_size: int = 1
    transform: Optional[Transform] = None
    enumerate_inputs: bool = False
    skip_errors: bool = True
    volume_handler: VolumeHandler = ReduceVolume()
    dataloader_kwargs: Dict[str, Any] = field(default_factory=dict)

    _models: List[Model] = field(default_factory=list, init=False)

    def add_model(self, model: Model) -> None:
        if not isinstance(model, Model):
            raise TypeError(f"Expected a {Model.__name__} but got {type(model).__name__}.")
        self._models.append(model)

    def prepare_dataset(self) -> Union[ChainDataset, ConcatDataset]:
        # Build the datasets
        datasets: List[Dataset] = [
            *self._prepare_dicom_datasets(),
            *self._prepare_tensor_datasets(),
            *self._prepare_image_datasets(),
            *self.custom_inputs,
        ]

        # If enumerate_inputs is True and every dataset is a Sequence, we will return a ConcatDataset
        #
        # If enumerate_inputs is False or inputs were given that don't have an associated map-style Dataset,
        # we will return a ChainDataset
        return ConcatDataset(datasets) if self.enumerate_inputs and all(isinstance(ds, Sequence) for ds in datasets) else ChainDataset(datasets)

    def _prepare_dicom_datasets(self) -> Iterator[Dataset]:
        if self.dicom_paths:
            ds = (DicomPathDataset if self.enumerate_inputs else DicomPathInput)(
                self.dicom_paths, 
                skip_errors=self.skip_errors, 
                volume_handler=self.volume_handler, 
                transform=self.transform,
            )
            yield ds

        if self.dicoms:
            ds = DicomInput(self.dicoms, skip_errors=self.skip_errors, volume_handler=self.volume_handler, transform=self.transform)
            yield ds

    def _prepare_tensor_datasets(self) -> Iterator[Dataset]:
        if self.tensor_paths:
            ds = TensorPathInput(self.tensor_paths, skip_errors=self.skip_errors, transform=self.transform)
            yield ds

        if self.tensors:
            ds = TensorInput(self.tensors, skip_errors=self.skip_errors, transform=self.transform)
            yield ds

    def _prepare_image_datasets(self) -> Iterator[Dataset]:
        if self.image_paths:
            ds = (ImagePathDataset if self.enumerate_inputs else ImagePathInput)(
                self.image_paths, 
                skip_errors=self.skip_errors, 
                transform=self.transform,
            )
            yield ds

        if self.images:
            ds = ImageInput(self.images, skip_errors=self.skip_errors, transform=self.transform)
            yield ds

    def prepare_dataloader(self, dataset: Dataset) -> DataLoader:
        # Set defaults
        dataloader_kwargs = copy(self.dataloader_kwargs)
        dataloader_kwargs.setdefault("collate_fn", collate_fn)

        return DataLoader(dataset, batch_size=self.batch_size, **dataloader_kwargs)



@torch.no_grad()
def predict(
    checkpoint: Optional[Path] = None,
    dicom_paths: Iterable[Path] = [],
    dicoms: Iterable[Dicom] = [],
    tensor_paths: Iterable[Path] = [],
    tensors: Iterable[Tensor] = [],
    img_size: Tuple[int, int] = (2048, 1536),
    batch_size: int = 1,
    device: torch.device = torch.device("cpu"),
    skip_errors: bool = True,
) -> Iterator[Dict[str, Any]]:
    r"""Runs the inference loop.

    Args:
        checkpoint: Path to the checkpoint to load. If `None`, read from environment variable.

        dicom_paths: A list of paths to DICOM files.

        dicoms: A list of DICOM objects.

        tensor_paths: A list of paths to tensors.

        tensors: A list of tensors.

        img_size: The size of the input images.

        batch_size: Batch size.

        device: Device to use for inference.

        skip_errors: If ``True``, skip files that cannot be read.

        dataloader_kwargs: Keyword arguments to pass to the DataLoader.

        volume_handler: A :class:`VolumeHandler` to use for handling 3D inputs.

    Returns:
        Iterator over the predictions.
    """
    # load the model
    model = BreastTriageTask.create(checkpoint).to(device)

    # prepare the transform
    # NOTE: We specify resize here and not in the DicomInput.img_size parameter.
    # This is because we want to crop the image and resize with aspect ratio preserved.
    crop = MinMaxCrop()
    resize = Resize(size=img_size)
    transform = Compose(
        [
            crop,
            resize,
        ]
    )

    # create the datapipes
    dicom_file_ds = DicomPathInput(
        dicom_paths, skip_errors=skip_errors, volume_handler=volume_handler, transform=transform
    )
    dicom_object_ds = DicomInput(dicoms, skip_errors=skip_errors, volume_handler=volume_handler, transform=transform)
    tensor_file_ds = TensorPathInput(tensor_paths, skip_errors=skip_errors, transform=transform)
    tensor_object_ds = TensorInput(tensors, skip_errors=skip_errors, transform=transform)
    ds = ChainDataset([dicom_file_ds, dicom_object_ds, tensor_file_ds, tensor_object_ds])

    # create the dataloader
    dataloader_kwargs.setdefault("collate_fn", collate_fn)
    dl = DataLoader(ds, batch_size=batch_size, **dataloader_kwargs)

    for batch_idx, batch in enumerate(dl):
        batch = model.transfer_batch_to_device(batch, device, 0)
        prediction = model.predict_step(batch, batch_idx)
        yield from prediction
