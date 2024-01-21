from dataclasses import dataclass, is_dataclass
from typing import Any, Dict, List, TypeVar

import torch

from ..preprocessing import datamodule_available
from .pipeline import InferencePipeline


B = TypeVar("B", bound=Dict[str, Any])


def _create_exception(name: str) -> ImportError:
    return ImportError(
        f"{name} requires pytorch-lightning. " "Please install it or install the `torch-dicom[datamodule]` extra."
    )


try:
    from pytorch_lightning import LightningModule
except ImportError as e:
    raise _create_exception("LightningInferencePipeline") from e


@dataclass
class LightningInferencePipeline(InferencePipeline):
    def __post_init__(self):
        if not datamodule_available():
            raise _create_exception(self.__class__.__name__)
        super().__post_init__()

    def infer_with_model(self, model: LightningModule, batch: Any, batch_idx: int) -> Dict[str, Any]:
        return model.predict_step(batch, batch_idx)

    def transfer_batch_to_device(self, models: List[LightningModule], batch: B, device: torch.device) -> B:
        assert len(models), "No models available."
        model = next(iter(models))

        # Lightning will fail the transfer if any batch element is a frozen dataclass.
        # We will find any such elements and pop them from the batch, transfer the batch,
        # and then reattach the frozen dataclasses.
        # We assume that batched frozen dataclasses will be lists of frozen dataclasses, and
        # we don't recurse to find nested frozen dataclasses.
        frozen_dataclasses = {
            k: v for k, v in batch.items() if is_dataclass(proto := v[0]) and proto.__dataclass_params__.frozen
        }
        for k in frozen_dataclasses.keys():
            batch.pop(k)
        batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)
        batch.update(frozen_dataclasses)
        return batch
