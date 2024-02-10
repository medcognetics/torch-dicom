from functools import partial
from typing import List, Sized

import numpy as np
import pandas as pd
import pytest
import yaml
from deep_helpers.structs import Mode
from dicom_utils.volume import ReduceVolume
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms.v2 import Resize

from torch_dicom.datasets import ImagePathDataset, MetadataDatasetWrapper, PreprocessingConfigMetadata
from torch_dicom.datasets.sampler import BatchComplementSampler, WeightedCSVSampler
from torch_dicom.preprocessing.datamodule import PreprocessedPNGDataModule
from torch_dicom.preprocessing.pipeline import OutputFormat, PreprocessingPipeline


@pytest.fixture(scope="session")
def sop_uids(dicoms):
    return [dcm.SOPInstanceUID for dcm in dicoms]


@pytest.fixture(scope="session")
def patient_ids(dicoms):
    return [dcm.PatientID for dcm in dicoms]


@pytest.fixture(scope="session")
def preprocessed_data(tmp_path_factory, files):
    root = tmp_path_factory.mktemp("preprocessed")
    pipeline = PreprocessingPipeline(files, output_format=OutputFormat.PNG, volume_handler=ReduceVolume())
    pipeline(root)
    return root


@pytest.fixture(scope="session")
def roi_csv(preprocessed_data, sop_uids):
    sop_uids_with_boxes = sop_uids[::2]
    csv_path = preprocessed_data / "roi.csv"
    data = {
        "SOPInstanceUID": sop_uids_with_boxes,
        "x1": [0] * len(sop_uids_with_boxes),
        "y1": [0] * len(sop_uids_with_boxes),
        "x2": [5] * len(sop_uids_with_boxes),
        "y2": [5] * len(sop_uids_with_boxes),
        "trait": ["malignant"] * len(sop_uids_with_boxes),
        "types": ["mass"] * len(sop_uids_with_boxes),
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="session")
def manifest_csv(preprocessed_data, patient_ids, sop_uids):
    csv_path = preprocessed_data / "manifest.csv"
    data = {
        "Patient": patient_ids,
        "SOPInstanceUID": sop_uids,
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="session")
def annotation_csv(preprocessed_data, sop_uids):
    csv_path = preprocessed_data / "annotation.csv"
    data = {
        "density": np.random.choice(["a", "b"], size=len(sop_uids)),
        "SOPInstanceUID": sop_uids,
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path


class PatientBatchSampler(BatchComplementSampler):
    @property
    def complement_size(self):
        return 2

    def find_complement(self, idx: int) -> List[int]:
        return self.select_random_complement(idx, "Patient")


class ModuleWithCustomSampler(PreprocessedPNGDataModule):
    def create_sampler(self, dataset, example_paths, root, mode):
        if mode == Mode.TRAIN:
            return WeightedCSVSampler(
                root / "annotation.csv",
                example_paths,
                "density",
                {"a": 0.5, "b": 0.5},
            )
        else:
            return super().create_sampler(dataset, example_paths, root, mode)


class ModuleWithCustomBatchSampler(PreprocessedPNGDataModule):
    def create_batch_sampler(self, dataset, sampler, example_paths, roots, mode):
        if mode == Mode.TRAIN:
            return PatientBatchSampler(
                sampler,
                self.batch_size,
                [root / "manifest.csv" for root in roots],
                example_paths,
            )
        else:
            return


class TestPreprocessedPNGDataModule:
    @pytest.fixture(scope="class")
    def datamodule_with_metadata(self, manifest_csv, annotation_csv, roi_csv):
        boxes_filename = roi_csv.name
        metadata_filenames = {
            "manifest": manifest_csv.name,
            "annotation": annotation_csv.name,
        }
        return partial(PreprocessedPNGDataModule, boxes_filename=boxes_filename, metadata_filenames=metadata_filenames)

    def test_create_dataset_no_extra_metadata(self, preprocessed_data):
        dm = PreprocessedPNGDataModule()
        output = dm.create_dataset(preprocessed_data, Mode.TRAIN)
        assert isinstance(output, PreprocessingConfigMetadata)
        assert isinstance(output.dataset, ImagePathDataset)
        example = output[0]
        assert "preprocessing" in example

    def test_create_dataset_with_csvs(self, preprocessed_data, datamodule_with_metadata):
        dm = datamodule_with_metadata()
        output = dm.create_dataset(preprocessed_data, Mode.TRAIN)
        assert isinstance(output, MetadataDatasetWrapper)
        example = output[0]
        assert "preprocessing" in example
        assert "bounding_boxes" in example
        assert "manifest" in example
        assert "annotation" in example

    @pytest.mark.parametrize(
        "mode,exp",
        [
            (Mode.TRAIN, RandomSampler),
            (Mode.VAL, SequentialSampler),
            (Mode.TEST, SequentialSampler),
        ],
    )
    def test_create_sampler(self, preprocessed_data, mode, exp):
        dm = PreprocessedPNGDataModule()
        ds = ImagePathDataset(preprocessed_data.rglob("*.png"))
        output = dm.create_sampler(ds, list(ds.files), preprocessed_data, mode)
        assert isinstance(output, exp)
        assert isinstance(output, Sized) and len(output) == len(ds)

    @pytest.mark.parametrize("multi_inputs", [False, True])
    @pytest.mark.parametrize(
        "stage,batch_size,fn",
        [
            ("fit", 4, "train_dataloader"),
            ("fit", 2, "val_dataloader"),
            ("test", 1, "test_dataloader"),
        ],
    )
    def test_setup(self, preprocessed_data, datamodule_with_metadata, stage, batch_size, fn, multi_inputs):
        preprocessed_data = [preprocessed_data] * 2 if multi_inputs else preprocessed_data
        module: PreprocessedPNGDataModule = datamodule_with_metadata(
            preprocessed_data,
            preprocessed_data,
            preprocessed_data,
            batch_size=batch_size,
        )
        module.setup(stage=str(stage))
        loader = getattr(module, fn)()
        assert isinstance(loader, DataLoader)
        assert len(next(iter(loader))["img"]) == batch_size

        # Check shuffling and train sampler
        if fn == "train_dataloader":
            b1 = next(iter(loader))
            b2 = next(iter(loader))
            assert b1["path"] != b2["path"]
            assert isinstance(module.train_sampler, RandomSampler)

    def test_weighted_csv_sampler(self, preprocessed_data, manifest_csv, annotation_csv, roi_csv):
        boxes_filename = roi_csv.name
        metadata_filenames = {
            "manifest": manifest_csv.name,
            "annotation": annotation_csv.name,
        }
        module = ModuleWithCustomSampler(
            preprocessed_data,
            boxes_filename=boxes_filename,
            metadata_filenames=metadata_filenames,
        )
        module.setup("fit")
        loader = module.train_dataloader()
        assert isinstance(next(iter(loader)), dict)

    def test_batch_complement_sampler(self, preprocessed_data, manifest_csv, annotation_csv, roi_csv):
        boxes_filename = roi_csv.name
        metadata_filenames = {
            "manifest": manifest_csv.name,
            "annotation": annotation_csv.name,
        }
        module = ModuleWithCustomBatchSampler(
            preprocessed_data,
            boxes_filename=boxes_filename,
            metadata_filenames=metadata_filenames,
        )
        module.setup("fit")
        loader = module.train_dataloader()
        assert isinstance(next(iter(loader)), dict)

    @pytest.mark.parametrize(
        "train_gpu_transforms",
        [
            None,
            Resize(32, antialias=True),
        ],
    )
    def test_gpu_transforms(self, mocker, preprocessed_data, datamodule_with_metadata, train_gpu_transforms):
        module: PreprocessedPNGDataModule = datamodule_with_metadata(
            preprocessed_data,
            preprocessed_data,
            preprocessed_data,
            train_gpu_transforms=train_gpu_transforms,
        )
        spy = mocker.spy(train_gpu_transforms, "forward") if train_gpu_transforms else None
        module.setup(stage="fit")
        module.trainer = mocker.MagicMock()
        module.trainer.training = True

        batch = next(iter(module.train_dataloader()))
        batch = module.on_after_batch_transfer(batch, 0)
        if spy is not None:
            spy.assert_called_once()

    def test_jsonargparse(self, tmp_path, preprocessed_data, manifest_csv, annotation_csv, roi_csv):
        jsonargparse = pytest.importorskip("jsonargparse")
        # Prepare config and save as YAML
        transform = {
            "class_path": "torchvision.transforms.v2.Compose",
            "init_args": {
                "transforms": [
                    {
                        "class_path": "torchvision.transforms.v2.Resize",
                        "init_args": {
                            "size": [512, 384],
                        },
                    }
                ],
            },
        }
        config = {
            "data": {
                "class_path": "torch_dicom.preprocessing.datamodule.PreprocessedPNGDataModule",
                "init_args": {
                    "train_inputs": [str(preprocessed_data)] * 2,
                    "val_inputs": str(preprocessed_data),
                    "test_inputs": str(preprocessed_data),
                    "batch_size": 2,
                    "train_transforms": transform,
                    "train_gpu_transforms": transform,
                    "val_transforms": transform,
                    "test_transforms": transform,
                    "boxes_filename": roi_csv.name,
                    "metadata_filenames": {
                        "manifest": manifest_csv.name,
                        "annotation": annotation_csv.name,
                    },
                    "train_sopuid_exclusions": ["1.2.3"],
                },
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Parse config
        parser = jsonargparse.ArgumentParser()
        parser.add_argument("--config", action=jsonargparse.ActionConfigFile)
        parser.add_subclass_arguments(PreprocessedPNGDataModule, "data")
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])
        cfg = parser.parse_args(["--config", str(config_path)])
        cfg = parser.instantiate_classes(cfg)
        assert isinstance(cfg.data, PreprocessedPNGDataModule)
        assert cfg.data.batch_size == config["data"]["init_args"]["batch_size"]
        cfg.data.setup("fit")

    @pytest.mark.parametrize("from_file", [False, True])
    @pytest.mark.parametrize(
        "stage,fn",
        [
            ("fit", "train_dataloader"),
            ("fit", "val_dataloader"),
            ("test", "test_dataloader"),
        ],
    )
    def test_sopuid_exclusions(self, preprocessed_data, datamodule_with_metadata, stage, from_file, tmp_path, fn):
        # Prepare a filter that only passes one example
        sop_uids = [f.stem for f in preprocessed_data.rglob("*.png")]
        if from_file:
            sopuid_exclusions = tmp_path / "sopuid_exclusions.txt"
            with open(sopuid_exclusions, "w") as f:
                f.write("\n".join(sop_uids[:-1]))
        else:
            sopuid_exclusions = sop_uids[:-1]

        # Check that the corresponding dataloader only has one example
        module: PreprocessedPNGDataModule = datamodule_with_metadata(
            preprocessed_data,
            preprocessed_data,
            preprocessed_data,
            train_sopuid_exclusions=sopuid_exclusions,
            val_sopuid_exclusions=sopuid_exclusions,
            test_sopuid_exclusions=sopuid_exclusions,
        )
        module.setup(stage=str(stage))
        loader = getattr(module, fn)()
        assert isinstance(loader, DataLoader)
        assert len(loader) == 1
