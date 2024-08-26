from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Final,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    cast,
)

import numpy as np
import pandas as pd
import pydicom
import torch
from dicom_utils.container import DicomImageFileRecord, MammogramFileRecord
from dicom_utils.dicom_factory import CompleteMammographyStudyFactory, DicomFactory
from dicom_utils.volume import ReduceVolume
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from torch_dicom.preprocessing import PreprocessingPipeline
from torch_dicom.preprocessing.datamodule import PreprocessedDataModule

from .preprocessing import require_datamodule


# Default filenames for test factory
DEFAULT_MANIFEST_FILENAME: Final = "manifest.csv"
DEFAULT_ANNOTATION_MANIFEST_FILENAME: Final = "annotation_manifest.csv"
DEFAULT_TRACE_MANIFEST_FILENAME: Final = "trace_manifest.csv"
DEFAULT_TRACE_EXTRA_KEYS: Final = ["trait", "types"]


if TYPE_CHECKING:
    from torch_dicom.preprocessing.datamodule import PreprocessedDataModule
else:
    try:
        from torch_dicom.preprocessing.datamodule import PreprocessedDataModule
    except ImportError:
        PreprocessedDataModule = None


M = TypeVar("M", bound=PreprocessedDataModule)


def choose_modulus(val: int, choices: Sequence[Any]) -> Any:
    """
    Chooses an element from a sequence based on the modulus of the provided value.

    Args:
        val: The value to calculate the modulus of.
        choices: The sequence of choices to select from.

    Returns:
        The chosen element from the sequence.
    """
    return choices[val % len(choices)]


def create_random_box(seed: int, canvas_size: Tuple[int, int]) -> BoundingBoxes:
    """
    Creates a random bounding box within a given canvas size.

    Args:
        seed: The seed for the random number generator.
        canvas_size: The size of the canvas (height, width) within which the bounding box will be created.

    Returns:
        A BoundingBoxes object representing the created bounding box.
    """
    H, W = canvas_size
    np.random.seed(seed)
    x1 = np.random.randint(0, H - 1, 1)
    y1 = np.random.randint(0, W - 1, 1)
    x2 = np.random.randint(x1 + 1, H, 1)
    y2 = np.random.randint(y1 + 1, W, 1)
    return BoundingBoxes(
        torch.from_numpy(np.array([x1, y1, x2, y2])).flatten(),
        format=BoundingBoxFormat.XYXY,
        canvas_size=canvas_size,
    )


@dataclass
class DicomTestFactory:
    r"""Factory for setting up DICOMs and preprocessed data for testing.

    Args:
        root: Root directory to save DICOMs and preprocessed data.
        dicom_size: Size of the DICOM images to generate.
        num_studies: Number of studies to generate.
        seed: Seed to use when generating DICOMs.
    """

    root: Path
    dicom_size: Tuple[int, int] = (64, 32)
    num_studies: int = 3
    seed: int = 42

    dicom_factory: ClassVar[Type[DicomFactory]] = DicomFactory

    def __post_init__(self):
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)  # pragma: no cover

    def iterate_dicoms(self, **kwargs) -> Iterator[pydicom.Dataset]:
        """
        Iterates over the DICOMs generated by the factory.

        Keyword Args:
            Forwarded to the :class:`DicomFactory` constructor.

        Yields:
            pydicom.Dataset: The next DICOM dataset in the sequence.
        """
        H, W = self.dicom_size
        factory = self.dicom_factory(Rows=H, Columns=W, **kwargs)

        for i in range(self.num_studies):
            dicom = factory(
                seed=self.seed + i,
                PatientID=f"patient-{i}",
                StudyInstanceUID=f"study-{i}",
                SOPInstanceUID=f"image-{i}",
            )
            yield dicom

    def create_dicom_files(self, dest: Path) -> List[Path]:
        """
        Creates DICOM files for testing.

        Args:
            dest: The destination path where the DICOM files will be saved.

        Returns:
            A list of paths to the created DICOM files.
        """
        if not dest.is_dir():
            raise NotADirectoryError(dest)  # pragma: no cover

        paths = []
        for dicom in self.iterate_dicoms():
            path = dest / f"{dicom.SOPInstanceUID}.dcm"
            dicom.save_as(path)
            paths.append(path)
        return paths

    def create_preprocessed_data(self, dest: Path, dicom_files: List[Path], **kwargs) -> List[Path]:
        """
        Creates preprocessed data from DICOM files. Uses a :class:`ReduceVolume` volume handler by default.

        Args:
            dest: The destination path where the preprocessed data will be saved.
            dicom_files: The list of DICOM files to be preprocessed.

        Keyword Args:
            Forwards any keyword arguments to the :class:`PreprocessingPipeline` constructor.

        Returns:
            A list of paths to the created preprocessed data files.
        """
        if not dest.is_dir():
            raise NotADirectoryError(dest)  # pragma: no cover

        kwargs.setdefault("volume_handler", ReduceVolume())
        kwargs.setdefault("use_bar", False)
        kwargs.setdefault("num_workers", 0)
        kwargs.setdefault("prefetch_factor", None)
        pipeline = PreprocessingPipeline(dicom_files, **kwargs)
        pipeline(dest)
        return list(dest.rglob("*.png"))

    def create_manifest(self, dicom_files: List[Path]) -> pd.DataFrame:
        """
        Creates a manifest DataFrame from a list of DICOM files. The DataFrame will contain metadata
        about each DICOM file such as the path, SOPInstanceUID, StudyInstanceUID, and Patient.

        Args:
            dicom_files: The list of DICOM files to create a manifest for.

        Returns:
            A DataFrame containing the path, SOPInstanceUID, StudyInstanceUID, and Patient for each DICOM file.
            The SOPInstanceUID is used as the index.
        """
        records = cast(List[DicomImageFileRecord], [DicomImageFileRecord.from_file(path) for path in dicom_files])
        paths = [record.path for record in records]
        sop_uids = [record.SOPInstanceUID for record in records]
        study_uids = [record.StudyInstanceUID for record in records]
        patient_ids = [record.PatientID for record in records]
        df = pd.DataFrame(
            {
                "path": paths,
                "SOPInstanceUID": sop_uids,
                "StudyInstanceUID": study_uids,
                "Patient": patient_ids,
            }
        )
        df.set_index("SOPInstanceUID", inplace=True)
        return df

    def create_annotation_manifest(self, dicom_files: List[Path]) -> pd.DataFrame:
        """
        Creates an annotation manifest DataFrame from a list of DICOM files. The DataFrame will contain metadata
        about each image beyond what is contained in the DICOM files, such as annotator provided labels.

        Args:
            dicom_files: The list of DICOM files to create annotations for.

        Returns:
            A DataFrame containing the SOPInstanceUID and trait for each DICOM file.
            The SOPInstanceUID is used as the index.
        """
        records = cast(List[DicomImageFileRecord], [DicomImageFileRecord.from_file(path) for path in dicom_files])

        # Sort SOPInstanceUIDs so we can generate consistent annotations
        sop_uids = cast(List[str], [record.SOPInstanceUID for record in records])
        sop_uids = sorted(sop_uids)

        traits = [choose_modulus(i, ("malignant", "benign", "unknown")) for i in range(len(sop_uids))]

        df = pd.DataFrame(
            {
                "SOPInstanceUID": sop_uids,
                "trait": traits,
            }
        )
        df.set_index("SOPInstanceUID", inplace=True)
        return df

    def create_trace_manifest(self, dicom_files: List[Path]) -> pd.DataFrame:
        """
        Creates a trace manifest DataFrame from a list of DICOM files. The DataFrame will contain metadata
        about bounding box traces for each image.

        Args:
            dicom_files: The list of DICOM files to create traces for.

        Returns:
            A DataFrame containing the SOPInstanceUID, coordinates of the trace, trait of the trace, and type of the trace for each DICOM file.
            The SOPInstanceUID is used as the index. Rows with no trace are dropped.
        """
        records = cast(List[DicomImageFileRecord], [DicomImageFileRecord.from_file(path) for path in dicom_files])

        # Sort SOPInstanceUIDs so we can generate consistent annotations
        sop_uids = cast(List[str], [record.SOPInstanceUID for record in records])
        sop_uids = sorted(sop_uids)

        # Choose properties of the traces
        has_trace = [choose_modulus(i, (True, False)) for i in range(len(sop_uids))]
        trace_traits = [
            choose_modulus(i, ("malignant", "benign", "unknown")) if has_trace[i] else None
            for i in range(len(sop_uids))
        ]
        trace_types = [choose_modulus(i, ("mass", "asymmetry")) if has_trace[i] else None for i in range(len(sop_uids))]
        trace_coords = [create_random_box(i, self.dicom_size) if has_trace[i] else None for i in range(len(sop_uids))]

        # Create the manifest
        coord_dict = {
            f"{coord}": [int(box.flatten()[i].item()) if box is not None else None for box in trace_coords]
            for i, coord in enumerate(["x1", "y1", "x2", "y2"])
        }
        df = pd.DataFrame(
            {
                "SOPInstanceUID": sop_uids,
                **coord_dict,
                "trait": trace_traits,
                "types": trace_types,
            }
        )
        df.set_index("SOPInstanceUID", inplace=True)

        # Drop rows with no trace
        df = df[df["trait"].apply(lambda t: t is not None)]

        return df

    def __call__(
        self,
        setup: bool = True,
        metadata_filenames: Dict[str, str] = {
            "manifest": DEFAULT_MANIFEST_FILENAME,
            "annotation": DEFAULT_ANNOTATION_MANIFEST_FILENAME,
        },
        boxes_filename: Optional[str] = DEFAULT_TRACE_MANIFEST_FILENAME,
        boxes_extra_keys: Iterable[str] = DEFAULT_TRACE_EXTRA_KEYS,
        datamodule_class: Type[M] = PreprocessedDataModule,
        **kwargs,
    ) -> M:
        r"""Creates a :class:`PreprocessedDataModule` using the factory.

        Args:
            setup: Whether to setup the created :class:`PreprocessedDataModule` instance.
                If ``True``, the :func:`setup` hook will be called for the ``fit`` and ``test`` stages.
            metadata_filenames: Filenames for the manifest and annotation manifest CSV files.
            boxes_filename: Filename for the trace manifest CSV file.
            boxes_extra_keys: Extra keys to include in the trace manifest CSV file.
            datamodule_class: The specific :class:`PreprocessedDataModule` class to use.

        Keyword Args:
            Forwarded to the :class:`PreprocessedDataModule` constructor.

        Returns:
            A :class:`PreprocessedDataModule` instance.
        """
        # PreprocessedDataModule is an optional dependency. Validate that it is installed.
        require_datamodule()

        # Create the DICOM files
        dicom_root = self.root / "dicoms"
        dicom_root.mkdir()
        dicom_files = self.create_dicom_files(dicom_root)

        # Preprocess them
        preprocessed_root = self.root / "preprocessed"
        preprocessed_root.mkdir()
        self.create_preprocessed_data(preprocessed_root, dicom_files)

        # Create the manifests
        manifest = self.create_manifest(dicom_files)
        annotation_manifest = self.create_annotation_manifest(dicom_files)
        trace_manifest = self.create_trace_manifest(dicom_files)
        manifest.to_csv(preprocessed_root / DEFAULT_MANIFEST_FILENAME)
        annotation_manifest.to_csv(preprocessed_root / DEFAULT_ANNOTATION_MANIFEST_FILENAME)
        trace_manifest.to_csv(preprocessed_root / DEFAULT_TRACE_MANIFEST_FILENAME)

        # Create the data module
        dm = datamodule_class(
            train_inputs=preprocessed_root,
            val_inputs=preprocessed_root,
            test_inputs=preprocessed_root,
            metadata_filenames=metadata_filenames,
            boxes_filename=boxes_filename,
            boxes_extra_keys=boxes_extra_keys,
            **kwargs,
        )

        # Possibly set up
        if setup:
            dm.setup(stage="fit")
            dm.setup(stage="test")
        return cast(M, dm)


@dataclass
class MammogramTestFactory(DicomTestFactory):
    r"""Factory for setting up mammographic DICOMs and preprocessed data for testing.

    Args:
        root: Root directory to save DICOMs and preprocessed data.
        dicom_size: Size of the DICOM images to generate.
        num_studies: Number of studies to generate.
        seed: Seed to use when generating DICOMs.
    """

    dicom_factory: ClassVar[Type[CompleteMammographyStudyFactory]] = CompleteMammographyStudyFactory

    def iterate_dicoms(self, **kwargs) -> Iterator[pydicom.Dataset]:
        H, W = self.dicom_size
        for i in range(self.num_studies):
            # We need to override this method because the study factory returns a list of DICOMs
            # for each study, rather than a single DICOM.
            factory = self.dicom_factory(
                PatientID=f"patient-{i}",
                StudyInstanceUID=f"study-{i}",
                Rows=H,
                Columns=W,
                seed=self.seed + i,
                **kwargs,
            )
            yield from factory()

    def create_manifest(self, dicom_files: List[Path]) -> pd.DataFrame:
        # Create the manifest
        manifest = super().create_manifest(dicom_files)
        records = cast(List[MammogramFileRecord], [MammogramFileRecord.from_file(path) for path in dicom_files])
        records.sort(key=lambda rec: rec.SOPInstanceUID)

        # Add mammography specific columns
        new_columns = pd.DataFrame(
            {
                "Laterality": [rec.laterality.simple_name for rec in records],
                "ViewPosition": [rec.laterality.simple_name for rec in records],
                "standard_view": [rec.is_standard_mammo_view for rec in records],
            },
            index=manifest.index,
        )
        manifest = manifest.join(new_columns)

        return manifest

    def create_annotation_manifest(self, dicom_files: List[Path]) -> pd.DataFrame:
        """
        Creates an annotation manifest DataFrame from a list of DICOM files. The DataFrame will contain metadata
        about each image beyond what is contained in the DICOM files, such as annotator provided labels.

        Args:
            dicom_files: The list of DICOM files to create annotations for.

        Returns:
            A DataFrame containing the SOPInstanceUID and trait for each DICOM file.
            The SOPInstanceUID is used as the index.
        """
        # Create the manifest
        manifest = super().create_annotation_manifest(dicom_files)
        records = cast(List[MammogramFileRecord], [MammogramFileRecord.from_file(path) for path in dicom_files])
        records.sort(key=lambda rec: rec.SOPInstanceUID)

        # Add mammography specific columns
        # TODO: We aren't ensuring consistency for certain fields. A study may have multiple density values
        # or may have images with impossible trait / trait laterality combinations. Is this something we want to
        # fix?
        new_columns = pd.DataFrame(
            {
                "density": [choose_modulus(i, ("a", "b", "c", "d", "unknown")) for i in range(len(records))],
                "trait_laterality": [choose_modulus(i, ("left", "right", "bilateral")) for i in range(len(records))],
            },
            index=manifest.index,
        )
        manifest = manifest.join(new_columns)
        return manifest
