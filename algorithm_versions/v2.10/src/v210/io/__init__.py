"""Canonical HFSS/measurement-compatible data interfaces."""

from .dataset_store import BasisDataset, DatasetManifest, load_basis_dataset, save_basis_dataset
from .hfss_ffd import HfssFfdExport, load_hfss_ffd_basis
from .touchstone import TouchstoneNetwork, load_touchstone, s_matrix_to_grid
from .amplitude_dataset import (
    AMPLITUDE_ONLY_TRAINING_POLICY,
    AmplitudeDataset,
    AmplitudeManifest,
    assert_amplitude_only_training_contract,
    load_amplitude_dataset,
    save_amplitude_dataset,
)
from .amplitude_split import (
    AmplitudeDataSplit,
    amplitude_dataset_fingerprint,
    assert_split_matches_dataset,
    load_amplitude_split,
    save_amplitude_split,
)
from .amplitude_synthesis import (
    RECEIVER_CHANNELS,
    amplitude_for_commands_from_basis,
    canonical_spherical_query_mask,
    generate_command_codebook,
    synthesize_amplitude_observations,
)

__all__ = [
    "BasisDataset",
    "DatasetManifest",
    "HfssFfdExport",
    "load_hfss_ffd_basis",
    "load_basis_dataset",
    "save_basis_dataset",
    "TouchstoneNetwork",
    "load_touchstone",
    "s_matrix_to_grid",
    "AmplitudeDataset",
    "AmplitudeManifest",
    "AMPLITUDE_ONLY_TRAINING_POLICY",
    "assert_amplitude_only_training_contract",
    "AmplitudeDataSplit",
    "amplitude_dataset_fingerprint",
    "assert_split_matches_dataset",
    "load_amplitude_split",
    "save_amplitude_split",
    "load_amplitude_dataset",
    "save_amplitude_dataset",
    "RECEIVER_CHANNELS",
    "amplitude_for_commands_from_basis",
    "canonical_spherical_query_mask",
    "generate_command_codebook",
    "synthesize_amplitude_observations",
]
