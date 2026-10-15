"""AESA-CNN V2.12 public contracts."""

from .audits import (
    GaugeInvariantSpectrum,
    NestedFiltrationSpectrum,
    NestedOperatorComparison,
    dephase_q_matrix,
    fourier_low_order_leakage,
    gauge_invariant_spectrum,
    nested_filtration_spectrum,
    nested_operator_comparison,
    operator_graph_roughness,
    rank_one_q_matrix,
)
from .basis import FiltrationSSHTransform, real_spherical_harmonics, ssh_mode_labels
from .codebook import MeasurementCodebook, build_baseline_codebook, recover_pair_entry
from .codebook_optimization import (
    CodebookOptimizationResult,
    information_metrics,
    optimize_supplemental_commands,
)
from .distillation import OperatorTarget, confidence_from_fisher, operator_distillation_loss
from .encoding import (
    DirectionFourierEncoder,
    FiltrationSSHEncoder,
    MatchedFourierEncoder,
    MatchedFourierBasis,
    RawSSHEncoder,
    ResidualizedFourierSSHEncoder,
    ResidualizedDirectionFourierEncoder,
    WeightedProjectionFit,
    fit_weighted_projection,
)
from .evaluation import (
    amplitude_metrics,
    angular_distance_quartile_report,
    cross_generalization_report,
    hemisphere_cells,
    nearest_training_angle,
    spatial_gap_partition,
    stratified_separation_subset,
)
from .inversion import (
    InversionResult,
    graph_synchronization_initialization,
    jacobian_diagnostics,
    measurement_jacobian,
    rank_spectrum_audit,
    recover_rank_one,
)
from .observation import ObservationKind, ReceiverObservation
from .operator import (
    BranchAblation,
    CoherentRank1Operator,
    CoherentRank1Prediction,
    operator_from_checkpoint,
)
from .stage1 import Stage1EncodingConfig, build_stage1_model
from .statistics import PairedComparison, holm_adjust, paired_comparison
from .training import (
    AmplitudeObjective,
    OperatorTrainConfig,
    OperatorTrainResult,
    amplitude_objective,
    train_operator_model,
)

__all__ = [
    "AmplitudeObjective",
    "CoherentRank1Operator",
    "CoherentRank1Prediction",
    "BranchAblation",
    "CodebookOptimizationResult",
    "DirectionFourierEncoder",
    "FiltrationSSHEncoder",
    "FiltrationSSHTransform",
    "GaugeInvariantSpectrum",
    "NestedFiltrationSpectrum",
    "NestedOperatorComparison",
    "InversionResult",
    "MatchedFourierEncoder",
    "MatchedFourierBasis",
    "MeasurementCodebook",
    "ObservationKind",
    "OperatorTrainConfig",
    "OperatorTrainResult",
    "OperatorTarget",
    "PairedComparison",
    "RawSSHEncoder",
    "ReceiverObservation",
    "ResidualizedFourierSSHEncoder",
    "ResidualizedDirectionFourierEncoder",
    "WeightedProjectionFit",
    "Stage1EncodingConfig",
    "amplitude_metrics",
    "angular_distance_quartile_report",
    "amplitude_objective",
    "build_baseline_codebook",
    "build_stage1_model",
    "confidence_from_fisher",
    "cross_generalization_report",
    "dephase_q_matrix",
    "fourier_low_order_leakage",
    "fit_weighted_projection",
    "gauge_invariant_spectrum",
    "hemisphere_cells",
    "graph_synchronization_initialization",
    "holm_adjust",
    "jacobian_diagnostics",
    "measurement_jacobian",
    "information_metrics",
    "nearest_training_angle",
    "nested_filtration_spectrum",
    "nested_operator_comparison",
    "operator_from_checkpoint",
    "operator_graph_roughness",
    "operator_distillation_loss",
    "optimize_supplemental_commands",
    "paired_comparison",
    "rank_one_q_matrix",
    "rank_spectrum_audit",
    "real_spherical_harmonics",
    "recover_pair_entry",
    "recover_rank_one",
    "ssh_mode_labels",
    "spatial_gap_partition",
    "stratified_separation_subset",
    "train_operator_model",
]
