"""Four-target max-min excitation optimization."""

from .problem import FourTargetProblem, FourTargetResult
from .projected_gradient import optimize_four_targets, project_command_iq
from .basis_verification import BasisVerification, nearest_basis_indices, verify_command_on_basis

__all__ = [
    "FourTargetProblem",
    "FourTargetResult",
    "optimize_four_targets",
    "project_command_iq",
    "BasisVerification",
    "nearest_basis_indices",
    "verify_command_on_basis",
]
