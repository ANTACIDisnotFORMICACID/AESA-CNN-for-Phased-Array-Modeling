"""Physics-structured operator correction and forward models."""

from .forward_model import PhysicsStructuredForwardModel
from .operator_residual import OperatorResidualCNN
from .amplitude_forward import (
    AmplitudeOnlyForwardModel,
    AmplitudeOperatorResidualCNN,
    AmplitudePrediction,
    contract_scalar_operator_grid_iq,
)

__all__ = [
    "OperatorResidualCNN",
    "PhysicsStructuredForwardModel",
    "AmplitudeOnlyForwardModel",
    "AmplitudeOperatorResidualCNN",
    "AmplitudePrediction",
    "contract_scalar_operator_grid_iq",
]
