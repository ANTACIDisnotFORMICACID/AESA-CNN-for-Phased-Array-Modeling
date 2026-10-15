"""V2.9 physics-layered scalar far-field models."""

from .amplitude_forward import AmplitudeOnlyForwardModel, V29AmplitudeForwardModel, model_from_config
from .angular_residual import AngularOperatorResidual
from .contraction import contract_scalar_operator_grid_iq, contract_scalar_operator_iq
from .device import IdentityDeviceModel
from .nominal_pattern import NominalElementPattern
from .port_coupling import StructuredPortCoupling, geometry_neighbour_mask
from .prediction import AmplitudePrediction
from .forward_model import PhysicsStructuredForwardModel
from .operator_residual import OperatorResidualCNN

__all__ = [
    "AmplitudeOnlyForwardModel",
    "AmplitudePrediction",
    "AngularOperatorResidual",
    "IdentityDeviceModel",
    "NominalElementPattern",
    "OperatorResidualCNN",
    "PhysicsStructuredForwardModel",
    "StructuredPortCoupling",
    "V29AmplitudeForwardModel",
    "contract_scalar_operator_grid_iq",
    "contract_scalar_operator_iq",
    "geometry_neighbour_mask",
    "model_from_config",
]
