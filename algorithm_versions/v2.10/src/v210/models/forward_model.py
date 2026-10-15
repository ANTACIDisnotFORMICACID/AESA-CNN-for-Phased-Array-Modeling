"""End-to-end command-to-field model with an explicit linear contraction."""

from __future__ import annotations

import torch
from torch import nn

from ..complex_ops import contract_operator_iq, field_power
from ..contracts import ArrayGeometry, FieldPrediction, OperatorPrediction, QueryBatch
from ..physics import IdealFarFieldOperator
from .operator_residual import OperatorResidualCNN


class PhysicsStructuredForwardModel(nn.Module):
    """Predict an effective operator, then apply all 16 command responses."""

    def __init__(
        self,
        geometry: ArrayGeometry | None = None,
        correction: OperatorResidualCNN | None = None,
    ) -> None:
        super().__init__()
        self.geometry = geometry or ArrayGeometry()
        self.ideal_operator = IdealFarFieldOperator(self.geometry)
        self.correction = correction or OperatorResidualCNN()

    def operator(self, query: QueryBatch) -> OperatorPrediction:
        ideal = self.ideal_operator(query)
        return self.correction(ideal, query)

    def forward(self, command_iq: torch.Tensor, query: QueryBatch) -> FieldPrediction:
        operator = self.operator(query)
        field = contract_operator_iq(operator.effective_iq, command_iq)
        power = field_power(field, query.range_m)
        return FieldPrediction(field_iq=field, power=power, operator=operator)

