"""Coherent single-polarization rank-one array operator for G1/G2/G3."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

import torch
from torch import nn

from v210.contracts import ArrayGeometry, QueryBatch
from v210.models.blocks_2d import ApertureCorrectionBranch

from .basis import FiltrationSSHTransform, real_spherical_harmonics
from .encoding import (
    DirectionFourierEncoder,
    MatchedFourierBasis,
    ResidualizedDirectionFourierEncoder,
)
from .observation import ObservationKind, ReceiverObservation


def _geometry_maps() -> torch.Tensor:
    row = torch.linspace(-1.0, 1.0, 4)[:, None].expand(4, 4)
    col = torch.linspace(-1.0, 1.0, 4)[None, :].expand(4, 4)
    edge = ((row.abs() == 1.0) | (col.abs() == 1.0)).to(torch.float32)
    corner = ((row.abs() == 1.0) & (col.abs() == 1.0)).to(torch.float32)
    return torch.stack((row, col, edge, corner), dim=0)


def _iq_to_complex(value: torch.Tensor, iq_axis: int) -> torch.Tensor:
    real, imag = value.unbind(dim=iq_axis)
    return torch.complex(real, imag)


def _complex_to_iq(value: torch.Tensor, iq_axis: int) -> torch.Tensor:
    return torch.stack((value.real, value.imag), dim=iq_axis)


class FourierEnvelopeResidual(nn.Module):
    """Zero-start Fourier-CNN residual on the dephased 4x4 envelope."""

    def __init__(
        self,
        *,
        width: int = 24,
        depth: int = 2,
        maximum_magnitude: float = 2.0,
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if maximum_magnitude <= 0:
            raise ValueError("maximum_magnitude must be positive")
        self.encoder = encoder or DirectionFourierEncoder(4)
        if getattr(self.encoder, "output_channels", None) != 27:
            raise ValueError("Fourier residual encoder must output 27 channels")
        self.branch = ApertureCorrectionBranch(31, width, depth=depth, output_channels=2)
        self.maximum_magnitude = float(maximum_magnitude)
        self.register_buffer("geometry_maps", _geometry_maps())

    def forward(self, unit_direction: torch.Tensor) -> torch.Tensor:
        if unit_direction.ndim != 3 or unit_direction.shape[-1] != 3:
            raise ValueError("unit_direction must have shape [B,Q,3]")
        batch, queries = unit_direction.shape[:2]
        encoded = self.encoder(unit_direction)
        direction_maps = encoded[..., None, None].expand(-1, -1, -1, 4, 4)
        geometry = self.geometry_maps.to(unit_direction)[None, None].expand(
            batch, queries, -1, -1, -1
        )
        features = torch.cat((direction_maps, geometry), dim=2)
        raw = self.branch(features.reshape(batch * queries, 31, 4, 4))
        raw = raw.reshape(batch, queries, 2, 4, 4)
        return self.maximum_magnitude * torch.tanh(raw / self.maximum_magnitude)


@dataclass(frozen=True)
class CoherentRank1Prediction:
    amplitude: torch.Tensor
    power: torch.Tensor
    field_iq: torch.Tensor
    g_iq: torch.Tensor
    envelope_iq: torch.Tensor
    terms_iq: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.amplitude.ndim != 2 or self.power.shape != self.amplitude.shape:
            raise ValueError("amplitude and power must have shape [B,Q]")
        if self.field_iq.shape != (*self.amplitude.shape, 2):
            raise ValueError("field_iq must have shape [B,Q,2]")
        expected = (*self.amplitude.shape, 2, 4, 4)
        if self.g_iq.shape != expected or self.envelope_iq.shape != expected:
            raise ValueError("g_iq and envelope_iq must have shape [B,Q,2,4,4]")

    @property
    def operator_iq(self) -> torch.Tensor:
        """Compatibility alias; ``g_iq`` is the stable V2.12 interface name."""
        return self.g_iq


@dataclass(frozen=True)
class BranchAblation:
    """Post-hoc reliance switches; these do not represent retrained capability."""

    disable_ssh: bool = False
    disable_fourier: bool = False
    disabled_ssh_degrees: tuple[int, ...] = ()


class CoherentRank1Operator(nn.Module):
    """G1/G2 operator with analytic array phase and no explicit coupling matrix."""

    def __init__(
        self,
        geometry: ArrayGeometry | None = None,
        *,
        ssh_transform: FiltrationSSHTransform | None = None,
        reference_distance_m: float = 1.0,
        residual_width: int = 24,
        residual_depth: int = 2,
        residual_maximum_magnitude: float = 2.0,
        envelope_scale: float = 1.0,
        ssh_initial_scale: float = 1e-3,
        ssh_seed: int = 21271,
        fourier_input_mode: str = "raw",
        training_direction: torch.Tensor | None = None,
        training_loss_weights: torch.Tensor | None = None,
        matched_fourier_basis: MatchedFourierBasis | None = None,
        residual_enabled: bool = True,
        observation: ReceiverObservation | None = None,
    ) -> None:
        super().__init__()
        if reference_distance_m <= 0 or envelope_scale <= 0 or ssh_initial_scale < 0:
            raise ValueError("reference distance and SSH scale are invalid")
        self.geometry = geometry or ArrayGeometry()
        self.reference_distance_m = float(reference_distance_m)
        self.observation = observation or ReceiverObservation(
            ObservationKind.HFSS_EPHI, self.geometry.frequency_hz
        )
        if not math.isclose(
            self.observation.frequency_hz,
            self.geometry.frequency_hz,
            rel_tol=1e-9,
        ):
            raise ValueError("observation and geometry frequencies must match")
        self.register_buffer(
            "positions_m",
            torch.as_tensor(self.geometry.positions_m.reshape(16, 3), dtype=torch.float32),
        )
        self.register_buffer("envelope_scale", torch.tensor(float(envelope_scale)))
        self.wavenumber_rad_m = 2.0 * math.pi / self.geometry.wavelength_m
        if fourier_input_mode not in {"raw", "filtration_residualized"}:
            raise ValueError("fourier_input_mode must be raw or filtration_residualized")
        if fourier_input_mode == "filtration_residualized":
            if ssh_transform is None or training_direction is None:
                raise ValueError("residualized Fourier requires SSH transform and training directions")
            residual_encoder: nn.Module = ResidualizedDirectionFourierEncoder(
                ssh_transform, training_direction, loss_weights=training_loss_weights
            )
        else:
            residual_encoder = DirectionFourierEncoder(4)
        self.fourier_input_mode = fourier_input_mode
        self.residual_enabled = bool(residual_enabled)
        self.residual = (
            FourierEnvelopeResidual(
                width=residual_width,
                depth=residual_depth,
                maximum_magnitude=residual_maximum_magnitude,
                encoder=residual_encoder,
            )
            if self.residual_enabled
            else None
        )

        anchor_epsilon = 1e-6
        anchor_raw = math.log(math.expm1(1.0 - anchor_epsilon))
        self.anchor_epsilon = anchor_epsilon
        self.b0_raw = nn.Parameter(torch.tensor(anchor_raw, dtype=torch.float32))
        self.b_real_rest = nn.Parameter(torch.ones(15))
        self.b_imag_rest = nn.Parameter(torch.zeros(15))

        if ssh_transform is not None and matched_fourier_basis is not None:
            raise ValueError("SSH and matched-Fourier auxiliary bases are mutually exclusive")
        self.matched_fourier_basis = matched_fourier_basis
        self.auxiliary_kind = (
            "ssh" if ssh_transform is not None else "matched_fourier"
            if matched_fourier_basis is not None else "none"
        )
        self.ssh_degree = -1 if ssh_transform is None else ssh_transform.degree
        if ssh_transform is None and matched_fourier_basis is None:
            self.register_buffer("ssh_transform", torch.empty(0, 0))
            self.ssh_ranks: tuple[int, ...] = ()
            self.ssh_slices: tuple[slice, ...] = ()
            self.register_parameter("ssh_beta_real", None)
            self.register_parameter("ssh_beta_imag", None)
            self.register_parameter("ssh_alpha", None)
        else:
            ranks = ssh_transform.ranks if ssh_transform is not None else matched_fourier_basis.ranks
            degree = len(ranks) - 1
            self.register_buffer(
                "ssh_transform",
                ssh_transform.transform.to(torch.float32)
                if ssh_transform is not None else torch.empty(0, 0),
            )
            self.ssh_degree = degree
            self.ssh_ranks = ranks
            start = 0
            slices: list[slice] = []
            for rank in ranks:
                slices.append(slice(start, start + rank))
                start += rank
            self.ssh_slices = tuple(slices)
            retained_nonconstant = sum(ranks[1:])
            generator = torch.Generator().manual_seed(ssh_seed)
            beta_real = ssh_initial_scale * torch.randn(
                retained_nonconstant, 16, generator=generator
            )
            beta_imag = ssh_initial_scale * torch.randn(
                retained_nonconstant, 16, generator=generator
            )
            self.ssh_beta_real = nn.Parameter(beta_real)
            self.ssh_beta_imag = nn.Parameter(beta_imag)
            self.ssh_alpha = nn.Parameter(torch.zeros(degree))

    @property
    def has_ssh(self) -> bool:
        return self.auxiliary_kind == "ssh" and self.ssh_degree >= 1

    @property
    def has_angular_auxiliary(self) -> bool:
        return self.auxiliary_kind != "none" and self.ssh_degree >= 1

    def baseline_gain(self) -> torch.Tensor:
        b0 = torch.nn.functional.softplus(self.b0_raw) + self.anchor_epsilon
        real = torch.cat((b0[None], self.b_real_rest))
        imag = torch.cat((torch.zeros_like(b0)[None], self.b_imag_rest))
        return torch.complex(real, imag)

    def _angular_envelope(
        self,
        unit_direction: torch.Tensor,
        ablation: BranchAblation,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        shape = (*unit_direction.shape[:2], 16)
        if not self.has_angular_auxiliary or (self.has_ssh and ablation.disable_ssh):
            return torch.zeros(shape, dtype=torch.complex64, device=unit_direction.device), {}
        if self.has_ssh:
            raw = real_spherical_harmonics(unit_direction, self.ssh_degree)
            phi = raw @ self.ssh_transform.to(raw)
        else:
            phi = self.matched_fourier_basis(unit_direction)
        nonconstant_blocks: list[torch.Tensor] = []
        for l_value in range(1, self.ssh_degree + 1):
            if self.has_ssh:
                block = phi[..., self.ssh_slices[l_value]]
            else:
                block = phi[..., self.matched_fourier_basis.block_slices[l_value - 1]]
            if block.shape[-1]:
                nonconstant_blocks.append(block)
        if not nonconstant_blocks:
            return torch.zeros(shape, dtype=torch.complex64, device=unit_direction.device), {}
        beta = torch.complex(self.ssh_beta_real, self.ssh_beta_imag).to(unit_direction.device)
        total = torch.zeros(shape, dtype=beta.dtype, device=unit_direction.device)
        by_degree: dict[int, torch.Tensor] = {}
        start = 0
        for l_value, block in zip(range(1, self.ssh_degree + 1), nonconstant_blocks):
            width = block.shape[-1]
            contribution = torch.einsum(
                "bqr,rn->bqn",
                (block * self.ssh_alpha[l_value - 1]).to(beta.dtype),
                beta[start : start + width],
            )
            if l_value not in ablation.disabled_ssh_degrees:
                total = total + contribution
            by_degree[l_value] = contribution
            start += width
        return total, by_degree

    def envelope(
        self,
        query: QueryBatch,
        ablation: BranchAblation | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ablation = ablation or BranchAblation()
        direction = query.unit_direction
        scale = self.envelope_scale.to(direction)
        baseline = scale * self.baseline_gain().to(direction.device)[None, None].expand(
            direction.shape[0], direction.shape[1], -1
        )
        angular, degree_terms = self._angular_envelope(direction, ablation)
        ssh = scale * angular
        if self.residual is None or ablation.disable_fourier:
            residual_iq = torch.zeros(
                *direction.shape[:2], 2, 4, 4,
                dtype=direction.dtype, device=direction.device,
            )
        else:
            residual_iq = self.residual(direction)
        residual = scale * _iq_to_complex(residual_iq, 2).reshape(
            *direction.shape[:2], 16
        )
        residual_iq = _complex_to_iq(
            residual.reshape(*direction.shape[:2], 4, 4), 2
        )
        envelope = baseline + ssh + residual
        terms = {
            "baseline_iq": _complex_to_iq(baseline.reshape(*direction.shape[:2], 4, 4), 2),
            "ssh_iq": _complex_to_iq(ssh.reshape(*direction.shape[:2], 4, 4), 2),
            "fourier_residual_iq": residual_iq,
        }
        for l_value, contribution in degree_terms.items():
            terms[f"ssh_l{l_value}_iq"] = _complex_to_iq(
                (scale * contribution).reshape(*direction.shape[:2], 4, 4), 2
            )
        return envelope, terms

    def operator(
        self,
        query: QueryBatch,
        ablation: BranchAblation | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        envelope, terms = self.envelope(query, ablation)
        direction = query.unit_direction
        positions = self.positions_m.to(direction)
        phase = self.wavenumber_rad_m * torch.einsum("bqk,nk->bqn", direction, positions)
        propagation = torch.exp(torch.complex(torch.zeros_like(phase), phase))
        operator = envelope * propagation
        operator_iq = _complex_to_iq(operator.reshape(*direction.shape[:2], 4, 4), 2)
        terms = {
            **terms,
            "envelope_iq": _complex_to_iq(
                envelope.reshape(*direction.shape[:2], 4, 4), 2
            ),
            "propagation_iq": _complex_to_iq(
                propagation.reshape(*direction.shape[:2], 4, 4), 2
            ),
        }
        return operator_iq, terms

    def q_matrix(self, query: QueryBatch, ablation: BranchAblation | None = None) -> torch.Tensor:
        operator_iq, _ = self.operator(query, ablation)
        g = _iq_to_complex(operator_iq, 2).reshape(*query.unit_direction.shape[:2], 16)
        return g.conj()[..., :, None] * g[..., None, :]

    def dephased_q_matrix(
        self, query: QueryBatch, ablation: BranchAblation | None = None
    ) -> torch.Tensor:
        envelope, _ = self.envelope(query, ablation)
        return envelope.conj()[..., :, None] * envelope[..., None, :]

    def forward(
        self,
        command_iq: torch.Tensor,
        query: QueryBatch,
        ablation: BranchAblation | None = None,
    ) -> CoherentRank1Prediction:
        if command_iq.ndim != 4 or command_iq.shape[-3:] != (2, 4, 4):
            raise ValueError("command_iq must have shape [B,2,4,4]")
        if command_iq.shape[0] != query.unit_direction.shape[0]:
            raise ValueError("command and query batch sizes must match")
        operator_iq, terms = self.operator(query, ablation)
        operator = _iq_to_complex(operator_iq, 2).reshape(*query.unit_direction.shape[:2], 16)
        command = _iq_to_complex(command_iq, 1).reshape(command_iq.shape[0], 16)
        field = torch.einsum("bqn,bn->bq", operator, command)
        if query.range_m is not None:
            field = field * (self.reference_distance_m / query.range_m.clamp_min(1e-8))
        amplitude = field.abs()
        return CoherentRank1Prediction(
            amplitude=amplitude,
            power=amplitude.square(),
            field_iq=_complex_to_iq(field, 2),
            g_iq=operator_iq,
            envelope_iq=terms["envelope_iq"],
            terms_iq=terms,
        )

    def forward_command_grid(
        self,
        command: torch.Tensor,
        unit_direction: torch.Tensor,
        ablation: BranchAblation | None = None,
    ) -> torch.Tensor:
        amplitude, _ = self.forward_command_grid_with_terms(command, unit_direction, ablation)
        return amplitude

    def forward_command_grid_with_terms(
        self,
        command: torch.Tensor,
        unit_direction: torch.Tensor,
        ablation: BranchAblation | None = None,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        if unit_direction.ndim != 2 or unit_direction.shape[-1] != 3:
            raise ValueError("unit_direction must have shape [Q,3]")
        if torch.is_complex(command):
            if command.ndim != 2 or command.shape[-1] != 16:
                raise ValueError("complex command must have shape [C,16]")
            command_complex = command
        else:
            if command.ndim != 4 or command.shape[-3:] != (2, 4, 4):
                raise ValueError("IQ command must have shape [C,2,4,4]")
            command_complex = _iq_to_complex(command, 1).reshape(command.shape[0], 16)
        operator_iq, terms = self.operator(QueryBatch(unit_direction[None]), ablation)
        operator = _iq_to_complex(operator_iq[0], 1).reshape(len(unit_direction), 16)
        amplitude = torch.einsum("qn,cn->cq", operator, command_complex.to(operator)).abs()
        return amplitude, terms

    def structured_regularization(
        self,
        terms_iq: Mapping[str, torch.Tensor],
        *,
        residual_energy_weight: float,
        coupling_l2_weight: float,
        reciprocity_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Retain only generic residual energy; there is no explicit C or SH spectrum loss."""
        del coupling_l2_weight, reciprocity_weight
        residual_energy = terms_iq["fourier_residual_iq"].square().mean()
        total = float(residual_energy_weight) * residual_energy
        zero = torch.zeros_like(residual_energy)
        return total, {
            "residual_energy": residual_energy,
            "coupling_l2": zero,
            "reciprocity": zero,
        }


def operator_from_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    training_direction: torch.Tensor,
    transform: FiltrationSSHTransform | None,
    observation: ReceiverObservation,
) -> CoherentRank1Operator:
    """Reconstruct every backward-compatible V2.12-S recipe before loading state."""
    recipe = str(checkpoint["recipe"])
    ssh_recipes = {
        "G2", "G3", "G2-J-LR240", "G2-I-LR240", "G2-T", "G2-TI", "SSH-only",
    }
    matched = None
    if recipe == "G1-M5-LR240":
        if transform is None:
            raise ValueError("matched-Fourier checkpoint requires its filtration artifact")
        matched = MatchedFourierBasis(
            training_direction,
            transform.ranks,
            extended_bands=6,
            seed=int(checkpoint["seed"]),
        )
    model = CoherentRank1Operator(
        ssh_transform=transform if recipe in ssh_recipes else None,
        observation=observation,
        envelope_scale=float(checkpoint.get("envelope_scale", 1.0)),
        ssh_seed=int(checkpoint.get("seed", 21271)),
        fourier_input_mode=str(checkpoint.get("fourier_input_mode", "raw")),
        training_direction=training_direction,
        matched_fourier_basis=matched,
        residual_enabled=recipe != "SSH-only",
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model
