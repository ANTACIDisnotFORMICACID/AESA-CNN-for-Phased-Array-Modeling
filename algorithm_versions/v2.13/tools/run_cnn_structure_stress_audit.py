"""Profile frozen V2.12 CNN structure and validation-only pointing robustness.

This is a diagnostic rehearsal for V2.13.  It never trains, never opens the
external test split, and never treats synthetic coordinate perturbations as
hardware evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
V212_ROOT = ROOT.parent / "v2.12"
V210_ROOT = ROOT.parent / "v2.10"
sys.path.insert(0, str(V212_ROOT / "src"))
sys.path.insert(0, str(V210_ROOT / "src"))

from v210.contracts import QueryBatch  # noqa: E402
from v210.io import load_amplitude_dataset, load_amplitude_split  # noqa: E402
from v212.basis import FiltrationSSHTransform  # noqa: E402
from v212.codebook import MeasurementCodebook  # noqa: E402
from v212.observation import ReceiverObservation  # noqa: E402
from v212.operator import BranchAblation, operator_from_checkpoint  # noqa: E402
from v212.representation import tangent_frame  # noqa: E402


SEEDS = tuple(range(21201, 21211))


def _run_dir(
    screen: Path,
    soft: Path,
    formal: Path,
    candidate_fix: Path,
    model: str,
    seed: int,
) -> Path:
    if model == "G1-M5":
        root = screen if seed <= 21203 else formal
        name = f"G1-M5-LR240_seed{seed}"
    elif model == "candidate":
        root = soft if seed <= 21203 else (candidate_fix if seed >= 21209 else formal)
        name = f"G2-J-LR240_sep0p001_seed{seed}"
    else:
        raise ValueError(model)
    return root / name


def _load_model(run: Path, training_direction: torch.Tensor, device: str):
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    if result.get("test_status") != "sealed":
        raise ValueError(f"external test is not sealed: {run}")
    if int(result["learnable_parameter_count"]) != 29142:
        raise ValueError(f"parameter mismatch in {run}")
    transform_path = run / "filtration_transform.npz"
    transform = FiltrationSSHTransform.load(transform_path) if transform_path.exists() else None
    model = operator_from_checkpoint(
        checkpoint,
        training_direction=training_direction,
        transform=transform,
        observation=ReceiverObservation(**checkpoint["observation"]),
    ).to(device)
    model.eval()
    return model


@torch.no_grad()
def _predict(model, commands, direction, ablation=None, chunk_size=512):
    chunks = []
    for start in range(0, len(direction), chunk_size):
        block = direction[start : start + chunk_size]
        chunks.append(model.forward_command_grid(commands, block, ablation).T.cpu())
    return torch.cat(chunks)


@torch.no_grad()
def _terms(model, direction, chunk_size=512):
    collected: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(direction), chunk_size):
        block = direction[start : start + chunk_size]
        _, terms = model.envelope(QueryBatch(block[None]))
        for name in ("baseline_iq", "ssh_iq", "fourier_residual_iq"):
            collected.setdefault(name, []).append(terms[name][0].cpu())
    output = {name: torch.cat(parts) for name, parts in collected.items()}
    output["envelope_iq"] = (
        output["baseline_iq"] + output["ssh_iq"] + output["fourier_residual_iq"]
    )
    return output


@torch.no_grad()
def _factorized_predict(
    model,
    commands,
    envelope_direction,
    propagation_direction,
    chunk_size=512,
):
    """Evaluate envelope and analytic propagation at independently chosen directions."""
    command = commands if torch.is_complex(commands) else torch.complex(
        commands[:, 0], commands[:, 1]
    ).reshape(len(commands), 16)
    chunks = []
    for start in range(0, len(envelope_direction), chunk_size):
        envelope_block = envelope_direction[start : start + chunk_size]
        propagation_block = propagation_direction[start : start + chunk_size]
        envelope, _ = model.envelope(QueryBatch(envelope_block[None]))
        phase = model.wavenumber_rad_m * torch.einsum(
            "qk,nk->qn", propagation_block, model.positions_m.to(propagation_block)
        )
        propagation = torch.exp(torch.complex(torch.zeros_like(phase), phase))
        operator = envelope[0] * propagation
        chunks.append(torch.einsum("qn,cn->qc", operator, command.to(operator)).abs().cpu())
    return torch.cat(chunks)


def _rotations(direction: torch.Tensor, angle_deg: float) -> list[torch.Tensor]:
    first, second = tangent_frame(direction)
    angle = torch.as_tensor(np.deg2rad(angle_deg), dtype=direction.dtype, device=direction.device)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    return [
        cosine * direction + sine * first,
        cosine * direction - sine * first,
        cosine * direction + sine * second,
        cosine * direction - sine * second,
    ]


def _nmse(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, np.ndarray]:
    numerator = (prediction - target).square()
    denominator = target.square()
    aggregate = float(10.0 * torch.log10(numerator.sum() / denominator.sum().clamp_min(1e-30)))
    per_direction = 10.0 * torch.log10(
        numerator.sum(dim=1) / denominator.sum(dim=1).clamp_min(1e-30)
    )
    return aggregate, per_direction.numpy()


def _ratio_nmse(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    ratio = (prediction - reference).square().sum() / reference.square().sum().clamp_min(1e-30)
    return float(10.0 * torch.log10(ratio.clamp_min(1e-30)))


def _summary(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _bootstrap_mean_ci(values, seed: int, samples: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draw = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return [float(item) for item in np.quantile(draw, (0.025, 0.975))]


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _correlation(left, right) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) == 0 or np.std(right) == 0:
        return {"pearson": 0.0, "spearman": 0.0}
    return {
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "spearman": float(np.corrcoef(_rank(left), _rank(right))[0, 1]),
    }


def _nearest_training_angle(validation: torch.Tensor, training: torch.Tensor) -> np.ndarray:
    largest = torch.full((len(validation),), -1.0, dtype=torch.float32)
    for start in range(0, len(training), 1024):
        cosine = validation @ training[start : start + 1024].T
        largest = torch.maximum(largest, cosine.max(dim=1).values)
    return torch.rad2deg(torch.acos(largest.clamp(-1.0, 1.0))).numpy()


def _parameter_groups(model) -> dict[str, int]:
    groups = {"baseline_complex_gain": 0, "ssh_auxiliary": 0, "fourier_cnn": 0, "other": 0}
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        if name.startswith(("b0_raw", "b_real_rest", "b_imag_rest")):
            groups["baseline_complex_gain"] += count
        elif name.startswith("ssh_"):
            groups["ssh_auxiliary"] += count
        elif name.startswith("residual."):
            groups["fourier_cnn"] += count
        else:
            groups["other"] += count
    groups["total"] = sum(parameter.numel() for parameter in model.parameters())
    return groups


def _energy(iq: torch.Tensor) -> np.ndarray:
    return iq.double().square().sum(dim=(1, 2, 3)).numpy()


def _worst_cells(direction: np.ndarray, error_db: np.ndarray, count=8) -> list[dict[str, float | int]]:
    error_db = np.asarray(error_db, dtype=np.float64)
    theta = np.rad2deg(np.arccos(np.clip(direction[:, 2], -1.0, 1.0)))
    phi = np.rad2deg(np.arctan2(direction[:, 1], direction[:, 0]))
    theta_bin = np.floor(theta / 10.0).astype(int) * 10
    phi_bin = np.floor((phi + 90.0) / 10.0).astype(int) * 10 - 90
    rows = []
    for tb, pb in sorted(set(zip(theta_bin.tolist(), phi_bin.tolist()))):
        mask = (theta_bin == tb) & (phi_bin == pb)
        rows.append({
            "theta_start_deg": int(tb),
            "phi_start_deg": int(pb),
            "direction_count": int(mask.sum()),
            "median_per_direction_nmse_db": float(np.median(error_db[mask])),
            "p95_per_direction_nmse_db": float(np.quantile(error_db[mask], 0.95)),
        })
    rows.sort(key=lambda row: row["median_per_direction_nmse_db"], reverse=True)
    return rows[:count]


def _make_plot(report: dict[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    angles = [row["angle_deg"] for row in report["pointing_robustness"]]
    candidate = [row["candidate_nmse_db"]["mean"] for row in report["pointing_robustness"]]
    control = [row["matched_fourier_nmse_db"]["mean"] for row in report["pointing_robustness"]]
    ablation = report["feature_reliance"]["nmse_degradation_db"]
    labels = list(ablation)
    values = [ablation[name]["mean"] for name in labels]
    groups = report["structure"]["parameter_groups"]
    group_labels = [name for name in groups if name not in {"total", "other"}]
    group_values = [groups[name] for name in group_labels]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].plot(angles, candidate, "o-", label="SSH+Fourier candidate")
    axes[0].plot(angles, control, "s--", label="matched Fourier control")
    axes[0].set_xlabel("direction metadata error (deg)")
    axes[0].set_ylabel("validation amplitude NMSE (dB)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, values, color="#d95f02")
    axes[1].set_ylabel("NMSE degradation (dB)")
    axes[1].set_title("post-hoc feature reliance")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[2].bar(group_labels, group_values, color="#1b9e77")
    axes[2].set_ylabel("learnable parameters")
    axes[2].set_title("candidate capacity allocation")
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].grid(True, axis="y", alpha=0.3)
    figure.suptitle("V2.13 frozen-CNN structure and robustness rehearsal")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--screen-root", required=True, type=Path)
    parser.add_argument("--soft-root", required=True, type=Path)
    parser.add_argument("--formal-root", required=True, type=Path)
    parser.add_argument("--candidate-fix-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot", required=True, type=Path)
    parser.add_argument("--angles-deg", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if any(angle <= 0 for angle in args.angles_deg):
        raise ValueError("all pointing angles must be positive")
    dataset = load_amplitude_dataset(args.dataset)
    split = load_amplitude_split(args.split_dir)
    codebook = MeasurementCodebook.load(args.codebook)
    all_direction = torch.as_tensor(dataset.direction, dtype=torch.float32)
    all_target = torch.as_tensor(dataset.amplitude.T.copy(), dtype=torch.float32)
    command_index = codebook.indices("validation")
    commands = codebook.weights[command_index].to(args.device)

    seed_rows = []
    feature_degradation: dict[str, list[float]] = {
        "disable_ssh": [], "disable_fourier": [],
        **{f"disable_l{degree}": [] for degree in range(1, 6)},
    }
    robustness: dict[float, dict[str, list[float]]] = {
        0.0: {
            "candidate": [], "control": [], "candidate_self_change": [], "control_self_change": [],
            "candidate_envelope_only_change": [], "candidate_propagation_only_change": [],
        },
        **{
            float(angle): {
                "candidate": [], "control": [], "candidate_self_change": [], "control_self_change": [],
                "candidate_envelope_only_change": [], "candidate_propagation_only_change": [],
            }
            for angle in args.angles_deg
        },
    }
    all_error = []
    all_sensitivity = []
    all_ssh_fraction = []
    all_fourier_fraction = []
    all_fourier_saturation = []
    parameter_groups = None
    validation_direction_reference = None
    target_reference = None
    train_index_reference = None
    validation_index_reference = None
    split_indices_identical_across_seeds = True
    factorization_closure_max_abs = []

    for seed in SEEDS:
        candidate_run = _run_dir(
            args.screen_root, args.soft_root, args.formal_root, args.candidate_fix_root, "candidate", seed
        )
        checkpoint = torch.load(candidate_run / "checkpoint.pt", map_location="cpu", weights_only=True)
        train_index = torch.as_tensor(
            checkpoint.get("train_query_indices", split.query_train), dtype=torch.long
        )
        validation_index = torch.as_tensor(
            checkpoint.get("validation_query_indices", split.query_validation), dtype=torch.long
        )
        training_direction = all_direction[train_index]
        validation_direction = all_direction[validation_index]
        target = all_target[validation_index][:, command_index]
        candidate = _load_model(candidate_run, training_direction, args.device)
        control = _load_model(
            _run_dir(args.screen_root, args.soft_root, args.formal_root, args.candidate_fix_root, "G1-M5", seed),
            training_direction,
            args.device,
        )
        direction_device = validation_direction.to(args.device)
        candidate_base = _predict(candidate, commands, direction_device, chunk_size=args.chunk_size)
        factorized_base = _factorized_predict(
            candidate, commands, direction_device, direction_device, args.chunk_size
        )
        factorization_closure_max_abs.append(float((candidate_base - factorized_base).abs().max()))
        control_base = _predict(control, commands, direction_device, chunk_size=args.chunk_size)
        candidate_nmse, candidate_error = _nmse(candidate_base, target)
        control_nmse, _ = _nmse(control_base, target)
        robustness[0.0]["candidate"].append(candidate_nmse)
        robustness[0.0]["control"].append(control_nmse)
        robustness[0.0]["candidate_self_change"].append(-300.0)
        robustness[0.0]["control_self_change"].append(-300.0)
        robustness[0.0]["candidate_envelope_only_change"].append(-300.0)
        robustness[0.0]["candidate_propagation_only_change"].append(-300.0)

        model_angle_sensitivity = None
        per_angle = {}
        for angle in args.angles_deg:
            candidate_shift = []
            control_shift = []
            candidate_envelope_only = []
            candidate_propagation_only = []
            shifted_directions = _rotations(direction_device, float(angle))
            for shifted in shifted_directions:
                candidate_shift.append(_predict(candidate, commands, shifted, chunk_size=args.chunk_size))
                control_shift.append(_predict(control, commands, shifted, chunk_size=args.chunk_size))
                candidate_envelope_only.append(_factorized_predict(
                    candidate, commands, shifted, direction_device, args.chunk_size
                ))
                candidate_propagation_only.append(_factorized_predict(
                    candidate, commands, direction_device, shifted, args.chunk_size
                ))
            candidate_stack = torch.stack(candidate_shift)
            control_stack = torch.stack(control_shift)
            envelope_only_stack = torch.stack(candidate_envelope_only)
            propagation_only_stack = torch.stack(candidate_propagation_only)
            repeated_target = target[None].expand_as(candidate_stack)
            candidate_shift_nmse, _ = _nmse(
                candidate_stack.reshape(-1, target.shape[1]),
                repeated_target.reshape(-1, target.shape[1]),
            )
            control_shift_nmse, _ = _nmse(
                control_stack.reshape(-1, target.shape[1]),
                repeated_target.reshape(-1, target.shape[1]),
            )
            candidate_self = _ratio_nmse(candidate_stack, candidate_base[None].expand_as(candidate_stack))
            control_self = _ratio_nmse(control_stack, control_base[None].expand_as(control_stack))
            envelope_only_change = _ratio_nmse(
                envelope_only_stack, candidate_base[None].expand_as(envelope_only_stack)
            )
            propagation_only_change = _ratio_nmse(
                propagation_only_stack, candidate_base[None].expand_as(propagation_only_stack)
            )
            robustness[float(angle)]["candidate"].append(candidate_shift_nmse)
            robustness[float(angle)]["control"].append(control_shift_nmse)
            robustness[float(angle)]["candidate_self_change"].append(candidate_self)
            robustness[float(angle)]["control_self_change"].append(control_self)
            robustness[float(angle)]["candidate_envelope_only_change"].append(envelope_only_change)
            robustness[float(angle)]["candidate_propagation_only_change"].append(propagation_only_change)
            per_angle[str(angle)] = {
                "candidate_nmse_db": candidate_shift_nmse,
                "matched_fourier_nmse_db": control_shift_nmse,
                "candidate_self_change_nmse_db": candidate_self,
                "matched_fourier_self_change_nmse_db": control_self,
                "candidate_envelope_only_change_nmse_db": envelope_only_change,
                "candidate_propagation_only_change_nmse_db": propagation_only_change,
                "outside_strict_front_hemisphere_fraction": float(
                    torch.stack([shifted[:, 0] <= 0 for shifted in shifted_directions]).float().mean()
                ),
            }
            if np.isclose(angle, 0.5):
                delta = (candidate_stack - candidate_base[None]).square().mean(dim=(0, 2))
                scale = candidate_base.square().mean(dim=1).clamp_min(1e-30)
                model_angle_sensitivity = (delta / scale).numpy()

        base = candidate_nmse
        ablations = {
            "disable_ssh": BranchAblation(disable_ssh=True),
            "disable_fourier": BranchAblation(disable_fourier=True),
            **{
                f"disable_l{degree}": BranchAblation(disabled_ssh_degrees=(degree,))
                for degree in range(1, 6)
            },
        }
        seed_ablation = {}
        for name, ablation in ablations.items():
            prediction = _predict(candidate, commands, direction_device, ablation, args.chunk_size)
            value, _ = _nmse(prediction, target)
            degradation = value - base
            feature_degradation[name].append(degradation)
            seed_ablation[name] = degradation

        terms = _terms(candidate, direction_device, args.chunk_size)
        baseline_energy = _energy(terms["baseline_iq"])
        ssh_energy = _energy(terms["ssh_iq"])
        fourier_energy = _energy(terms["fourier_residual_iq"])
        envelope_energy = _energy(terms["envelope_iq"])
        ssh_fraction = ssh_energy / np.maximum(envelope_energy, 1e-30)
        fourier_fraction = fourier_energy / np.maximum(envelope_energy, 1e-30)
        residual_abs = terms["fourier_residual_iq"].abs().numpy()
        residual_saturation = np.mean(residual_abs > 1.8, axis=(1, 2, 3))
        if model_angle_sensitivity is None:
            smallest = min(args.angles_deg, key=lambda value: abs(value - 0.5))
            model_angle_sensitivity = np.zeros(len(validation_direction), dtype=np.float64)
            sensitivity_note = f"0.5 degree was not requested; correlation omitted (nearest requested {smallest})"
        else:
            sensitivity_note = "four-direction mean relative output change at 0.5 degree"
        all_error.append(candidate_error)
        all_sensitivity.append(model_angle_sensitivity)
        all_ssh_fraction.append(ssh_fraction)
        all_fourier_fraction.append(fourier_fraction)
        all_fourier_saturation.append(residual_saturation)
        if parameter_groups is None:
            parameter_groups = _parameter_groups(candidate)
            validation_direction_reference = validation_direction.numpy()
            target_reference = target.numpy()
            train_index_reference = train_index.clone()
            validation_index_reference = validation_index.clone()
        else:
            split_indices_identical_across_seeds = split_indices_identical_across_seeds and bool(
                torch.equal(train_index, train_index_reference)
                and torch.equal(validation_index, validation_index_reference)
            )
        seed_rows.append({
            "seed": seed,
            "baseline_candidate_nmse_db": candidate_nmse,
            "baseline_matched_fourier_nmse_db": control_nmse,
            "candidate_advantage_db": control_nmse - candidate_nmse,
            "pointing": per_angle,
            "ablation_nmse_degradation_db": seed_ablation,
            "latent_energy": {
                "baseline": _summary(baseline_energy),
                "ssh_fraction_of_envelope": _summary(ssh_fraction),
                "fourier_fraction_of_envelope": _summary(fourier_fraction),
                "fourier_output_abs_fraction_above_90pct_bound": float(np.mean(residual_abs > 1.8)),
                "fourier_output_abs_fraction_above_99pct_bound": float(np.mean(residual_abs > 1.98)),
            },
        })

    error = np.mean(np.stack(all_error), axis=0)
    sensitivity = np.mean(np.stack(all_sensitivity), axis=0)
    ssh_fraction = np.mean(np.stack(all_ssh_fraction), axis=0)
    fourier_fraction = np.mean(np.stack(all_fourier_fraction), axis=0)
    fourier_saturation = np.mean(np.stack(all_fourier_saturation), axis=0)
    assert validation_direction_reference is not None and target_reference is not None
    nearest = _nearest_training_angle(
        torch.from_numpy(validation_direction_reference),
        all_direction[torch.as_tensor(split.query_train, dtype=torch.long)],
    )
    signal = np.sqrt(np.mean(target_reference**2, axis=1))
    signal_db = 20.0 * np.log10(np.maximum(signal / signal.max(), 1e-30))

    robustness_rows = []
    for index, angle in enumerate(sorted(robustness)):
        row = robustness[angle]
        candidate_values = row["candidate"]
        control_values = row["control"]
        advantage = np.asarray(control_values) - np.asarray(candidate_values)
        robustness_rows.append({
            "angle_deg": angle,
            "candidate_nmse_db": _summary(candidate_values),
            "matched_fourier_nmse_db": _summary(control_values),
            "candidate_advantage_db": _summary(advantage),
            "candidate_advantage_bootstrap_95ci_db": _bootstrap_mean_ci(
                advantage, 21350 + index, args.bootstrap_samples
            ),
            "candidate_self_change_nmse_db": _summary(row["candidate_self_change"]),
            "matched_fourier_self_change_nmse_db": _summary(row["control_self_change"]),
            "candidate_envelope_only_change_nmse_db": _summary(row["candidate_envelope_only_change"]),
            "candidate_propagation_only_change_nmse_db": _summary(row["candidate_propagation_only_change"]),
            "outside_strict_front_hemisphere_fraction": (
                0.0 if angle == 0.0 else seed_rows[0]["pointing"][str(angle)][
                    "outside_strict_front_hemisphere_fraction"
                ]
            ),
        })

    report = {
        "schema": "v213-frozen-cnn-structure-stress-audit-1",
        "status": "validation_only_synthetic_pointing_rehearsal",
        "claim_boundary": (
            "No training and no external-test access. Direction perturbations are synthetic metadata-error "
            "diagnostics, not measured hardware robustness. Latent branch quantities are architectural only."
        ),
        "external_test_used": False,
        "training_performed": False,
        "models": {
            "candidate": "frozen V2.12-S G2-J-LR240 sep=0.001, ten seeds",
            "control": "frozen equal-parameter G1-M5 matched-Fourier, ten paired seeds",
        },
        "validation": {
            "direction_count": int(len(validation_direction_reference)),
            "command_count": int(len(command_index)),
            "seeds": list(SEEDS),
            "command_indices": command_index.tolist(),
            "split_indices_identical_across_seeds": split_indices_identical_across_seeds,
            "train_validation_disjoint": bool(
                len(set(train_index_reference.tolist()) & set(validation_index_reference.tolist())) == 0
            ),
            "factorized_forward_max_abs_closure": max(factorization_closure_max_abs),
        },
        "structure": {
            "parameter_groups": parameter_groups,
            "direction_input_channels": 27,
            "aperture_shape": [4, 4],
            "fourier_cnn": "31-channel input, width 24, two residual blocks, 3x3 stem/head",
            "physics_tail": "analytic propagation -> rank-1 Q -> coherent command contraction -> amplitude",
        },
        "baseline": {
            "candidate_nmse_db": _summary(robustness[0.0]["candidate"]),
            "matched_fourier_nmse_db": _summary(robustness[0.0]["control"]),
            "paired_candidate_advantage_db": _summary(
                np.asarray(robustness[0.0]["control"]) - np.asarray(robustness[0.0]["candidate"])
            ),
        },
        "pointing_robustness": robustness_rows,
        "feature_reliance": {
            "interpretation": "post-hoc removal reliance, not independently retrained branch capability",
            "nmse_degradation_db": {
                name: {
                    **_summary(values),
                    "bootstrap_95ci_mean": _bootstrap_mean_ci(
                        values, 21400 + offset, args.bootstrap_samples
                    ),
                }
                for offset, (name, values) in enumerate(feature_degradation.items())
            },
        },
        "error_correlates": {
            "sensitivity_note": sensitivity_note,
            "pointing_sensitivity_0p5deg": _correlation(error, sensitivity),
            "ssh_energy_fraction": _correlation(error, ssh_fraction),
            "fourier_energy_fraction": _correlation(error, fourier_fraction),
            "fourier_output_saturation_fraction": _correlation(error, fourier_saturation),
            "nearest_training_angle_deg": _correlation(error, nearest),
            "relative_signal_db": _correlation(error, signal_db),
        },
        "direction_statistics": {
            "mean_seed_per_direction_nmse_db": _summary(error),
            "relative_signal_db": _summary(signal_db),
            "nearest_training_angle_deg": _summary(nearest),
            "pointing_sensitivity_0p5deg": _summary(sensitivity),
            "ssh_energy_fraction": _summary(ssh_fraction),
            "fourier_energy_fraction": _summary(fourier_fraction),
            "fourier_output_saturation_fraction": _summary(fourier_saturation),
            "worst_10deg_cells": _worst_cells(validation_direction_reference, error),
        },
        "per_seed": seed_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _make_plot(report, args.plot)
    print(json.dumps({
        "status": report["status"],
        "baseline": report["baseline"],
        "pointing_robustness": report["pointing_robustness"],
        "feature_reliance": report["feature_reliance"],
        "error_correlates": report["error_correlates"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
