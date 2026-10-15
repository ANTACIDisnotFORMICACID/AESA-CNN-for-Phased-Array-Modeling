"""Audit why frozen V2.12-S SSH features help, without retraining or opening test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
V210_ROOT = ROOT.parent / "v2.10"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V210_ROOT / "src"))

from v210.contracts import QueryBatch  # noqa: E402
from v210.io import load_amplitude_dataset, load_amplitude_split  # noqa: E402
from v212.basis import FiltrationSSHTransform  # noqa: E402
from v212.codebook import MeasurementCodebook  # noqa: E402
from v212.evaluation import (  # noqa: E402
    amplitude_metrics,
    angular_distance_quartile_report,
    hemisphere_cells,
    nearest_training_angle,
    stratified_separation_subset,
)
from v212.observation import ReceiverObservation  # noqa: E402
from v212.operator import BranchAblation, operator_from_checkpoint  # noqa: E402
from v212.representation import (  # noqa: E402
    branch_energy_decomposition,
    per_direction_nmse_db,
    tangent_jacobian_norm,
    tangent_perturbations,
)


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
    elif model == "G2-J":
        root = screen if seed <= 21203 else formal
        name = f"G2-J-LR240_seed{seed}"
    elif model == "candidate":
        root = soft if seed <= 21203 else (candidate_fix if seed >= 21209 else formal)
        name = f"G2-J-LR240_sep0p001_seed{seed}"
    else:
        raise ValueError(model)
    return root / name


def _load_model(run: Path, training_direction: torch.Tensor, device: str):
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    if int(checkpoint.get("learnable_parameter_count", 29142)) != 29142:
        # Older checkpoint payloads may omit this key; result.json is authoritative.
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        if int(result["learnable_parameter_count"]) != 29142:
            raise ValueError(f"parameter mismatch in {run}")
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    if result.get("test_status") != "sealed":
        raise ValueError(f"external test is not sealed: {run}")
    transform_path = run / "filtration_transform.npz"
    transform = FiltrationSSHTransform.load(transform_path) if transform_path.exists() else None
    model = operator_from_checkpoint(
        checkpoint,
        training_direction=training_direction,
        transform=transform,
        observation=ReceiverObservation(**checkpoint["observation"]),
    ).to(device)
    model.eval()
    return model, checkpoint, transform


@torch.no_grad()
def _predict(model, commands, direction, ablation=None, chunk_size=1024):
    chunks = []
    for start in range(0, len(direction), chunk_size):
        chunks.append(model.forward_command_grid(
            commands, direction[start : start + chunk_size], ablation
        ).T.cpu())
    return torch.cat(chunks)


@torch.no_grad()
def _branch_terms(model, direction, chunk_size=512):
    ssh, fourier, degrees = [], [], {}
    for start in range(0, len(direction), chunk_size):
        block = direction[start : start + chunk_size]
        _, terms = model.envelope(QueryBatch(block[None]))
        ssh.append(terms["ssh_iq"][0].cpu())
        fourier.append(terms["fourier_residual_iq"][0].cpu())
        for key, value in terms.items():
            if key.startswith("ssh_l"):
                degrees.setdefault(key, []).append(value[0].cpu())
    return torch.cat(ssh), torch.cat(fourier), {
        key: torch.cat(parts) for key, parts in degrees.items()
    }


def _iq_complex(value: torch.Tensor) -> torch.Tensor:
    return torch.complex(value[:, 0], value[:, 1]).reshape(len(value), 16)


def _summary(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
    }


def _bootstrap_ci(values, seed=21290) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(200_000, len(array)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, (0.025, 0.975))]


def _correlation(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
    x = x.double()
    y = y.double()
    x_center = x - x.mean()
    y_center = y - y.mean()
    pearson = float(
        (x_center * y_center).sum()
        / (x_center.square().sum().sqrt() * y_center.square().sum().sqrt()).clamp_min(1e-12)
    )
    rank_x = torch.empty_like(x)
    rank_y = torch.empty_like(y)
    rank_x[torch.argsort(x)] = torch.arange(len(x), dtype=x.dtype)
    rank_y[torch.argsort(y)] = torch.arange(len(y), dtype=y.dtype)
    rank_x -= rank_x.mean()
    rank_y -= rank_y.mean()
    spearman = float(
        (rank_x * rank_y).sum()
        / (rank_x.square().sum().sqrt() * rank_y.square().sum().sqrt()).clamp_min(1e-12)
    )
    slope = float((x_center * y_center).sum() / x_center.square().sum().clamp_min(1e-12))
    return pearson, spearman, slope


def _distance_report(
    candidate_prediction: torch.Tensor,
    reference_prediction: torch.Tensor,
    target: torch.Tensor,
    validation_direction: torch.Tensor,
    training_direction: torch.Tensor,
) -> dict:
    distance_deg = torch.rad2deg(nearest_training_angle(validation_direction, training_direction))
    improvement = per_direction_nmse_db(reference_prediction, target) - per_direction_nmse_db(
        candidate_prediction, target
    )
    pearson, spearman, slope = _correlation(distance_deg, improvement)
    boundaries = torch.quantile(distance_deg, torch.linspace(0.1, 0.9, 9))
    groups = torch.bucketize(distance_deg, boundaries)
    bins = []
    for index in range(10):
        mask = groups == index
        candidate_metrics = amplitude_metrics(candidate_prediction[mask], target[mask])
        reference_metrics = amplitude_metrics(reference_prediction[mask], target[mask])
        bins.append({
            "count": int(mask.sum()),
            "distance_mean_deg": float(distance_deg[mask].mean()),
            "distance_min_deg": float(distance_deg[mask].min()),
            "distance_max_deg": float(distance_deg[mask].max()),
            "nmse_improvement_mean_db": float(improvement[mask].mean()),
            "nmse_improvement_median_db": float(improvement[mask].median()),
            "aggregate_nmse_improvement_db": float(
                reference_metrics["nmse_db"] - candidate_metrics["nmse_db"]
            ),
        })
    return {
        "pearson": pearson,
        "spearman": spearman,
        "linear_slope_db_per_degree": slope,
        "deciles": bins,
    }


@torch.no_grad()
def _jacobian_report(model, direction, step_rad, chunk_size):
    center_ssh, center_fourier, _ = _branch_terms(model, direction, chunk_size)
    perturbed = tangent_perturbations(direction, step_rad=step_rad)
    ssh_samples, fourier_samples = [], []
    for sample in perturbed:
        ssh, fourier, _ = _branch_terms(model, sample, chunk_size)
        ssh_samples.append(ssh)
        fourier_samples.append(fourier)
    ssh_raw, ssh_normalized = tangent_jacobian_norm(
        *ssh_samples, center_ssh, step_rad=step_rad
    )
    fourier_raw, fourier_normalized = tangent_jacobian_norm(
        *fourier_samples, center_fourier, step_rad=step_rad
    )
    return {
        "ssh_raw": _summary(ssh_raw.numpy()),
        "fourier_raw": _summary(fourier_raw.numpy()),
        "ssh_normalized": _summary(ssh_normalized.numpy()),
        "fourier_normalized": _summary(fourier_normalized.numpy()),
        "fraction_ssh_raw_less_than_fourier": float((ssh_raw < fourier_raw).float().mean()),
        "fraction_ssh_normalized_less_than_fourier": float(
            (ssh_normalized < fourier_normalized).float().mean()
        ),
        "mean_raw_ratio_ssh_to_fourier": float(
            ssh_raw.mean() / fourier_raw.mean().clamp_min(1e-12)
        ),
        "mean_normalized_ratio_ssh_to_fourier": float(
            ssh_normalized.mean() / fourier_normalized.mean().clamp_min(1e-12)
        ),
    }


def _make_plot(payload: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    seeds = payload["per_seed"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    distance_x = np.asarray([
        [entry["distance_mean_deg"] for entry in row["distance_benefit"]["deciles"]]
        for row in seeds
    ])
    distance_y = np.asarray([
        [entry["nmse_improvement_mean_db"] for entry in row["distance_benefit"]["deciles"]]
        for row in seeds
    ])
    axes[0, 0].plot(distance_x.mean(0), distance_y.mean(0), marker="o")
    axes[0, 0].fill_between(
        distance_x.mean(0), np.quantile(distance_y, .1, axis=0), np.quantile(distance_y, .9, axis=0), alpha=.2
    )
    axes[0, 0].axhline(0, color="black", linewidth=.8)
    axes[0, 0].set(title="Candidate benefit vs train distance", xlabel="Nearest angle (deg)", ylabel="NMSE benefit (dB)")

    degrees = list(range(1, 6))
    full = np.asarray([[row["degree_zero_out"][f"l{d}"]["nmse_degradation_db"] for d in degrees] for row in seeds])
    q4 = np.asarray([[row["degree_zero_out"][f"l{d}"]["q4_nmse_degradation_db"] for d in degrees] for row in seeds])
    axes[0, 1].bar(np.asarray(degrees) - .18, full.mean(0), .36, label="all validation-D")
    axes[0, 1].bar(np.asarray(degrees) + .18, q4.mean(0), .36, label="farthest quartile")
    axes[0, 1].set(title="Post-hoc degree reliance", xlabel="SSH filtration block removed", ylabel="NMSE degradation (dB)")
    axes[0, 1].legend()

    labels = ["G2-J", "Candidate"]
    ssh_ratio = [
        np.mean([row["jacobian"][name]["mean_raw_ratio_ssh_to_fourier"] for row in seeds])
        for name in ("G2-J", "candidate")
    ]
    normalized_ratio = [
        np.mean([row["jacobian"][name]["mean_normalized_ratio_ssh_to_fourier"] for row in seeds])
        for name in ("G2-J", "candidate")
    ]
    x = np.arange(2)
    axes[1, 0].bar(x - .18, ssh_ratio, .36, label="raw derivative ratio")
    axes[1, 0].bar(x + .18, normalized_ratio, .36, label="scale-normalized ratio")
    axes[1, 0].axhline(1, color="black", linewidth=.8)
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set(title="SSH / Fourier tangent Jacobian", ylabel="ratio (<1 means slower SSH)")
    axes[1, 0].legend()

    metrics = ("fourier_low_energy", "ssh_energy", "combined_low_energy")
    changes = [np.mean([row["energy_transfer"]["change_candidate_minus_joint"][metric] for row in seeds]) for metric in metrics]
    axes[1, 1].bar(["Fourier low", "SSH", "Combined low"], changes)
    axes[1, 1].axhline(0, color="black", linewidth=.8)
    axes[1, 1].set(title="Soft-separation energy reallocation", ylabel="candidate - joint energy")

    fig.suptitle("V2.12-S2 frozen representation audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


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
    parser.add_argument("--jacobian-per-cell", type=int, default=20)
    parser.add_argument("--jacobian-step-rad", type=float, default=1e-3)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = load_amplitude_dataset(args.dataset)
    split = load_amplitude_split(args.split_dir)
    codebook = MeasurementCodebook.load(args.codebook)
    direction = torch.as_tensor(dataset.direction, dtype=torch.float32)
    target_all = torch.as_tensor(dataset.amplitude.T.copy(), dtype=torch.float32)
    validation_commands = codebook.indices("validation")
    commands = codebook.weights[validation_commands].to(args.device)
    per_seed = []

    for seed in SEEDS:
        runs = {
            name: _run_dir(
                args.screen_root, args.soft_root, args.formal_root, args.candidate_fix_root, name, seed
            )
            for name in ("G1-M5", "G2-J", "candidate")
        }
        candidate_checkpoint = torch.load(runs["candidate"] / "checkpoint.pt", map_location="cpu", weights_only=True)
        q_train = torch.as_tensor(
            candidate_checkpoint.get("train_query_indices", split.query_train), dtype=torch.long
        )
        q_validation = torch.as_tensor(
            candidate_checkpoint.get("validation_query_indices", split.query_validation), dtype=torch.long
        )
        training_direction_cpu = direction[q_train]
        validation_direction_cpu = direction[q_validation]
        training_direction = training_direction_cpu.to(args.device)
        validation_direction = validation_direction_cpu.to(args.device)
        target = target_all[q_validation][:, validation_commands]

        models = {}
        transforms = {}
        for name, run in runs.items():
            models[name], _, transforms[name] = _load_model(run, training_direction_cpu, args.device)

        predictions = {
            name: _predict(model, commands, validation_direction, chunk_size=args.chunk_size)
            for name, model in models.items()
        }
        distance = _distance_report(
            predictions["candidate"], predictions["G1-M5"], target,
            validation_direction_cpu, training_direction_cpu,
        )

        full_nmse = float(amplitude_metrics(predictions["candidate"], target)["nmse_db"])
        full_q4 = float(angular_distance_quartile_report(
            predictions["candidate"], target, validation_direction_cpu, training_direction_cpu
        )["Q4"]["nmse_db"])
        zero_out = {}
        for degree in range(1, 6):
            prediction = _predict(
                models["candidate"], commands, validation_direction,
                BranchAblation(disabled_ssh_degrees=(degree,)), args.chunk_size,
            )
            metrics = amplitude_metrics(prediction, target)
            q4_metrics = angular_distance_quartile_report(
                prediction, target, validation_direction_cpu, training_direction_cpu
            )["Q4"]
            zero_out[f"l{degree}"] = {
                "nmse_degradation_db": float(metrics["nmse_db"] - full_nmse),
                "mae_degradation_db": float(metrics["mae_db"] - amplitude_metrics(predictions["candidate"], target)["mae_db"]),
                "q4_nmse_degradation_db": float(q4_metrics["nmse_db"] - full_q4),
            }

        jacobian_relative_index, _ = stratified_separation_subset(
            validation_direction_cpu,
            per_cell=args.jacobian_per_cell,
            seed=21290,
            require_all_cells=False,
        )
        jacobian_direction = validation_direction_cpu[jacobian_relative_index].to(args.device)
        jacobian = {
            name: _jacobian_report(
                models[name], jacobian_direction, args.jacobian_step_rad, args.chunk_size
            )
            for name in ("G2-J", "candidate")
        }

        energy = {}
        degree_rms = {}
        for name in ("G2-J", "candidate"):
            ssh_iq, fourier_iq, degree_terms = _branch_terms(
                models[name], training_direction, args.chunk_size
            )
            transform = transforms[name]
            if transform is None:
                raise ValueError(f"SSH transform is missing for {runs[name]}")
            phi = transform.evaluate(training_direction_cpu.double())
            energy[name] = branch_energy_decomposition(
                phi, _iq_complex(ssh_iq), _iq_complex(fourier_iq)
            )
            degree_rms[name] = {
                key: float(value.square().mean().sqrt()) for key, value in degree_terms.items()
            }
        energy_change = {
            key: float(energy["candidate"][key] - energy["G2-J"][key]) for key in energy["G2-J"]
        }
        degree_change = {
            key: float(degree_rms["candidate"][key] - degree_rms["G2-J"][key])
            for key in degree_rms["G2-J"]
        }
        historical_branch = json.loads(
            (runs["candidate"] / "branch_reliance.json").read_text(encoding="utf-8")
        )
        per_seed.append({
            "seed": seed,
            "jacobian_sample": {
                "count": int(len(jacobian_relative_index)),
                "nonempty_validation_cells": int(
                    torch.unique(hemisphere_cells(validation_direction_cpu)[2]).numel()
                ),
                "source": "validation directions only; empty cells are not backfilled",
            },
            "distance_benefit": distance,
            "degree_zero_out": zero_out,
            "whole_branch_zero_out": {
                name: {
                    "nmse_degradation_db": float(
                        historical_branch["ablations"][name]["nmse_degradation_vs_full_db"]
                    ),
                    "source": "existing sealed post-hoc branch audit",
                }
                for name in ("ssh_zero", "fourier_zero")
            },
            "jacobian": jacobian,
            "energy_transfer": {
                "G2-J": energy["G2-J"],
                "candidate": energy["candidate"],
                "change_candidate_minus_joint": energy_change,
                "degree_rms_G2-J": degree_rms["G2-J"],
                "degree_rms_candidate": degree_rms["candidate"],
                "degree_rms_change_candidate_minus_joint": degree_change,
            },
        })
        print(f"finished seed {seed}", flush=True)

    slope = [row["distance_benefit"]["linear_slope_db_per_degree"] for row in per_seed]
    pearson = [row["distance_benefit"]["pearson"] for row in per_seed]
    spearman = [row["distance_benefit"]["spearman"] for row in per_seed]
    degree_summary = {}
    for degree in range(1, 6):
        key = f"l{degree}"
        degree_summary[key] = {
            metric: {
                "mean": float(np.mean(values := [row["degree_zero_out"][key][metric] for row in per_seed])),
                "bootstrap_95_ci": _bootstrap_ci(values),
                "positive_degradation_wins": int(np.sum(np.asarray(values) > 0)),
            }
            for metric in ("nmse_degradation_db", "mae_degradation_db", "q4_nmse_degradation_db")
        }
    whole_branch_summary = {
        name: {
            "nmse_degradation_db_mean": float(np.mean(values := [
                row["whole_branch_zero_out"][name]["nmse_degradation_db"] for row in per_seed
            ])),
            "bootstrap_95_ci": _bootstrap_ci(values),
            "positive_degradation_wins": int(np.sum(np.asarray(values) > 0)),
        }
        for name in ("ssh_zero", "fourier_zero")
    }
    jacobian_summary = {}
    for name in ("G2-J", "candidate"):
        jacobian_summary[name] = {
            key: {
                "mean_across_seeds": float(np.mean([row["jacobian"][name][key] for row in per_seed])),
                "bootstrap_95_ci": _bootstrap_ci([row["jacobian"][name][key] for row in per_seed]),
            }
            for key in (
                "fraction_ssh_raw_less_than_fourier",
                "fraction_ssh_normalized_less_than_fourier",
                "mean_raw_ratio_ssh_to_fourier",
                "mean_normalized_ratio_ssh_to_fourier",
            )
        }
    energy_keys = tuple(per_seed[0]["energy_transfer"]["change_candidate_minus_joint"])
    energy_summary = {
        key: {
            "mean_change_candidate_minus_joint": float(np.mean(values := [
                row["energy_transfer"]["change_candidate_minus_joint"][key] for row in per_seed
            ])),
            "bootstrap_95_ci": _bootstrap_ci(values),
            "direction_wins": int(np.sum(np.asarray(values) > 0)),
            "mean_relative_change_vs_joint": float(np.mean([
                row["energy_transfer"]["change_candidate_minus_joint"][key]
                / max(abs(row["energy_transfer"]["G2-J"][key]), 1e-12)
                for row in per_seed
            ])),
        }
        for key in energy_keys
    }
    payload = {
        "schema": "v212s-s2-frozen-representation-audit-1",
        "seeds": list(SEEDS),
        "external_test_used": False,
        "model_state_changed": False,
        "interpretation_boundary": {
            "a_and_branches": "latent representation / architectural specialization only",
            "physical_truth": "requires gauge-invariant Q_tilde audits",
        },
        "aggregate": {
            "distance_benefit": {
                "slope_db_per_degree_mean": float(np.mean(slope)),
                "slope_bootstrap_95_ci": _bootstrap_ci(slope),
                "positive_slope_wins": int(np.sum(np.asarray(slope) > 0)),
                "pearson_mean": float(np.mean(pearson)),
                "spearman_mean": float(np.mean(spearman)),
            },
            "degree_zero_out": degree_summary,
            "whole_branch_zero_out": whole_branch_summary,
            "jacobian": jacobian_summary,
            "energy_transfer": energy_summary,
        },
        "per_seed": per_seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _make_plot(payload, args.plot)
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
