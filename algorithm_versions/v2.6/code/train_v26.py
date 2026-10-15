"""Train the 15-channel V2.6 residual CNN against exact HFSS-oracle shards."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from controller_v26 import VitisBeamformingCNNV26
from hfss_physics_v26 import HfssPatternData, unit_vector_hfss


V26_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = V26_ROOT / "output"


class TeacherShardDataset(Dataset):
    def __init__(self, shard_paths: Iterable[Path], data: HfssPatternData):
        arrays = {}
        required = (
            "deployment_x",
            "initial_weights",
            "teacher_weights",
            "target_hfss_deg",
            "calibration",
            "health_mask",
            "oracle_metrics",
        )
        loaded = {key: [] for key in required}
        for path in shard_paths:
            with np.load(path) as shard:
                for key in required:
                    loaded[key].append(np.asarray(shard[key]))
        if not loaded["deployment_x"]:
            raise ValueError("no teacher shards were provided")
        self.arrays = {key: np.concatenate(value, axis=0) for key, value in loaded.items()}
        ht, hp = [], []
        for theta, phi in self.arrays["target_hfss_deg"]:
            a, b = data.response_at(float(theta), float(phi))
            ht.append(a)
            hp.append(b)
        self.arrays["target_ht"] = np.asarray(ht, dtype=np.complex64)
        self.arrays["target_hp"] = np.asarray(hp, dtype=np.complex64)

    def __len__(self) -> int:
        return len(self.arrays["deployment_x"])

    def __getitem__(self, index: int):
        return {key: torch.from_numpy(np.asarray(value[index])) for key, value in self.arrays.items()}


@dataclass(frozen=True)
class LossWeights:
    teacher: float
    target: float
    lobe: float
    int8: float


FIXED_LOSS_GRID = (
    LossWeights(1.0, 0.25, 0.50, 0.05),
    LossWeights(1.0, 0.50, 0.50, 0.05),
    LossWeights(1.0, 0.25, 1.00, 0.10),
)


class ExactHfssLoss(nn.Module):
    def __init__(self, data: HfssPatternData, weights: LossWeights, residual_scale: float = 0.5):
        super().__init__()
        self.weights = weights
        self.residual_scale = residual_scale
        self.register_buffer("htheta", torch.from_numpy(data.etheta.reshape(4, -1).astype(np.complex64)))
        self.register_buffer("hphi", torch.from_numpy(data.ephi.reshape(4, -1).astype(np.complex64)))
        theta, phi = data.flat_theta_phi
        self.register_buffer("grid_unit", torch.from_numpy(unit_vector_hfss(theta, phi).astype(np.float32)))
        self.cos_exclusion = float(np.cos(np.deg2rad(8.0)))

    @staticmethod
    def _project(weights: torch.Tensor) -> torch.Tensor:
        magnitude = torch.abs(weights)
        return weights * torch.clamp(1.0 / torch.clamp(magnitude, min=1e-12), max=1.0)

    def command_from_delta(self, x: torch.Tensor, delta_iq: torch.Tensor) -> torch.Tensor:
        initial = torch.complex(x[:, 0, 0], x[:, 1, 0])
        delta = torch.complex(delta_iq[:, 0, 0], delta_iq[:, 1, 0])
        health = x[:, 4, 0]
        return self._project(initial * (1.0 + self.residual_scale * delta)) * health

    def pattern_terms(self, command: torch.Tensor, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        calibration = batch["calibration"].to(command.device)
        health = batch["health_mask"].to(command.device)
        actual = command * calibration * health
        et = torch.einsum("bn,nk->bk", actual, self.htheta)
        ep = torch.einsum("bn,nk->bk", actual, self.hphi)
        full_power = torch.abs(et) ** 2 + torch.abs(ep) ** 2
        target_et = torch.sum(actual * batch["target_ht"].to(command.device), dim=1)
        target_ep = torch.sum(actual * batch["target_hp"].to(command.device), dim=1)
        target_power = torch.abs(target_et) ** 2 + torch.abs(target_ep) ** 2
        target_unit = torch.from_numpy(
            unit_vector_hfss(
                batch["target_hfss_deg"][:, 0].detach().cpu().numpy(),
                batch["target_hfss_deg"][:, 1].detach().cpu().numpy(),
            ).astype(np.float32)
        ).to(command.device)
        outside = target_unit @ self.grid_unit.T < self.cos_exclusion
        outside_peak = full_power.masked_fill(~outside, -torch.inf).amax(dim=1)
        margin_db = 10.0 * torch.log10((target_power + 1e-20) / (outside_peak + 1e-20))
        return target_power, outside_peak, margin_db

    def forward(self, x: torch.Tensor, delta_iq: torch.Tensor, batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = self.command_from_delta(x, delta_iq)
        teacher = batch["teacher_weights"].to(prediction.device)
        target_power, _, margin = self.pattern_terms(prediction, batch)
        teacher_target, _, teacher_margin = self.pattern_terms(teacher, batch)
        teacher_loss = torch.mean(torch.abs(prediction - teacher) ** 2)
        target_regret = torch.relu(10.0 * torch.log10((teacher_target + 1e-20) / (target_power + 1e-20))).mean()
        lobe_regret = torch.relu(teacher_margin - margin).mean()
        # Straight-through W8 output consistency proxy; exact integer disk is evaluated separately.
        quantized = torch.complex(torch.round(prediction.real * 127.0) / 127.0, torch.round(prediction.imag * 127.0) / 127.0)
        quantized = prediction + (self._project(quantized) - prediction).detach()
        int8_loss = torch.mean(torch.abs(prediction - quantized) ** 2)
        total = (
            self.weights.teacher * teacher_loss
            + self.weights.target * target_regret
            + self.weights.lobe * lobe_regret
            + self.weights.int8 * int8_loss
        )
        terms = {
            "total": total.detach(),
            "teacher": teacher_loss.detach(),
            "target_regret_db": target_regret.detach(),
            "lobe_regret_db": lobe_regret.detach(),
            "int8": int8_loss.detach(),
            "margin_p05_db": torch.quantile(margin.detach(), 0.05),
        }
        return total, terms


def evaluate(model, loader, criterion, device: str) -> dict:
    model.eval()
    all_terms = []
    false_feasible = 0
    sample_count = 0
    target_regrets = []
    margins = []
    with torch.no_grad():
        for batch in loader:
            x = batch["deployment_x"].to(device)
            delta = model(x)
            _, terms = criterion(x, delta, batch)
            command = criterion.command_from_delta(x, delta)
            target, outside, margin = criterion.pattern_terms(command, batch)
            teacher_feasible = batch["oracle_metrics"][:, 3].to(device) > 0.5
            predicted_feasible = margin >= 3.0
            false_feasible += int(torch.sum(predicted_feasible & ~teacher_feasible))
            teacher_target, _, _ = criterion.pattern_terms(batch["teacher_weights"].to(device), batch)
            target_regrets.extend(torch.relu(10.0 * torch.log10((teacher_target + 1e-20) / (target + 1e-20))).cpu().tolist())
            margins.extend(margin.cpu().tolist())
            all_terms.append({key: float(value.cpu()) for key, value in terms.items()})
            sample_count += len(x)
    return {
        "samples": sample_count,
        "false_feasible": false_feasible,
        "median_target_power_regret_db": float(np.median(target_regrets)),
        "margin_p05_db": float(np.percentile(margins, 5)),
        "mean_loss": float(np.mean([item["total"] for item in all_terms])),
    }


def train_candidate(train_data, validation_data, data, weights, epochs, batch_size, device, run_name):
    torch.manual_seed(26006)
    model = VitisBeamformingCNNV26().to(device)
    criterion = ExactHfssLoss(data, weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(26006))
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    history = []
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["deployment_x"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = criterion(x, model(x), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        metrics = evaluate(model, validation_loader, criterion, device)
        metrics["epoch"] = epoch + 1
        history.append(metrics)
        print(run_name, json.dumps(metrics, sort_keys=True))
    return model, criterion, history[-1], history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-glob", default=str(OUTPUT_ROOT / "teacher" / "train_*.npz"))
    parser.add_argument("--validation-glob", default=str(OUTPUT_ROOT / "teacher" / "validation_*.npz"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_paths = sorted(Path().glob(args.train_glob)) if not Path(args.train_glob).is_absolute() else sorted(Path(args.train_glob).parent.glob(Path(args.train_glob).name))
    val_paths = sorted(Path().glob(args.validation_glob)) if not Path(args.validation_glob).is_absolute() else sorted(Path(args.validation_glob).parent.glob(Path(args.validation_glob).name))
    data = HfssPatternData.load()
    train_data = TeacherShardDataset(train_paths, data)
    val_data = TeacherShardDataset(val_paths, data)
    candidates = []
    for index, weights in enumerate(FIXED_LOSS_GRID):
        model, criterion, metrics, history = train_candidate(
            train_data, val_data, data, weights, args.epochs, args.batch_size, args.device, f"candidate_{index}"
        )
        rank = (metrics["false_feasible"], -metrics["margin_p05_db"], metrics["median_target_power_regret_db"])
        candidates.append((rank, index, model, metrics, history, weights))
    candidates.sort(key=lambda item: item[0])
    _, index, model, metrics, history, weights = candidates[0]
    checkpoint = OUTPUT_ROOT / "checkpoint" / "v26_fp32.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "model_state_dict": model.state_dict(),
            "model": {"input_channels": 15, "width": 16, "blocks": 3},
            "loss_weights": weights.__dict__,
            "selection_order": ["zero false feasible", "strong-peak margin", "target gain"],
            "validation_metrics": metrics,
            "history": history,
        },
        checkpoint,
    )
    report = {
        "selected_candidate": index,
        "checkpoint": str(checkpoint),
        "validation": metrics,
        "all_candidates": [
            {"index": item[1], "weights": item[5].__dict__, "metrics": item[3]} for item in candidates
        ],
    }
    report_path = OUTPUT_ROOT / "reports" / "training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

