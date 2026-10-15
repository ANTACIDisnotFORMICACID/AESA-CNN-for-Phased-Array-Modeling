"""Regression tests for prospective-data isolation and restartable training."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from v29.research.protocol import (PROTOCOL, block_keys, command_signature, make_codebook, write_json)
from v29.research.train import converged, ranking, run_training
from v29.research.evaluate import maximize_minimum, verify_freeze


class ResearchCampaignTests(unittest.TestCase):
    def test_frozen_budgets_and_model_seeds(self):
        self.assertEqual(PROTOCOL["rounds"], {"screen": 300, "confirm": 800, "final": 2000})
        self.assertEqual(PROTOCOL["model_seeds"], [29001, 29002, 29003])
        self.assertNotIn(PROTOCOL["split_seed"], PROTOCOL["model_seeds"])

    def test_codebook_counts_power_and_phase_invariant_uniqueness(self):
        command, family, partition = make_codebook()
        self.assertEqual(command.shape, (416, 4, 4))
        self.assertEqual(np.bincount(partition).tolist(), [292, 62, 62])
        np.testing.assert_allclose(np.sum(np.abs(command)**2, axis=(1, 2)), 1, atol=1e-6)
        self.assertLessEqual(float(np.max(np.abs(command))), 1)
        self.assertEqual(len(set(command_signature(w) for w in command)), 416)
        for family_name in np.unique(family):
            self.assertEqual(set(partition[family == family_name]), {0, 1, 2})
        np.testing.assert_array_equal(command, make_codebook()[0])

    def test_global_phase_duplicate_is_not_a_new_command(self):
        command = make_codebook()[0][40]
        self.assertEqual(command_signature(command), command_signature(command*np.exp(0.42j)))

    def test_absolute_block_origin_matches_both_grids(self):
        np.testing.assert_array_equal(block_keys(np.array([1., 11., 171.]), np.array([-179., 1., 179.])),
                                      block_keys(np.array([1.5, 11.5, 171.5]), np.array([-178.5, 1.5, 179.5])))
        self.assertNotEqual(block_keys(np.array([9.5]), np.array([0.5]))[0],
                            block_keys(np.array([10.5]), np.array([0.5]))[0])

    def test_convergence_requires_minimum_epochs_and_true_plateau(self):
        flat = [{"best_nmse_db": -10.0} for _ in range(400)]
        self.assertTrue(converged(flat))
        self.assertFalse(converged(flat[:399]))
        improving = [{"best_nmse_db": -10-i*.002} for i in range(400)]
        self.assertFalse(converged(improving))

    def test_recipe_selection_uses_validation_only(self):
        rows = [{"recipe": "R1", "validation": {"global_amplitude_nmse_db": -10,
                  "global_power_nmse_db": -10, "amplitude_db_mae": 1}, "test": -100},
                {"recipe": "R2", "validation": {"global_amplitude_nmse_db": -11,
                  "global_power_nmse_db": -10, "amplitude_db_mae": 1}, "test": 100}]
        self.assertEqual(ranking(rows), ["R2", "R1"])

    def test_blind_requires_final_freeze(self):
        with tempfile.TemporaryDirectory() as name, patch("v29.research.evaluate.assert_protocol", return_value={}):
            with self.assertRaises(FileNotFoundError):
                verify_freeze(Path(name))

    def test_training_forbidden_after_blind_open(self):
        with tempfile.TemporaryDirectory() as name, patch("v29.research.train.assert_protocol", return_value={}):
            root = Path(name)
            write_json(root/"blind_opened.json", {})
            with self.assertRaisesRegex(RuntimeError, "closed"):
                run_training(root, "final", "R1", 29001, 2000, "cpu")

    def test_final_budget_cannot_silently_be_reduced(self):
        with tempfile.TemporaryDirectory() as name, patch("v29.research.train.assert_protocol", return_value={}):
            with self.assertRaisesRegex(ValueError, "budget"):
                run_training(Path(name), "final", "R1", 29001, 1000, "cpu")

    def test_multistart_reference_respects_command_constraint_and_initial_bound(self):
        torch.manual_seed(1)
        h = torch.randn(4, 16, dtype=torch.complex64)
        initial = torch.ones(16, dtype=torch.complex64)/4
        initial_value = float((h@initial).abs().square().min())
        w, value = maximize_minimum(h, 123, initial, steps=20, restarts=4)
        self.assertAlmostEqual(float(w.norm()), 1, places=5)
        self.assertGreaterEqual(value+1e-6, initial_value)

    def test_interrupted_resume_matches_uninterrupted_cpu_run(self):
        rng = np.random.default_rng(99)
        commands = np.exp(1j*rng.uniform(-np.pi, np.pi, (4, 4, 4)))/4
        iq = np.stack((commands.real, commands.imag), axis=1).astype(np.float32)
        direction = rng.normal(size=(8, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        amplitude = np.abs(rng.normal(size=(4, 8))).astype(np.float32)+.1
        options = {"batch_size": 4, "queries_per_epoch": 8, "validation_batch_size": 4,
                   "checkpoint_every": 2}
        with tempfile.TemporaryDirectory() as name:
            roots = [Path(name)/"continuous", Path(name)/"resumed"]
            for root in roots:
                write_json(root/"protocol.json", {})
                (root/"dev").mkdir()
                for split in ("train", "validation"):
                    np.savez(root/"dev"/(split+".npz"), direction=direction, command_iq=iq,
                             family=np.array(["a", "a", "b", "b"]), amplitude=amplitude)
            run_training(roots[0], "screen", "R1", 29001, 4, "cpu", test_options=options)
            from v29.research.train import atomic_torch_save
            def interrupt_after_save(data, path):
                atomic_torch_save(data, path)
                if path.name == "latest.pt" and len(data["history"]) == 2:
                    raise RuntimeError("simulated interruption")
            with patch("v29.research.train.atomic_torch_save", side_effect=interrupt_after_save):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    run_training(roots[1], "screen", "R1", 29001, 4, "cpu", test_options=options)
            run_training(roots[1], "screen", "R1", 29001, 4, "cpu", test_options=options)
            a, b = [torch.load(root/"runs/screen/R1_seed29001/latest.pt", weights_only=False) for root in roots]
            self.assertEqual(a["history"], b["history"])
            for key in a["model_state"]:
                torch.testing.assert_close(a["model_state"][key], b["model_state"][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
