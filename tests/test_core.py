"""Fast CPU checks; run python -m unittest discover -s tests -v."""

from dataclasses import replace
import unittest

import numpy as np
import torch

from levda_core.assimilation import AssimilationConfig, assimilate
from levda_core.metrics import evaluate_ensemble
from levda_core.networks import ResFourierLDNN
from levda_core.problems import PROBLEMS
from levda_core.runner import load_model, prepare_observations, set_seed


class CoreTests(unittest.TestCase):
    def setUp(self):
        set_seed(7)
        torch.set_num_threads(2)

    def test_checkpoint_compatibility(self):
        for problem in PROBLEMS.values():
            with self.subTest(problem=problem.name):
                model = load_model(problem, torch.device("cpu"))
                self.assertEqual(model.num_latent_states, problem.dynamics_sizes[-1])
                self.assertIs(model.B, model.rec.fourier_embedding)

    def test_decoder_chunking(self):
        model = ResFourierLDNN(3, [3, 8, 2], [4, 8, 8, 1])
        coords, states = torch.randn(3, 17, 2), torch.randn(3, 2)
        torch.testing.assert_close(model.decode(coords, states, 0), model.decode(coords, states, 4))

    def test_forward_and_smoothing(self):
        for current_background in (True, False):
            model = ResFourierLDNN(3, [3, 8, 2], [4, 8, 8, 1])
            data = {"x": torch.randn(1, 3, 16, 2), "u": torch.zeros(1, 1), "dt": torch.tensor(.1)}
            cfg = AssimilationConfig((4, 4), 2, ensemble_size=3, max_iter=2,
                                      current_observation_uses_background=current_background)
            observations = torch.randn(1, 3, 2, 2, 1)
            with torch.no_grad():
                self.assertEqual(model(data).shape, (1, 3, 16, 1))
            result = assimilate(model, data, observations, cfg, device="cpu", decode_chunk_size=5)
            self.assertEqual(result.shape, (3, 3, 16, 1))
            self.assertEqual(model.parameter_history.shape, (3, 3, 1))
            self.assertTrue(torch.isfinite(result).all())

    def test_metrics_distinguish_member_error(self):
        truth = np.ones((1, 2, 3, 1), dtype=np.float32)
        result = evaluate_ensemble(np.concatenate([truth * 2, truth * 4]), truth)
        np.testing.assert_allclose(result["relative_rmse"], 2)
        np.testing.assert_allclose(result["ser"], .5)
        np.testing.assert_allclose(result["member_rms_relative_rmse"], np.sqrt(5))
        result = evaluate_ensemble(np.full((2, 2, 3, 1), np.nan), truth)
        self.assertTrue(np.isnan(result["relative_rmse"]).all())

    def test_original_observation_draw_order(self):
        for name, problem in PROBLEMS.items():
            problem = replace(problem, assimilation=replace(problem.assimilation,
                                                             grid_shape=(4, 4), observation_stride=2))
            y = torch.arange(3 * 16 * 2, dtype=torch.float32).reshape(1, 3, 16, 2) / 20
            set_seed(42)
            if name == "kolmogorov_flow":
                expected = y + torch.randn_like(y) * .1 * y
                expected = expected.reshape(1, 3, 4, 4, 2)[:, :, ::2, ::2]
            else:
                expected = y.reshape(1, 3, 4, 4, 2)[:, :, ::2, ::2]
                expected = expected + .1 * torch.randn_like(expected) * expected
            set_seed(42)
            actual = prepare_observations({"y": y}, problem, "cpu")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
