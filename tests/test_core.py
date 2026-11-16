"""Fast CPU checks; run python -m unittest discover -s tests -v."""

from dataclasses import replace
import unittest

import numpy as np
import torch

from levda_core.assimilation import AssimilationConfig, assimilate
from levda_core.metrics import evaluate_ensemble
from levda_core.networks import ResFourierLDNN, ResLDNN
from levda_core.runner import average, load_model, prepare_observations, set_seed
from kolmogorov_flow.problem import PROBLEM as KOLMOGOROV
from planetswe.problem import PROBLEM as PLANETSWE
from shallow_water.problem import PROBLEM as SHALLOW_WATER


PROBLEMS = (KOLMOGOROV, PLANETSWE, SHALLOW_WATER)


class CoreTests(unittest.TestCase):
    def setUp(self):
        set_seed(7)
        torch.set_num_threads(2)

    def test_checkpoint_compatibility(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem.name):
                model = load_model(problem, torch.device("cpu"))
                self.assertEqual(model.num_latent_states, problem.dynamics_sizes[-1])
                self.assertEqual(model.num_outputs, problem.decoder_sizes[-1])
                if problem.fourier_size is not None:
                    self.assertIs(model.B, model.rec.fourier_embedding)
                else:
                    self.assertIsInstance(model, ResLDNN)

    def test_decoder_chunking(self):
        coords, states = torch.randn(3, 17, 2), torch.randn(3, 2)
        for model in (ResFourierLDNN(3, [3, 8, 2], [4, 8, 8, 1]),
                      ResLDNN([3, 8, 2], [4, 8, 8, 1])):
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
        for problem in PROBLEMS:
            problem = replace(problem, assimilation=replace(problem.assimilation,
                                                             grid_shape=(4, 4), observation_stride=2))
            y = torch.arange(3 * 16 * 2, dtype=torch.float32).reshape(1, 3, 16, 2) / 20
            set_seed(42)
            if problem.name == "kolmogorov_flow":
                expected = y + torch.randn_like(y) * .1 * y
                expected = expected.reshape(1, 3, 4, 4, 2)[:, :, ::2, ::2]
            else:
                expected = y.reshape(1, 3, 4, 4, 2)[:, :, ::2, ::2]
                if problem.name == "shallow_water":
                    expected = expected + expected * .1 * torch.randn_like(expected)
                else:
                    expected = expected + .1 * torch.randn_like(expected) * expected
            set_seed(42)
            actual = prepare_observations({"y": y}, problem, "cpu")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_shallow_water_complete_window_and_parameter_history(self):
        model = ResLDNN([3, 8, 2], [4, 8, 8, 1])
        data = {"x": torch.randn(1, 4, 16, 2), "u": torch.zeros(1, 4, 1),
                "dt": torch.tensor(.1)}
        config = replace(SHALLOW_WATER.assimilation, grid_shape=(4, 4),
                         observation_stride=2, ensemble_size=3, max_iter=2,
                         smoothing_steps=2)
        observations = torch.randn(1, 4, 2, 2, 1)
        result = assimilate(model, data, observations, config, device="cpu")
        self.assertEqual(result.shape, (3, 4, 16, 1))
        self.assertEqual(model.analysis_indices, [0, 1])
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(model.forecast_parameter_history[:, 1:],
                                   model.parameter_history[:, :-1], rtol=0, atol=0)
        for index in (2, 3):
            torch.testing.assert_close(model.parameter_history[:, index],
                                       model.parameter_history[:, 1], rtol=0, atol=0)
        with torch.no_grad():
            expected = model.state_history[-2] + data["dt"] * model.dyn(
                torch.cat((model.parameter_history[:, -1], model.state_history[-2]), dim=1))
        torch.testing.assert_close(model.state_history[-1], expected, rtol=0, atol=0)

    def test_shallow_water_forecast_only_short_trajectory(self):
        model = ResLDNN([3, 8, 2], [4, 8, 8, 1])
        data = {"x": torch.randn(1, 2, 16, 2), "u": torch.zeros(1, 2, 1),
                "dt": torch.tensor(.1)}
        config = replace(SHALLOW_WATER.assimilation, grid_shape=(4, 4),
                         observation_stride=2, ensemble_size=3, max_iter=1)
        result = assimilate(model, data, torch.randn(1, 2, 2, 2, 1), config, device="cpu")
        self.assertEqual(model.analysis_indices, [])
        self.assertTrue(torch.isfinite(result).all())

    def test_shallow_water_zero_noise_does_not_consume_random_draws(self):
        data = {"y": torch.ones(1, 1, 22500, 3)}
        rng = torch.get_rng_state()
        observations = prepare_observations(data, SHALLOW_WATER, "cpu", noise=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(observations.shape, (1, 1, 10, 10, 3))

    def test_normalization(self):
        values = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
        low = np.array([-.2, -.2, -.5], dtype=np.float32)
        high = np.array([.2, .2, 1.], dtype=np.float32)
        expected = .5 * (low + high + (high - low) * values)
        np.testing.assert_array_equal(SHALLOW_WATER.load_normalization().normalize_back(values), expected)
        for problem in (KOLMOGOROV, PLANETSWE):
            with np.load(problem.folder / "saved_model/mean_std.npz") as stats:
                probe = values[..., :problem.decoder_sizes[-1]]
                expected = probe * stats["std"] + stats["mean"]
            np.testing.assert_array_equal(problem.load_normalization().normalize_back(probe), expected)

    def test_initialization_averaging_preserves_nonfinite_policy(self):
        values = [[1., np.nan], [3., np.inf]]
        np.testing.assert_equal(average(values, finite_only=True), [2., np.nan])
        self.assertTrue(np.isnan(average(values)[1]))
        self.assertTrue(np.isnan(average([np.nan, np.inf], finite_only=True)))


if __name__ == "__main__":
    unittest.main()
