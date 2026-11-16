"""Exercise the shared CLI and each problem's initialization conventions."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from kolmogorov_flow.problem import PROBLEM as KOLMOGOROV
from planetswe.problem import PROBLEM as PLANETSWE
from shallow_water.problem import PROBLEM as SHALLOW_WATER
from levda_core import runner


class RunnerTests(unittest.TestCase):
    def test_all_problem_runners_share_observations_and_keep_seed_conventions(self):
        for problem in (KOLMOGOROV, PLANETSWE, SHALLOW_WATER):
            with self.subTest(problem=problem.name):
                config = replace(problem.assimilation, grid_shape=(4, 4), observation_stride=2)
                problem = replace(problem, fourier_size=None, dynamics_sizes=(3, 8, 2),
                                  decoder_sizes=(4, 8, 8, 3), assimilation=config,
                                  normalization_bounds=((0., 0., 0.), (1., 1., 1.)))
                data = {"x": torch.zeros(1, 2, 16, 2), "y": torch.ones(1, 2, 16, 3),
                        "u": torch.zeros(1, 1), "dt": torch.tensor(.1)}
                observation_draws = []
                assimilate = runner.assimilate

                def record_observations(model, model_data, observations, *args, **kwargs):
                    observation_draws.append(observations.clone())
                    return assimilate(model, model_data, observations, *args, **kwargs)

                def build_model(problem, device, *args):
                    return problem.build_model(device).eval()

                with patch.object(runner, "load_data", return_value=data), \
                     patch.object(runner, "load_model", side_effect=build_model), \
                     patch.object(runner, "set_seed", wraps=runner.set_seed) as seeds, \
                     patch.object(runner, "assimilate", side_effect=record_observations), \
                     redirect_stdout(io.StringIO()) as output:
                    metrics = runner.run(problem, ["--device", "cpu", "--num-inits", "2",
                                                   "--ensemble-size", "3", "--max-iter", "1",
                                                   "--smoothing-steps", "1", "--threads", "2"])
                expected_seeds = [42, 43, 44] if problem.seed_before_model else [42, 43]
                self.assertEqual([call.args[0] for call in seeds.call_args_list], expected_seeds)
                self.assertEqual(len(observation_draws), 2)
                torch.testing.assert_close(*observation_draws, rtol=0, atol=0)
                self.assertTrue(np.isfinite(metrics["relative_rmse"]).all())
                self.assertIn("Average relative RMSE over 2 initialization(s)", output.getvalue())


if __name__ == "__main__":
    unittest.main()
