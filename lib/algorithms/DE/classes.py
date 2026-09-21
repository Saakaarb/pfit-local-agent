import time
from pathlib import Path

import numpy as np

from lib.utils.classes import ProblemObjectBase
from lib.utils.yamlread import YAMLReader


class FitParamsDE:
    def __init__(self, input_reader: YAMLReader, problem_object: ProblemObjectBase):
        self.problem_obj = problem_object
        self.input_reader = input_reader
        self.random_seed = getattr(input_reader, "random_seed", None)
        if self.random_seed is None:
            self.random_seed = 42

        self.min_search_list = []
        self.max_search_list = []
        self.min_search_axis = []
        self.max_search_axis = []
        self.axis_logscale = input_reader.axis_logscale

        for i_axis in range(input_reader.n_search_axes):
            if input_reader.axis_logscale[i_axis]:
                curr_axis_min = np.log10(input_reader.min_axis_values[i_axis])
                curr_axis_max = np.log10(input_reader.max_axis_values[i_axis])
            else:
                curr_axis_min = input_reader.min_axis_values[i_axis]
                curr_axis_max = input_reader.max_axis_values[i_axis]

            self.min_search_axis.append(curr_axis_min)
            self.max_search_axis.append(curr_axis_max)
            self.min_search_list.append(
                self.scale_value(curr_axis_min, curr_axis_min, curr_axis_max)
            )
            self.max_search_list.append(
                self.scale_value(curr_axis_max, curr_axis_min, curr_axis_max)
            )

        self.n_search_axes = input_reader.n_search_axes
        self.best_pos = None
        self.best_cost = None

    def scale_value(self, val: float, min_val: float, max_val: float) -> float:
        return 2 * (val - min_val) / (max_val - min_val) - 1

    def unscale_value(self, val: float, min_val: float, max_val: float) -> float:
        return (1 + val) * (max_val - min_val) / 2 + min_val

    def unscale_design_point(self, position: np.ndarray) -> np.ndarray:
        if not isinstance(position, np.ndarray):
            raise ValueError("Query point to unscale_design_point must be a np array")

        unscaled_position = np.zeros_like(position)
        for i_axis in range(self.input_reader.n_search_axes):
            if self.axis_logscale[i_axis]:
                unscaled_position[i_axis] = 10 ** (
                    self.unscale_value(
                        position[i_axis],
                        self.min_search_axis[i_axis],
                        self.max_search_axis[i_axis],
                    )
                )
            else:
                unscaled_position[i_axis] = self.unscale_value(
                    position[i_axis],
                    self.min_search_axis[i_axis],
                    self.max_search_axis[i_axis],
                )
        return unscaled_position

    def run(self, file_obj: Path) -> tuple[np.ndarray, float]:
        bounds = [(-1.0, 1.0)] * self.n_search_axes
        n_pop = self.input_reader.n_particles
        rng = np.random.default_rng(self.random_seed)
        population = self._latin_hypercube(n_pop, self.n_search_axes, rng)
        costs = self.problem_obj.compute_all_losses(population)
        best_idx = int(np.argmin(costs))
        best_pos = population[best_idx].copy()
        best_cost = float(costs[best_idx])

        print(f"DE population size: {n_pop}, max iterations: {self.input_reader.n_iters_pop}")
        with open(file_obj, "a") as f:
            f.write(f"Population size: {n_pop}\n")
            f.write(f"Max iterations: {self.input_reader.n_iters_pop}\n\n")

        for iter_no in range(self.input_reader.n_iters_pop):
            t1 = time.time()
            f_scale = rng.uniform(0.5, 1.0)
            trial_population = np.empty_like(population)

            for i in range(n_pop):
                candidates = np.delete(np.arange(n_pop), i)
                r1, r2 = rng.choice(candidates, size=2, replace=False)
                mutant = best_pos + f_scale * (population[r1] - population[r2])
                mutant = np.clip(mutant, -1.0, 1.0)

                crossover = rng.random(self.n_search_axes) < 0.7
                crossover[rng.integers(self.n_search_axes)] = True
                trial_population[i] = np.where(crossover, mutant, population[i])

            trial_costs = self.problem_obj.compute_all_losses(trial_population)
            improved = trial_costs < costs
            population[improved] = trial_population[improved]
            costs[improved] = trial_costs[improved]

            best_idx = int(np.argmin(costs))
            best_pos = population[best_idx].copy()
            best_cost = float(costs[best_idx])

            iteration_time = time.time() - t1
            print(f"Iteration: {iter_no + 1}")
            print(f"Best Cost: {best_cost:.4E}")
            print(f"Time for iteration: {iteration_time:.4f}")
            print("-------")
            with open(file_obj, "a") as f:
                f.write(f"{iter_no + 1}, {best_cost:.4E}, {iteration_time:.4f}\n")
                f.flush()

            stop_flag = Path(self.input_reader.output_dir) / "stop_fitting.flag"
            if stop_flag.exists():
                break

        self.best_pos = best_pos
        self.best_cost = best_cost
        print(f"Final best position (scaled): {self.best_pos}")
        print(f"Final best cost: {self.best_cost:.4E}")
        print("Done with DE search")
        return self.best_pos, self.best_cost

    def _latin_hypercube(
        self, n_samples: int, n_dimensions: int, rng: np.random.Generator
    ) -> np.ndarray:
        samples = np.empty((n_samples, n_dimensions))
        for dim in range(n_dimensions):
            permutation = rng.permutation(n_samples)
            samples[:, dim] = (permutation + rng.random(n_samples)) / n_samples
        return 2.0 * samples - 1.0
