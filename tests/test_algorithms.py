"""Regression checks for the standalone introductory algorithms."""
import contextlib
import importlib.util
import io
import itertools
from pathlib import Path
import random
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


operators = load_module('operators', 'Lab4/operators.py')
solutions = [load_module(f'solution_{n}', f'Lab5/solution_{n}.py') for n in [1, 2]]
partition = load_module('partition', 'Lab6/brute-force.py')


class OperatorTests(unittest.TestCase):
    def test_crossovers_preserve_permutations_and_parents(self):
        random.seed(42)
        for size in [2, 3, 5, 10]:
            parent1 = list(range(size))
            parent2 = parent1[::-1]
            for _ in range(50):
                children = [
                    operators.one_point_crossover(parent1, parent2),
                    operators.two_point_crossover(parent1, parent2),
                    operators.n_point_crossover(parent1, parent2, size - 1),
                    operators.uniform_crossover(parent1, parent2),
                ]
                for child in children:
                    self.assertEqual(sorted(child), parent1)
                self.assertEqual(parent2, parent1[::-1])

    def test_small_permutations(self):
        for parent in [[], [0]]:
            for crossover in [operators.one_point_crossover, operators.two_point_crossover]:
                self.assertEqual(crossover(parent, parent), parent)
                self.assertIsNot(crossover(parent, parent), parent)

    def test_two_point_can_copy_last_gene(self):
        with patch.object(operators.random, 'sample', return_value=[2, 3]):
            self.assertEqual(operators.two_point_crossover([0, 1, 2], [2, 1, 0]), [1, 0, 2])

    def test_small_and_equal_cost_populations(self):
        population = [SimpleNamespace(fitness=5) for _ in range(3)]
        for selector in [operators.tournament_selection, operators.roulette_wheel_selection,
                         operators.rank_selection, lambda p: operators.boltzmann_selection(p, 1000)]:
            for _ in range(20):
                selected = selector(population)
                self.assertTrue(any(selected is c for c in population))

    def test_zero_cost_and_invalid_roulette(self):
        optimum = SimpleNamespace(fitness=0)
        self.assertIs(operators.roulette_wheel_selection([SimpleNamespace(fitness=5), optimum]), optimum)
        for cost in [-1, float('inf'), float('nan')]:
            with self.assertRaises(ValueError):
                operators.roulette_wheel_selection([SimpleNamespace(fitness=cost)])

    def test_selection_favors_lower_costs(self):
        population = [SimpleNamespace(fitness=x) for x in [1, 10, 100]]
        for selector in [operators.roulette_wheel_selection, operators.rank_selection,
                         lambda p: operators.boltzmann_selection(p, 1000)]:
            random.seed(12)
            counts = {1: 0, 10: 0, 100: 0}
            for _ in range(4000):
                counts[selector(population).fitness] += 1
            self.assertGreater(counts[1], counts[10])
            self.assertGreater(counts[10], counts[100])


class KnapsackTests(unittest.TestCase):
    def test_zero_generations_considers_every_initial_member(self):
        for module in solutions:
            count = len(module.ITEMS if hasattr(module, 'ITEMS') else module.PROJECTS)
            empty = module.Chromosome([False] * count)
            selected = module.Chromosome([True] + [False] * (count - 1))
            with patch.object(module, 'create_population', return_value=[empty, selected]):
                self.assertEqual(module.genetic_algorithm(2, 0).fitness, selected.fitness)

    def test_final_generation_is_evaluated(self):
        for module in solutions:
            count = len(module.ITEMS if hasattr(module, 'ITEMS') else module.PROJECTS)
            empty = module.Chromosome([False] * count)

            def improve(chromosome):
                chromosome.selected[0] = True
                chromosome.fitness = chromosome.calculate_fitness()

            with patch.object(module, 'create_population', return_value=[empty, empty]), \
                 patch.object(module.Chromosome, 'mutate', improve), contextlib.redirect_stdout(io.StringIO()):
                self.assertGreater(module.genetic_algorithm(2, 1).fitness, 0)

    def test_infeasible_population_returns_feasible_empty_selection(self):
        for module in solutions:
            count = len(module.ITEMS if hasattr(module, 'ITEMS') else module.PROJECTS)
            infeasible = module.Chromosome([True] * count)
            self.assertEqual(infeasible.fitness, 0)
            with patch.object(module, 'create_population', return_value=[infeasible]):
                result = module.genetic_algorithm(1, 0)
                self.assertFalse(any(result.selected))

    def test_small_populations_remain_feasible(self):
        for module in solutions:
            for size in [1, 2, 5, 12]:
                random.seed(42)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = module.genetic_algorithm(size, 5)
                decoded = result.decode()
                self.assertEqual(result.fitness, result.calculate_fitness())
                if hasattr(module, 'CAPACITY'):
                    self.assertLessEqual(decoded[1], module.CAPACITY)
                else:
                    self.assertLessEqual(decoded[1], module.TOTAL_BUDGET)
                    self.assertLessEqual(decoded[2], module.TOTAL_TIME)

    def test_importing_generator_does_not_write_files(self):
        with patch('builtins.open', side_effect=AssertionError('Unexpected file write')), \
             patch.object(Path, 'open', side_effect=AssertionError('Unexpected file write')):
            load_module('instance_generator', 'Lab5/instance_generator.py')

    def test_generator_refuses_to_overwrite_instances(self):
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'open') as open_file, \
             patch('builtins.open') as builtin_open:
            with self.assertRaises(FileExistsError):
                runpy.run_path(str(ROOT / 'Lab5/instance_generator.py'), run_name='__main__')
            open_file.assert_not_called()
            builtin_open.assert_not_called()


class PartitionTests(unittest.TestCase):
    def test_partition_matches_exhaustive_optimum(self):
        for numbers in [[], [10**12], [1, 2, 4], [4, 5, 8, 9], [-4, 1, 7]]:
            a, b, difference = partition.find_partition(numbers)
            optimum = min(abs(sum(x * sign for x, sign in zip(numbers, signs)))
                          for signs in itertools.product([-1, 1], repeat=len(numbers)))
            self.assertEqual(sorted(a + b), sorted(numbers))
            self.assertEqual(difference, optimum)


if __name__ == '__main__':
    unittest.main()
