"""Small deterministic regression checks for the TSP search notebooks."""
import contextlib
import io
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]


def load_functions(relative):
    namespace = {'np': np, 'random': random}
    notebook = json.loads((ROOT / relative).read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and source.startswith('def '):
            exec(compile(source, relative, 'exec'), namespace)
    return namespace


ga = load_functions('Lab7/GA_k_opt.ipynb')
sa = load_functions('Lab8/sa.ipynb')
ts = load_functions('Lab8/ts.ipynb')


class GeneticSearchTests(unittest.TestCase):
    def test_local_search_can_move_the_last_city(self):
        route = [0, 1, 3, 2]
        for method in ['two_opt', 'three_opt']:
            with self.subTest(method=method):
                result = ga[method](route, SQUARE)
                self.assertEqual(sorted(result), [0, 1, 2, 3])
                self.assertAlmostEqual(ga['calculate_tour_distance'](result, SQUARE), 4)
                self.assertEqual(route, [0, 1, 3, 2])

    def test_final_offspring_is_evaluated(self):
        with patch.dict(ga, {
            'initialize_population': lambda size, count: [[0, 2, 1, 3] for _ in range(size)],
            'order_crossover': lambda p1, p2: [0, 1, 2, 3],
        }):
            route, cost, history = ga['genetic_algorithm'](
                SQUARE, 0, pop_size=3, generations=1, mutation_rate=0)
        self.assertEqual(route, [0, 1, 2, 3])
        self.assertEqual(cost, 4)
        self.assertEqual(len(history), 2)
        self.assertGreater(history[0], history[1])
        self.assertEqual(history[-1], cost)

    def test_history_records_generations_without_improvement(self):
        with patch.dict(ga, {
            'initialize_population': lambda size, count: [[0, 1, 2, 3] for _ in range(size)],
            'order_crossover': lambda p1, p2: [0, 1, 2, 3],
        }):
            _, cost, history = ga['genetic_algorithm'](
                SQUARE, 0, pop_size=3, generations=3, mutation_rate=0)
        self.assertEqual(cost, 4)
        self.assertEqual(history, [4, 4, 4, 4])

    def test_zero_generations_and_final_local_search(self):
        with patch.dict(ga, {
            'initialize_population': lambda size, count: [[0, 1, 3, 2] for _ in range(size)],
        }), contextlib.redirect_stdout(io.StringIO()):
            _, cost, history = ga['genetic_algorithm'](
                SQUARE, 0, pop_size=3, generations=0, final_3opt=True)
        self.assertEqual(cost, 4)
        self.assertEqual(len(history), 2)
        self.assertGreater(history[0], history[1])
        self.assertEqual(history[-1], cost)


class SearchInputAndTabuTests(unittest.TestCase):
    def test_coordinates_keep_fractional_parts(self):
        for namespace in [ga, sa, ts]:
            with patch.dict(namespace, {
                'open': lambda *args, **kwargs: io.StringIO('1.25 2.75\n3.1 4.9\n'),
            }):
                self.assertEqual(namespace['read_cities_from_file']('unused'),
                                 [(1.25, 2.75), (3.1, 4.9)])

    def test_tabu_tenure_blocks_exactly_the_next_iterations(self):
        # Reversing positions 0 and 1 worsens the square tour; repeating it undoes the move.
        for tenure in [0, 1, 2]:
            with self.subTest(tenure=tenure), \
                 patch.object(random, 'shuffle', return_value=None), \
                 patch.object(random, 'sample', return_value=[0, 1]), \
                 contextlib.redirect_stdout(io.StringIO()):
                route, cost, history = ts['tabu_search'](
                    SQUARE, max_iterations=tenure + 2, tabu_tenure=tenure, n_neighbors=1)
            self.assertEqual(history[0], 4)
            self.assertGreater(history[1], 4)
            self.assertTrue(all(value == history[1] for value in history[1:tenure + 2]))
            self.assertEqual(history[-1], 4)
            self.assertEqual(cost, 4)
            self.assertEqual(sorted(route), [0, 1, 2, 3])


if __name__ == '__main__':
    unittest.main()
