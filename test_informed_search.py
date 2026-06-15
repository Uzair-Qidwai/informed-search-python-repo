"""Unit tests for the informed search repository."""

# Import the unittest framework from Python's standard library.
import unittest

# Import the search implementations under test.
from informed_search import (
    a_star_search,
    beam_search,
    bidirectional_a_star_search,
    greedy_best_first_search,
    ida_star_search,
    rbfs_search,
    reconstruct_path,
    sma_star_search,
    weighted_a_star_search,
)


# Define the shared test case for all informed search strategies.
class TestInformedSearchAlgorithms(unittest.TestCase):
    # Build reusable graph fixtures before each test method runs.
    def setUp(self) -> None:
        # Define a small weighted graph used by most tests.
        self.graph = {
            'A': [('B', 1), ('C', 4)],
            'B': [('D', 2), ('E', 5)],
            'C': [('F', 1)],
            'D': [('G', 7)],
            'E': [('G', 1)],
            'F': [('G', 3)],
            'G': [],
            'Z': [],
        }
        # Define a reverse graph for bidirectional A* tests.
        self.reverse_graph = {
            'A': [],
            'B': [('A', 1)],
            'C': [('A', 4)],
            'D': [('B', 2)],
            'E': [('B', 5)],
            'F': [('C', 1)],
            'G': [('D', 7), ('E', 1), ('F', 3)],
            'Z': [],
        }
        # Define a forward heuristic estimate to goal node G.
        self.heuristic = {
            'A': 5,
            'B': 4,
            'C': 2,
            'D': 6,
            'E': 1,
            'F': 2,
            'G': 0,
            'Z': 99,
        }
        # Define a reverse heuristic estimate to start node A.
        self.reverse_heuristic = {
            'A': 0,
            'B': 1,
            'C': 4,
            'D': 3,
            'E': 6,
            'F': 5,
            'G': 6,
            'Z': 99,
        }
        # Store the expected optimal path for algorithms that should find it.
        self.optimal_path = ['A', 'B', 'E', 'G']
        # Store the expected optimal cost for that path.
        self.optimal_cost = 7

    # Verify that reconstruct_path rebuilds a forward path correctly.
    def test_reconstruct_path(self) -> None:
        # Build a small parent map.
        parent = {'A': None, 'B': 'A', 'E': 'B', 'G': 'E'}
        # Assert that the reconstructed path is correct.
        self.assertEqual(reconstruct_path(parent, 'G'), ['A', 'B', 'E', 'G'])

    # Verify that Greedy Best-First Search returns a valid start-to-goal path.
    def test_greedy_best_first_search_returns_valid_path(self) -> None:
        # Run Greedy Best-First Search on the shared graph.
        path = greedy_best_first_search(self.graph, self.heuristic, 'A', 'G')
        # Assert that a path was found.
        self.assertIsNotNone(path)
        # Assert that the path starts at the requested start node.
        self.assertEqual(path[0], 'A')
        # Assert that the path ends at the requested goal node.
        self.assertEqual(path[-1], 'G')

    # Verify that A* returns the expected optimal path and cost.
    def test_a_star_search_returns_optimal_result(self) -> None:
        # Run A* search on the shared graph.
        result = a_star_search(self.graph, self.heuristic, 'A', 'G')
        # Assert that a solution was found.
        self.assertIsNotNone(result)
        # Unpack the returned cost and path.
        cost, path = result
        # Assert that the total cost is optimal.
        self.assertEqual(cost, self.optimal_cost)
        # Assert that the path is the expected optimal path.
        self.assertEqual(path, self.optimal_path)

    # Verify that Weighted A* returns a valid path and numeric cost.
    def test_weighted_a_star_search_returns_valid_result(self) -> None:
        # Run Weighted A* with a modest heuristic weight.
        result = weighted_a_star_search(self.graph, self.heuristic, 'A', 'G', weight=1.5)
        # Assert that a solution was found.
        self.assertIsNotNone(result)
        # Unpack the result.
        cost, path = result
        # Assert that the path begins at the start node.
        self.assertEqual(path[0], 'A')
        # Assert that the path ends at the goal node.
        self.assertEqual(path[-1], 'G')
        # Assert that the reported cost is positive.
        self.assertGreater(cost, 0)

    # Verify that Bidirectional A* returns the expected optimal solution.
    def test_bidirectional_a_star_search_returns_optimal_result(self) -> None:
        # Run Bidirectional A* on the forward and reverse graphs.
        result = bidirectional_a_star_search(
            self.graph,
            self.heuristic,
            self.reverse_heuristic,
            'A',
            'G',
            self.reverse_graph,
        )
        # Assert that a solution was found.
        self.assertIsNotNone(result)
        # Unpack the result.
        cost, path = result
        # Assert that the total cost is optimal.
        self.assertEqual(cost, self.optimal_cost)
        # Assert that the path matches the known best path.
        self.assertEqual(path, self.optimal_path)

    # Verify that IDA* returns the expected optimal path and cost.
    def test_ida_star_search_returns_optimal_result(self) -> None:
        # Run IDA* search on the shared graph.
        result = ida_star_search(self.graph, self.heuristic, 'A', 'G')
        # Assert that a solution was found.
        self.assertIsNotNone(result)
        # Unpack the result.
        cost, path = result
        # Assert that the total cost is optimal.
        self.assertEqual(cost, self.optimal_cost)
        # Assert that the path matches the known best path.
        self.assertEqual(path, self.optimal_path)

    # Verify that RBFS returns the expected optimal path and cost.
    def test_rbfs_search_returns_optimal_result(self) -> None:
        # Run RBFS on the shared graph.
        result = rbfs_search(self.graph, self.heuristic, 'A', 'G')
        # Assert that a solution was found.
        self.assertIsNotNone(result)
        # Unpack the result.
        cost, path = result
        # Assert that the total cost is optimal.
        self.assertEqual(cost, self.optimal_cost)
        # Assert that the path matches the known best path.
        self.assertEqual(path, self.optimal_path)

    # Verify that SMA* returns a valid path under a memory bound.
    def test_sma_star_search_returns_valid_result(self) -> None:
        # Run SMA* with a small but sufficient memory limit.
        result = sma_star_search(self.graph, self.heuristic, 'A', 'G', memory_limit=4)
        # Assert that a solution was found.
        self.assertIsNotNone(result)
        # Unpack the result.
        cost, path = result
        # Assert that the path begins at the start node.
        self.assertEqual(path[0], 'A')
        # Assert that the path ends at the goal node.
        self.assertEqual(path[-1], 'G')
        # Assert that the cost is positive.
        self.assertGreater(cost, 0)

    # Verify that Beam Search returns a valid path when the beam is wide enough.
    def test_beam_search_returns_valid_path(self) -> None:
        # Run Beam Search with a beam width of two.
        path = beam_search(self.graph, self.heuristic, 'A', 'G', beam_width=2)
        # Assert that a path was found.
        self.assertIsNotNone(path)
        # Assert that the path begins at the start node.
        self.assertEqual(path[0], 'A')
        # Assert that the path ends at the goal node.
        self.assertEqual(path[-1], 'G')

    # Verify that A* reports failure when the goal is unreachable.
    def test_a_star_search_returns_none_for_unreachable_goal(self) -> None:
        # Ask A* to search for an isolated node.
        result = a_star_search(self.graph, self.heuristic, 'A', 'Z')
        # Assert that the algorithm reports failure.
        self.assertIsNone(result)

    # Verify that Greedy Best-First Search reports failure when the goal is unreachable.
    def test_greedy_best_first_search_returns_none_for_unreachable_goal(self) -> None:
        # Ask Greedy Best-First Search to search for an isolated node.
        result = greedy_best_first_search(self.graph, self.heuristic, 'A', 'Z')
        # Assert that the algorithm reports failure.
        self.assertIsNone(result)

    # Verify that Beam Search reports failure when the goal is unreachable.
    def test_beam_search_returns_none_for_unreachable_goal(self) -> None:
        # Ask Beam Search to search for an isolated node.
        result = beam_search(self.graph, self.heuristic, 'A', 'Z', beam_width=2)
        # Assert that the algorithm reports failure.
        self.assertIsNone(result)


# Run the test suite when the file is executed directly.
if __name__ == '__main__':
    # Start unittest's command-line test runner.
    unittest.main()
