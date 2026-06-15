"""Runnable examples for informed search strategies."""

# Import all search implementations from the main module.
from informed_search import (
    a_star_search,
    beam_search,
    bidirectional_a_star_search,
    greedy_best_first_search,
    ida_star_search,
    rbfs_search,
    sma_star_search,
    weighted_a_star_search,
)

# Define a small weighted graph for demonstrations.
GRAPH = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 2), ('E', 5)],
    'C': [('F', 1)],
    'D': [('G', 7)],
    'E': [('G', 1)],
    'F': [('G', 3)],
    'G': [],
}

# Define a reverse graph for bidirectional A*.
REVERSE_GRAPH = {
    'A': [],
    'B': [('A', 1)],
    'C': [('A', 4)],
    'D': [('B', 2)],
    'E': [('B', 5)],
    'F': [('C', 1)],
    'G': [('D', 7), ('E', 1), ('F', 3)],
}

# Define a heuristic estimate to the goal for the forward search.
HEURISTIC = {
    'A': 5,
    'B': 4,
    'C': 2,
    'D': 6,
    'E': 1,
    'F': 2,
    'G': 0,
}

# Define a reverse heuristic estimate to the start for bidirectional A*.
REVERSE_HEURISTIC = {
    'A': 0,
    'B': 1,
    'C': 4,
    'D': 3,
    'E': 6,
    'F': 5,
    'G': 6,
}

# Run examples only when this file is executed directly.
if __name__ == '__main__':
    # Print the result of Greedy Best-First Search.
    print('Greedy Best-First:', greedy_best_first_search(GRAPH, HEURISTIC, 'A', 'G'))
    # Print the result of A* Search.
    print('A*:', a_star_search(GRAPH, HEURISTIC, 'A', 'G'))
    # Print the result of Bidirectional A* Search.
    print('Bidirectional A*:', bidirectional_a_star_search(GRAPH, HEURISTIC, REVERSE_HEURISTIC, 'A', 'G', REVERSE_GRAPH))
    # Print the result of IDA* Search.
    print('IDA*:', ida_star_search(GRAPH, HEURISTIC, 'A', 'G'))
    # Print the result of RBFS.
    print('RBFS:', rbfs_search(GRAPH, HEURISTIC, 'A', 'G'))
    # Print the result of SMA*.
    print('SMA*:', sma_star_search(GRAPH, HEURISTIC, 'A', 'G', memory_limit=4))
    # Print the result of Beam Search.
    print('Beam Search:', beam_search(GRAPH, HEURISTIC, 'A', 'G', beam_width=2))
    # Print the result of Weighted A* Search.
    print('Weighted A*:', weighted_a_star_search(GRAPH, HEURISTIC, 'A', 'G', weight=1.5))
