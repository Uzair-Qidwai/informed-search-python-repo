# Informed Search Strategies in Python

A collection of Python implementations for classic informed search methods.

## Included algorithms

- Greedy Best-First Search
- A* Search
- Bidirectional A* Search
- IDA* (Iterative Deepening A*)
- RBFS (Recursive Best-First Search)
- SMA* (Simplified Memory-Bounded A*)
- Beam Search
- Weighted A* Search

## Repository structure

```text
informed-search-python-repo/
├── README.md
├── LICENSE
├── requirements.txt
├── informed_search.py
└── examples.py
```

## Features

- Clean Python implementations with docstrings and line-by-line comments.
- Shared utilities for path reconstruction and heuristic handling.
- Works with weighted graphs represented as adjacency lists.
- Demo examples for each strategy.
- MIT licensed for public sharing.

## Graph format

Weighted graphs are represented as:

```python
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 2), ('E', 5)],
    'C': [('F', 1)],
    'D': [],
    'E': [('G', 1)],
    'F': [('G', 3)],
    'G': []
}
```

Heuristics are represented as:

```python
heuristic = {
    'A': 5,
    'B': 4,
    'C': 2,
    'D': 4,
    'E': 1,
    'F': 2,
    'G': 0,
}
```

## Quick start

```bash
python examples.py
```

## Notes

- A* uses `f(n) = g(n) + h(n)`.
- Weighted A* uses `f(n) = g(n) + w * h(n)`.
- Beam Search keeps only the best `beam_width` frontier candidates at each layer.
- SMA* here is a practical, compact educational approximation of memory-bounded best-first search suitable for a teaching repo.

## Suggested GitHub topics

- python
- algorithms
- artificial-intelligence
- heuristic-search
- astar
- beam-search
- ida-star
- weighted-astar

## Repo description

Python implementations of major informed search strategies including A*, IDA*, RBFS, SMA*, Beam Search, Greedy Best-First Search, and Weighted A*.
