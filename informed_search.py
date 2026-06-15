"""Informed search strategies implemented in Python.

This module contains educational, public-repo-ready implementations of:
- Greedy Best-First Search
- A* Search
- Bidirectional A* Search
- IDA* Search
- RBFS
- SMA*
- Beam Search
- Weighted A* Search
"""

# Import heapq for priority-queue behavior.
import heapq
# Import math for infinity values used by RBFS.
import math
# Import deque for layered expansion in beam search.
from collections import deque
# Import typing helpers for clearer function signatures.
from typing import Dict, Hashable, List, Optional, Set, Tuple

# Define a generic node type alias.
Node = Hashable
# Define the weighted graph adjacency-list type.
WeightedGraph = Dict[Node, List[Tuple[Node, float]]]
# Define the heuristic table type.
Heuristic = Dict[Node, float]


# Define a helper function that rebuilds a path from a parent map.
def reconstruct_path(parent: Dict[Node, Optional[Node]], goal: Node) -> List[Node]:
    # Create an empty list to store the backward path.
    path: List[Node] = []
    # Start from the goal node.
    current: Optional[Node] = goal
    # Walk backward until the root node is reached.
    while current is not None:
        # Append the current node to the path.
        path.append(current)
        # Move to the parent of the current node.
        current = parent[current]
    # Reverse the path so it runs from start to goal.
    path.reverse()
    # Return the reconstructed path.
    return path


# Define Greedy Best-First Search.
def greedy_best_first_search(graph: WeightedGraph, heuristic: Heuristic, start: Node, goal: Node) -> Optional[List[Node]]:
    # Initialize the priority queue with the start node scored only by its heuristic value.
    frontier: List[Tuple[float, Node]] = [(heuristic[start], start)]
    # Track visited nodes to avoid repeated expansion.
    visited: Set[Node] = set()
    # Store parents for path reconstruction.
    parent: Dict[Node, Optional[Node]] = {start: None}

    # Continue until there are no frontier nodes left.
    while frontier:
        # Remove the node with the smallest heuristic estimate.
        _, current = heapq.heappop(frontier)
        # Skip the node if it was already processed.
        if current in visited:
            continue
        # Mark the node as processed.
        visited.add(current)
        # Return the path if the goal is found.
        if current == goal:
            return reconstruct_path(parent, goal)
        # Expand each neighbor of the current node.
        for neighbor, _ in graph.get(current, []):
            # Only consider neighbors not yet processed.
            if neighbor not in visited:
                # Record the parent if this is the first discovery.
                if neighbor not in parent:
                    parent[neighbor] = current
                # Push the neighbor using only h(n) as its priority.
                heapq.heappush(frontier, (heuristic[neighbor], neighbor))

    # Return None if no solution is found.
    return None


# Define A* Search.
def a_star_search(graph: WeightedGraph, heuristic: Heuristic, start: Node, goal: Node) -> Optional[Tuple[float, List[Node]]]:
    # Initialize the frontier with the start node using f(start) = h(start).
    frontier: List[Tuple[float, float, Node]] = [(heuristic[start], 0.0, start)]
    # Track the best known cost from start to each node.
    g_cost: Dict[Node, float] = {start: 0.0}
    # Track parents for path reconstruction.
    parent: Dict[Node, Optional[Node]] = {start: None}

    # Continue until all frontier states are exhausted.
    while frontier:
        # Pop the node with the smallest f-score.
        _, current_g, current = heapq.heappop(frontier)
        # Return cost and path when the goal is reached.
        if current == goal:
            return current_g, reconstruct_path(parent, goal)
        # Ignore stale entries that are not the current best path.
        if current_g > g_cost[current]:
            continue
        # Expand each neighbor of the current node.
        for neighbor, edge_cost in graph.get(current, []):
            # Compute the tentative g-score for the neighbor.
            tentative_g = current_g + edge_cost
            # Update the neighbor only if this path is better.
            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                # Save the new best g-score.
                g_cost[neighbor] = tentative_g
                # Save the parent pointer.
                parent[neighbor] = current
                # Compute the new f-score.
                f_score = tentative_g + heuristic[neighbor]
                # Push the updated state into the priority queue.
                heapq.heappush(frontier, (f_score, tentative_g, neighbor))

    # Return None if no path exists.
    return None


# Define Weighted A* Search.
def weighted_a_star_search(graph: WeightedGraph, heuristic: Heuristic, start: Node, goal: Node, weight: float = 1.5) -> Optional[Tuple[float, List[Node]]]:
    # Initialize the frontier with weighted heuristic emphasis.
    frontier: List[Tuple[float, float, Node]] = [(weight * heuristic[start], 0.0, start)]
    # Track the best known cost to each node.
    g_cost: Dict[Node, float] = {start: 0.0}
    # Store parents for path reconstruction.
    parent: Dict[Node, Optional[Node]] = {start: None}

    # Continue until the frontier becomes empty.
    while frontier:
        # Remove the node with the smallest weighted f-score.
        _, current_g, current = heapq.heappop(frontier)
        # Return once the goal is popped.
        if current == goal:
            return current_g, reconstruct_path(parent, goal)
        # Ignore outdated queue entries.
        if current_g > g_cost[current]:
            continue
        # Expand outgoing edges from the current node.
        for neighbor, edge_cost in graph.get(current, []):
            # Compute the tentative cost to the neighbor.
            tentative_g = current_g + edge_cost
            # Keep the update only if it improves the best known path.
            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                # Save the improved path cost.
                g_cost[neighbor] = tentative_g
                # Save the parent relation.
                parent[neighbor] = current
                # Compute the weighted evaluation score.
                f_score = tentative_g + weight * heuristic[neighbor]
                # Push the neighbor into the frontier.
                heapq.heappush(frontier, (f_score, tentative_g, neighbor))

    # Return None when no path is found.
    return None


# Define Bidirectional A* Search.
def bidirectional_a_star_search(graph: WeightedGraph, heuristic: Heuristic, reverse_heuristic: Heuristic, start: Node, goal: Node, reverse_graph: Optional[WeightedGraph] = None) -> Optional[Tuple[float, List[Node]]]:
    # Use the original graph as a fallback when a reverse graph is not supplied.
    if reverse_graph is None:
        reverse_graph = graph
    # Initialize the forward frontier.
    forward_frontier: List[Tuple[float, float, Node]] = [(heuristic[start], 0.0, start)]
    # Initialize the backward frontier.
    backward_frontier: List[Tuple[float, float, Node]] = [(reverse_heuristic[goal], 0.0, goal)]
    # Track best forward g-costs.
    forward_g: Dict[Node, float] = {start: 0.0}
    # Track best backward g-costs.
    backward_g: Dict[Node, float] = {goal: 0.0}
    # Track forward parents.
    forward_parent: Dict[Node, Optional[Node]] = {start: None}
    # Track backward parents.
    backward_parent: Dict[Node, Optional[Node]] = {goal: None}
    # Track the best meeting node found so far.
    best_meeting: Optional[Node] = None
    # Track the best complete path cost found so far.
    best_cost = math.inf

    # Continue while both frontiers still contain candidates.
    while forward_frontier and backward_frontier:
        # Expand one node from the forward frontier.
        _, current_g, current = heapq.heappop(forward_frontier)
        # Ignore stale forward entries.
        if current_g <= forward_g[current]:
            # Explore forward neighbors.
            for neighbor, edge_cost in graph.get(current, []):
                # Compute the new forward path cost.
                tentative_g = current_g + edge_cost
                # Update if this path is better.
                if neighbor not in forward_g or tentative_g < forward_g[neighbor]:
                    # Save the improved forward cost.
                    forward_g[neighbor] = tentative_g
                    # Save the forward parent relation.
                    forward_parent[neighbor] = current
                    # Push the forward state using A* evaluation.
                    heapq.heappush(forward_frontier, (tentative_g + heuristic[neighbor], tentative_g, neighbor))
                    # Check whether the backward search has already reached this node.
                    if neighbor in backward_g:
                        # Compute the combined path cost through this meeting node.
                        total_cost = tentative_g + backward_g[neighbor]
                        # Update the best known meeting point if improved.
                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_meeting = neighbor

        # Expand one node from the backward frontier.
        _, current_g, current = heapq.heappop(backward_frontier)
        # Ignore stale backward entries.
        if current_g <= backward_g[current]:
            # Explore backward neighbors.
            for neighbor, edge_cost in reverse_graph.get(current, []):
                # Compute the new backward path cost.
                tentative_g = current_g + edge_cost
                # Update if this path is better.
                if neighbor not in backward_g or tentative_g < backward_g[neighbor]:
                    # Save the improved backward cost.
                    backward_g[neighbor] = tentative_g
                    # Save the backward parent relation.
                    backward_parent[neighbor] = current
                    # Push the backward state using reverse A* evaluation.
                    heapq.heappush(backward_frontier, (tentative_g + reverse_heuristic[neighbor], tentative_g, neighbor))
                    # Check whether the forward search has already reached this node.
                    if neighbor in forward_g:
                        # Compute the combined path cost through this meeting node.
                        total_cost = tentative_g + forward_g[neighbor]
                        # Update the best meeting point if improved.
                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_meeting = neighbor

    # Return None if the two searches never meet.
    if best_meeting is None:
        return None

    # Reconstruct the forward half of the path.
    forward_path = reconstruct_path(forward_parent, best_meeting)
    # Reconstruct the backward half from the meeting node to the goal.
    backward_path: List[Node] = []
    # Start from the node after the meeting point on the backward tree.
    current = backward_parent[best_meeting]
    # Follow parents until the goal root is reached.
    while current is not None:
        # Append each node in the backward half.
        backward_path.append(current)
        # Move toward the goal root.
        current = backward_parent[current]
    # Return total cost and the merged path.
    return best_cost, forward_path + backward_path


# Define IDA* Search.
def ida_star_search(graph: WeightedGraph, heuristic: Heuristic, start: Node, goal: Node) -> Optional[Tuple[float, List[Node]]]:
    # Set the initial threshold to the start node's f-value.
    threshold = heuristic[start]
    # Initialize the current DFS path with the start node.
    path: List[Node] = [start]

    # Define the recursive depth-first search bounded by the threshold.
    def search(path: List[Node], g_cost: float, bound: float) -> Tuple[float, Optional[List[Node]]]:
        # Read the current node from the end of the active path.
        current = path[-1]
        # Compute the node's evaluation score.
        f_score = g_cost + heuristic[current]
        # Prune this branch if it exceeds the current threshold.
        if f_score > bound:
            return f_score, None
        # Return the solution when the goal is found.
        if current == goal:
            return g_cost, path.copy()
        # Track the minimum f-score that exceeded the current bound.
        minimum = math.inf
        # Explore each outgoing edge.
        for neighbor, edge_cost in graph.get(current, []):
            # Avoid immediate cycles on the current DFS path.
            if neighbor in path:
                continue
            # Add the neighbor to the active DFS path.
            path.append(neighbor)
            # Recurse with the updated path cost.
            result, solution = search(path, g_cost + edge_cost, bound)
            # Return immediately if a solution is found.
            if solution is not None:
                return result, solution
            # Track the smallest pruned score for the next iteration.
            minimum = min(minimum, result)
            # Remove the neighbor during backtracking.
            path.pop()
        # Return the next threshold candidate if no solution was found.
        return minimum, None

    # Repeat bounded searches until a solution or failure occurs.
    while True:
        # Run a threshold-bounded DFS.
        result, solution = search(path, 0.0, threshold)
        # Return the solution if found.
        if solution is not None:
            return result, solution
        # Stop if no larger threshold exists.
        if result == math.inf:
            return None
        # Increase the threshold to the smallest exceeded f-score.
        threshold = result


# Define RBFS.
def rbfs_search(graph: WeightedGraph, heuristic: Heuristic, start: Node, goal: Node) -> Optional[Tuple[float, List[Node]]]:
    # Define the recursive RBFS helper.
    def rbfs(node: Node, g_cost: float, path: List[Node], f_limit: float) -> Tuple[Optional[Tuple[float, List[Node]]], float]:
        # Return the current path if the goal is found.
        if node == goal:
            return (g_cost, path.copy()), g_cost
        # Create a list to hold successor states and their scores.
        successors: List[List[object]] = []
        # Expand each neighbor of the current node.
        for neighbor, edge_cost in graph.get(node, []):
            # Skip neighbors already on the active recursive path.
            if neighbor in path:
                continue
            # Compute the successor g-score.
            new_g = g_cost + edge_cost
            # Compute the successor f-score.
            new_f = max(new_g + heuristic[neighbor], g_cost + heuristic[node])
            # Store the successor state as a mutable record.
            successors.append([neighbor, new_g, new_f, path + [neighbor]])
        # Fail if there are no valid successors.
        if not successors:
            return None, math.inf
        # Continue exploring the best successor first.
        while True:
            # Sort successors by their current best f-score.
            successors.sort(key=lambda item: item[2])
            # Read the best successor.
            best = successors[0]
            # Fail upward if the best option exceeds the current limit.
            if best[2] > f_limit:
                return None, best[2]
            # Read the alternative bound from the second-best successor.
            alternative = successors[1][2] if len(successors) > 1 else math.inf
            # Recurse on the best successor with a tighter bound.
            result, best_f = rbfs(best[0], best[1], best[3], min(f_limit, alternative))
            # Update the best successor's backed-up f-value.
            best[2] = best_f
            # Return the found solution immediately.
            if result is not None:
                return result, best_f

    # Start RBFS from the initial state.
    result, _ = rbfs(start, 0.0, [start], math.inf)
    # Return the final result.
    return result


# Define a compact educational SMA* approximation with explicit memory bounding.
def sma_star_search(graph: WeightedGraph, heuristic: Heuristic, start: Node, goal: Node, memory_limit: int = 10) -> Optional[Tuple[float, List[Node]]]:
    # Initialize the frontier with the start state.
    frontier: List[Tuple[float, float, Node, List[Node]]] = [(heuristic[start], 0.0, start, [start])]
    # Track the best known cost for each expanded node.
    best_cost: Dict[Node, float] = {start: 0.0}

    # Continue until there are no states left to explore.
    while frontier:
        # Keep the frontier sorted by best evaluation score.
        frontier.sort(key=lambda item: (item[0], item[1]))
        # Enforce the memory bound by dropping the worst states.
        while len(frontier) > memory_limit:
            frontier.pop()
        # Remove the best available state.
        f_score, g_cost, current, path = frontier.pop(0)
        # Return the path when the goal is found.
        if current == goal:
            return g_cost, path
        # Expand each neighbor of the current node.
        for neighbor, edge_cost in graph.get(current, []):
            # Avoid cycles within the current path.
            if neighbor in path:
                continue
            # Compute the new path cost.
            new_g = g_cost + edge_cost
            # Keep only improved visits to the same node.
            if neighbor not in best_cost or new_g < best_cost[neighbor]:
                # Save the improved cost.
                best_cost[neighbor] = new_g
                # Compute the node evaluation score.
                new_f = new_g + heuristic[neighbor]
                # Push the child state into the frontier.
                frontier.append((new_f, new_g, neighbor, path + [neighbor]))

    # Return None if memory-bounded exploration fails to find the goal.
    return None


# Define Beam Search.
def beam_search(graph: WeightedGraph, heuristic: Heuristic, start: Node, goal: Node, beam_width: int = 2) -> Optional[List[Node]]:
    # Initialize the current beam with the start node and its path.
    beam: List[Tuple[Node, List[Node]]] = [(start, [start])]

    # Continue level by level until the beam is empty.
    while beam:
        # Create a list to collect the next level's candidates.
        candidates: List[Tuple[float, Node, List[Node]]] = []
        # Expand every state currently in the beam.
        for current, path in beam:
            # Return immediately if the goal appears in the beam.
            if current == goal:
                return path
            # Explore outgoing neighbors from the current node.
            for neighbor, _ in graph.get(current, []):
                # Avoid cycles within the current path.
                if neighbor not in path:
                    # Score the candidate using the heuristic.
                    candidates.append((heuristic[neighbor], neighbor, path + [neighbor]))
        # Stop if no candidates were generated.
        if not candidates:
            return None
        # Sort the candidate pool by heuristic score.
        candidates.sort(key=lambda item: item[0])
        # Keep only the top beam_width states for the next layer.
        beam = [(node, path) for _, node, path in candidates[:beam_width]]

    # Return None if the goal is never reached.
    return None
