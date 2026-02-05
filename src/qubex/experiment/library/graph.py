from __future__ import annotations

import time
from collections import deque

import networkx as nx
from networkx.algorithms.coloring import greedy_color

Edge = tuple


def _simple_undirected(
    G: nx.Graph,
) -> nx.Graph:
    """
    Convert a graph to a simple undirected graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph, possibly directed, with self-loops or multiedges.

    Returns
    -------
    nx.Graph
        Simple undirected copy of `G` without self-loops.
    """
    H = G.to_undirected() if G.is_directed() else G.copy()
    H = nx.Graph(H)  # drop multiedges
    H.remove_edges_from(nx.selfloop_edges(H))
    return H


def _graph_power_by_distance(
    G: nx.Graph,
    p: int,
) -> nx.Graph:
    """
    Return the p-th power of a graph based on shortest-path distance.

    In the p-th power of a graph, two nodes are adjacent if their shortest
    path distance in G is at most p.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    p : int
        Distance threshold.

    Returns
    -------
    nx.Graph
        Graph where edges connect nodes of G with distance <= p.
    """
    if p < 1:
        raise ValueError("p must be >= 1")
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for s in G.nodes():
        # Bounded BFS up to depth p
        q = deque([(s, 0)])
        seen = {s}
        while q:
            u, d = q.popleft()
            if 0 < d <= p:
                H.add_edge(s, u)
            if d == p:
                continue
            for w in G.neighbors(u):
                if w not in seen:
                    seen.add(w)
                    q.append((w, d + 1))
    return H


def verify_strong_matching(
    G: nx.Graph,
    colored_edges: dict[int, list[Edge]],
) -> bool:
    """
    Verify that each edge set is a strong matching.

    Parameters
    ----------
    G : nx.Graph
        Original graph.
    colored_edges : dict[int, list[Edge]]
        Dictionary mapping color indices to lists of edges.

    Returns
    -------
    bool
        True if all sets are valid strong matchings, False otherwise.
    """
    H = _simple_undirected(G)
    L = nx.line_graph(H)

    # Precompute distance-1 and distance-2 neighbors in L(G)
    d1 = {n: set(L.neighbors(n)) for n in L.nodes()}
    d2 = {n: set() for n in L.nodes()}
    for n in L.nodes():
        for a in d1[n]:
            d2[n].update(L.neighbors(a))
        d2[n].discard(n)

    for color, edges in colored_edges.items():
        if not verify_strong_matching(G, {color: edges}):
            return False
    return True


def strong_edge_coloring(
    G: nx.Graph,
) -> dict[int, list[Edge]]:
    """
    Heuristic strong edge coloring for general graphs.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    dict[int, list[Edge]]
        Dict of colored edges with no two edges within distance 2 in L(G).
    """
    H = _simple_undirected(G)
    if H.number_of_edges() == 0:
        return {}
    L = nx.line_graph(H)
    L2 = _graph_power_by_distance(L, p=2)

    strategies = [
        "saturation_largest_first",
        "largest_first",
        "smallest_last",
        "independent_set",
        "connected_sequential_bfs",
        "connected_sequential_dfs",
    ]

    best_colors = {}
    best_k = float("inf")

    for strat in strategies:
        edge_color_map = greedy_color(L2, strategy=strat)
        k = (max(edge_color_map.values()) + 1) if edge_color_map else 0
        if k < best_k:
            best_k, best_colors = k, edge_color_map

    n_colors = int(best_k)

    colored_edges = {color: [] for color in range(n_colors)}
    for e, c in best_colors.items():
        # L(G) nodes are original edges (tuples)
        colored_edges[c].append(e)
    return colored_edges


def tree_center(G):
    u = max(
        nx.single_source_shortest_path_length(G, list(G.nodes)[0]).items(),
        key=lambda x: x[1],
    )[0]
    lengths = nx.single_source_shortest_path_length(G, u)
    v = max(lengths.items(), key=lambda x: x[1])[0]
    path = nx.shortest_path(G, u, v)
    L = len(path)
    if L % 2 == 1:
        return [path[L // 2]]
    else:
        return [path[L // 2 - 1], path[L // 2]]


def get_max_undirected_weight(
    G: nx.Graph,
    edge: tuple[str, str],
    property: str = "fidelity",
) -> float:
    u, v = edge
    if G.is_multigraph():
        vals = []
        if G.has_edge(u, v):
            vals.extend([data.get(property, 0.0) for _, data in G[u][v].items()])
        if G.has_edge(v, u):
            vals.extend([data.get(property, 0.0) for _, data in G[v][u].items()])
        return float(max(vals)) if vals else 0.0
    else:
        vals = []
        if G.has_edge(u, v):
            vals.append(G[u][v].get(property, 0.0))
        if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)) and G.has_edge(v, u):
            vals.append(G[v][u].get(property, 0.0))
        return float(max(vals)) if vals else 0.0


def find_longest_1d_chain(
    G: nx.Graph,
    *,
    weight_attr: str = "fidelity",
    time_limit: float | None = 5.0,
    eps: float = 1e-12,
) -> tuple[list, list, float]:
    """
    Find the longest simple path in a 1D chain graph, maximizing fidelity

    The search considers connectivity as undirected even when ``G`` is directed,
    and uses an aggregated per-step weight across directions and parallel edges
    (maximum for ``'fidelity'``, minimum for ``'duration'``).

    Parameters
    ----------
    G : networkx.Graph
        Input graph. Can be ``nx.Graph``, ``nx.DiGraph``, ``nx.MultiGraph``, or
        ``nx.MultiDiGraph``.
    weight_attr : str, default 'fidelity'
        Edge attribute used for tie-breaking among same-length paths. Supported:
        - ``'fidelity'``: aggregate per-step as the maximum across directions/parallel edges;
          tie-break cost is the sum of ``-log(fidelity)`` (infidelity in log-sum form).
        - ``'duration'``: aggregate per-step as the minimum across directions/parallel edges;
          tie-break cost is the sum of ``duration``.
    time_limit : float or None, default 5.0
        Time budget (seconds) for the search. ``None`` disables time-based cutoff.
    eps : float, default 1e-12
        Lower bound to clip fidelity when using log-based scoring.

    Returns
    -------
    path_nodes : list
        Node sequence of the best path found.
    path_edges : list of tuple
        Oriented edges in path order ``(u, v)`` corresponding to ``path_nodes``.
    score : float
        Tie-break cost (lower is better): for fidelity it's ``sum(-log f)``; for duration it's
        ``sum(duration)``.

    Notes
    -----
    - Edge direction is ignored when building the internal undirected view; the
      search is over simple paths (no repeated nodes).
    - The problem is NP-hard on general graphs. This implementation uses greedy
      seeding and reachability-based pruning to perform well on typical instances.
    - For fidelity, the computation uses log-sum cost to avoid
      underflow.
    """

    # Basic validation and setup
    assert weight_attr in {
        "fidelity",
        "duration",
    }, "weight_attr must be 'fidelity' or 'duration'"
    # Treat connectivity as undirected even if G is DiGraph/MultiDiGraph.
    is_digraph = isinstance(G, (nx.DiGraph, nx.MultiDiGraph))

    # Build an undirected simple graph UG.
    # Collapse all edges between u and v (any direction, any multiplicity) to a single
    # undirected edge with an aggregate per-step weight.
    UG = nx.Graph()
    UG.add_nodes_from(G.nodes())

    # Use weight_attr directly as the UG edge attribute key for the aggregated value

    def aggregate_value(u, v) -> float | None:
        """
        Aggregate weight across directions/parallel edges for the pair (u, v).
        - fidelity: maximum value (missing attributes treated as 0.0 if edge exists)
        - duration: minimum value (edges without the attribute are ignored)
        Returns None if no edge exists in either direction or no usable value found (duration).
        """
        vals: list[float] = []
        if G.is_multigraph():
            if G.has_edge(u, v):
                if weight_attr == "fidelity":
                    vals.extend(
                        [data.get(weight_attr, 0.0) for _, data in G[u][v].items()]
                    )
                else:
                    vals.extend(
                        [
                            data[weight_attr]
                            for _, data in G[u][v].items()
                            if weight_attr in data
                        ]
                    )
            if G.has_edge(v, u):
                if weight_attr == "fidelity":
                    vals.extend(
                        [data.get(weight_attr, 0.0) for _, data in G[v][u].items()]
                    )
                else:
                    vals.extend(
                        [
                            data[weight_attr]
                            for _, data in G[v][u].items()
                            if weight_attr in data
                        ]
                    )
        else:
            if G.has_edge(u, v):
                if weight_attr == "fidelity":
                    vals.append(G[u][v].get(weight_attr, 0.0))
                elif weight_attr in G[u][v]:
                    vals.append(G[u][v][weight_attr])
            if is_digraph and G.has_edge(v, u):
                if weight_attr == "fidelity":
                    vals.append(G[v][u].get(weight_attr, 0.0))
                elif weight_attr in G[v][u]:
                    vals.append(G[v][u][weight_attr])
        if not vals:
            # For fidelity, if there's connectivity but no attribute at all, treat as 0.0; else None.
            if G.has_edge(u, v) or (is_digraph and G.has_edge(v, u)):
                return 0.0 if weight_attr == "fidelity" else None
            return None
        if weight_attr == "duration":
            return float(min(vals))
        return float(max(vals))

    # Add undirected edges for all connected pairs (in any direction), storing the
    # aggregated value under weight_attr.
    pairs: set[tuple] = set()
    for a, b in G.edges():
        if a == b:
            continue
        u, v = (a, b) if a <= b else (b, a)
        pairs.add((u, v))
    for u, v in pairs:
        w = aggregate_value(u, v)
        if w is None:
            # skip edges without usable value when using duration
            continue
        prev = UG[u][v].get(weight_attr) if UG.has_edge(u, v) else None
        # For fidelity, keep the larger aggregated weight; for duration, keep the smaller.
        if prev is None or (
            (weight_attr == "duration" and w < prev)
            or (weight_attr != "duration" and w > prev)
        ):
            UG.add_edge(u, v, **{weight_attr: w})

    # Pre-sort neighbors by weight (heuristic ordering) to help DFS find a
    # strong incumbent quickly (desc for fidelity, asc for duration).
    if weight_attr == "duration":
        neighbors_sorted = {
            u: sorted(
                UG.neighbors(u), key=lambda v: UG[u][v].get(weight_attr, float("inf"))
            )
            for u in UG.nodes()
        }
    else:
        neighbors_sorted = {
            u: sorted(
                UG.neighbors(u),
                key=lambda v: UG[u][v].get(weight_attr, 0.0),
                reverse=True,
            )
            for u in UG.nodes()
        }

    # Cost helpers (lower is better):
    # - fidelity: sum(-log f)
    # - duration: sum(duration)
    def extend_cost(cost: float, f: float) -> float:
        if weight_attr == "duration":
            return cost + f
        # Clip to avoid log(0) and keep costs finite/stable.
        f = max(f, eps)
        import math

        return cost + (-math.log(f))

    def score_better(a_len: int, a_cost: float, b_len: int, b_cost: float) -> bool:
        if a_len != b_len:
            return a_len > b_len
        return a_cost < b_cost

    # Reachability-based pruning: can we tie or exceed current best length from u?

    def can_tie_or_exceed(u, visited: set, cur_len: int, best_len: int) -> bool:
        # To tie the incumbent, we need (best_len - cur_len) more edges from u,
        # which requires at least that many new nodes; counting u, that's +1.
        need_nodes = (best_len - cur_len) + 1
        if best_len <= cur_len:
            return True
        q = deque([u])
        seen = {u}
        count = 1
        while q:
            x = q.popleft()
            for y in neighbors_sorted[x]:
                if y in seen or y in visited:
                    continue
                seen.add(y)
                q.append(y)
                count += 1
                if count >= need_nodes:
                    return True
        return False

    # Greedy seed to obtain a strong initial lower bound quickly.
    def greedy_path_from(start: str) -> tuple[list, float]:
        visited = {start}
        path = [start]
        cost = 0.0
        u = start
        while True:
            nbrs = [v for v in neighbors_sorted[u] if v not in visited]
            if not nbrs:
                break
            # Prefer higher fidelity or lower duration; break ties by neighbor degree (more options ahead).
            if weight_attr == "duration":
                v = min(
                    nbrs,
                    key=lambda x: (
                        UG[u][x].get(weight_attr, float("inf")),
                        -len(neighbors_sorted[x]),
                    ),
                )
            else:
                v = max(
                    nbrs,
                    key=lambda x: (
                        UG[u][x].get(weight_attr, 0.0),
                        len(neighbors_sorted[x]),
                    ),
                )
            f = UG[u][v].get(weight_attr, 0.0)
            path.append(v)
            visited.add(v)
            cost = extend_cost(cost, f)
            u = v
        return path, cost

    def node_weight(u):
        # Weighted degree used to rank promising starting nodes.
        vals = [UG[u][v].get(weight_attr) for v in neighbors_sorted[u]]
        vals = [x for x in vals if x is not None]
        if not vals:
            return 0.0
        total = sum(vals)
        # For duration smaller is better; flip sign so higher is better for ranking.
        return total if weight_attr != "duration" else -total

    candidate_starts = (
        sorted(UG.nodes(), key=node_weight, reverse=True)[:3] or list(UG.nodes())[:1]
    )  # Always have at least one start.

    best_len = -1
    best_score = float("inf")
    best_path: list | None = None

    # Establish incumbent via greedy runs (improves pruning effectiveness).
    for s in candidate_starts:
        p, s_cost = greedy_path_from(s)
        l = len(p) - 1
        if score_better(l, s_cost, best_len, best_score):
            best_len, best_score, best_path = l, s_cost, p

    def dfs(u, visited: set, cur_cost: float, path: list):
        nonlocal best_len, best_score, best_path
        # Honor optional time budget.
        if time_limit is not None and (time.perf_counter() - start_time) > time_limit:
            return
        cur_len = len(path) - 1
        if best_path is None or score_better(cur_len, cur_cost, best_len, best_score):
            best_len, best_score, best_path = cur_len, cur_cost, path.copy()
        if not can_tie_or_exceed(u, visited, cur_len, best_len):
            return
        for v in neighbors_sorted[u]:
            if v in visited:
                continue
            f = UG[u][v].get(weight_attr, 0.0)
            visited.add(v)
            path.append(v)
            dfs(v, visited, extend_cost(cur_cost, f), path)
            path.pop()
            visited.remove(v)

    start_time = time.perf_counter()
    # Explore promising starts first, then remaining nodes if time allows.
    for s in candidate_starts + [x for x in UG.nodes() if x not in candidate_starts]:
        if time_limit is not None and (time.perf_counter() - start_time) > time_limit:
            break
        visited = {s}
        dfs(s, visited, 0.0, [s])

    if not best_path:
        return [], [], 0.0

    # Oriented edge sequence along the undirected path (for convenience).
    edges_in_path_order = list(zip(best_path[:-1], best_path[1:]))
    return best_path, edges_in_path_order, best_score


__all__ = [
    "strong_edge_coloring",
    "find_longest_1d_chain",
    "get_max_undirected_weight",
    "tree_center",
]
