# `functions.py` — function reference

The solution flow is: `precompute` → `aco` (which calls `split`) → `trips_from_back`
→ `refine_trip` → `expand`. Everything else is a helper.

---

## Precompute & shortest paths

### `precompute(G, beta)`
Preparation step: before searching for a solution it computes, once, all the
shortest-path information of the graph so the rest of the solver can reuse it
without recomputing anything. It extracts the graph edges into NumPy arrays,
then calls `all_pairs` twice — once with weight = distance (distance routing)
and once with weight = `e**beta` (penalty routing) — and fills in the
"secondary" sums with `sum_along`. If `beta == 1` the two routings coincide, so
it skips the second one and returns `None` for the `*p` matrices.
- **In**: graph `G`, cost parameter `beta`.
- **Out**: ` D, SB, PR, Dp, SBp, PRp`.
- **Example**: `n, D, SB, PR, Dp, SBp, PRp = precompute(G, 2)`.

### `edge_lookup(ea, eb, vals, n)`
Builds a dense `n x n` lookup table of per-edge values: `W[a, b] = vals` if
`a-b` is an edge, 0 otherwise. It is used by `sum_along`, which needs fast
access to the value of a single edge `pred -> b`. Both directions are filled in
(the graph is undirected), vectorised, with no Python loops.
- **In**: edge endpoints `ea, eb`, per-edge values `vals`, node count `n`.
- **Out**: symmetric dense `n x n` matrix.
- **Example**: edges (0-1, 1-2) with `vals = [16, 4]` → `W[0,1]=W[1,0]=16`,
  `W[1,2]=W[2,1]=4`.

### `all_pairs(n, ea, eb, w)`
Computes shortest paths between **all** pairs of nodes, under the weights `w`
you pass. It builds the sparse adjacency matrix using scipy.sparse.csr and runs SciPy's compiled
Dijkstra from every node. It returns two matrices: the minimum distances and the
predecessors. Key point: the distance matrix represents "the sum of the
minimised weight", so feeding it distance weights yields `D`, while feeding it
`e**beta` weights yields `SBp` directly.
- **In**: nodes `n`, edges `ea, eb`, weights `w`.
- **Out**: `D` (minimum distances), `PR` (predecessors).
- **Example**: graph 0-1=4, 1-2=2, 0-2=9 → `D[0,2]=6` (via 1, not 9),
  `PR[0,2]=1`.

### `sum_along(PR, Wedge, n)`
For every pair `(a, b)`, computes the sum of a per-edge quantity along the
already-found shortest path (encoded in `PR`). It is needed because Dijkstra
only gives you the sum of the weight it minimised, but you also need a
**different** sum along that same path (e.g. `sum(e**beta)` along the
distance-shortest paths). It does so with a small vectorised dynamic program:
`S[a,b] = S[a,pred] + Wedge[pred,b]`, applied to the whole matrix and repeated
until nothing changes (as many passes as the graph diameter).
- **In**: predecessors `PR`, per-edge quantity `Wedge`, nodes `n`.
- **Out**: matrix `S` of the sums along the paths.
- **Example**: with `Wedge = e**2` on the graph above → `SB[0,2] = 16 + 4 = 20`.

### `reconstruct(PR, a, b)`
Rebuilds the explicit node sequence of the shortest path `a -> b` from the
predecessor matrix. It walks backwards from `b` (each step taking `PR[a, .]`)
until it reaches `a`, then reverses the list. It is used in `expand` to turn a
"logical" leg between two cities into the actual node-by-node route.
- **In**: predecessors `PR`, start `a`, end `b`.
- **Out**: list of nodes of the path.
- **Example**: `reconstruct(PR, 0, 2) -> [0, 1, 2]`.

---

## Cost of a leg

### `legcost(a, b, w, D, SB, Dp, SBp, alpha, beta)`
Returns the cost of travelling a leg `a -> b` while carrying load `w`. If empty
(`w = 0`) it is simply the distance `D[a,b]`, because the penalty is zero. If
carrying a load, it computes the cost with **both** routings — distance routing
(`D + pw*SB`) and penalty routing (`Dp + pw*SBp`), with `pw = (alpha*w)**beta` —
and keeps the smaller. This is where the idea "pick the cheaper path depending
on the load" actually lives.
- **In**: endpoints `a, b`, load `w`, the matrices, `alpha/beta`.
- **Out**: cost (float).
- **Example**: `w=0 -> D[a,b]`; `w>0 -> min(D+pw*SB, Dp+pw*SBp)`.

### `leg_use_penalty(a, b, w, D, SB, Dp, SBp, alpha, beta)`
Performs the same comparison as `legcost`, but instead of the cost it returns
*which* routing wins (`True` = penalty routing is cheaper). It is used in
`expand`: when rebuilding the actual route of a leg, you must know whether to
follow the `PR` predecessors (distance) or the `PRp` ones (penalty), to stay
consistent with the cost the DP computed.
- **In**: same as `legcost`.
- **Out**: `True/False`.
- **Example**: with a high load and `beta=2`, often `True`.

---

## Trip split

### `split(order, gold, D, SB, alpha, beta, bound, Dp, SBp)`
Given the fixed city order produced by the ACO, it decides **where to cut it
into trips** (base -> block of cities -> base) at minimum total cost, with a
dynamic program. It defines `f[k]` = minimum cost to serve the first `k` cities,
and builds it noticing that the last trip of the optimal solution covers a final
block `order[i:k]`: hence `f[k] = min over i of f[i] + trip_cost(order[i:k])`.
It computes trip costs incrementally (extending the open path one leg at a time)
and uses `bound` to prune branches that already exceed the incumbent. It also
returns `back`, the pointers to reconstruct the cuts.
- **In**: city order, gold per node, matrices, `bound`, penalty routing.
- **Out**: `(optimal_cost, back)`.
- **Example**: `order=[A,B]` → chooses between "[A,B] in one trip" and
  "A alone + B alone".

### `trip_cost_ordered(seq, gold, D, SB, alpha, beta, Dp, SBp)`
Computes the exact cost of **a single trip** visiting cities `seq` in that
order: `0 -> seq -> 0`, with the gold accumulating along the way. It sums the
`legcost` of every leg plus the return leg to base. It is the "measuring stick"
used by `refine_trip` to check whether a reordering helps.
- **In**: city sequence `seq`, gold, matrices.
- **Out**: trip cost (float).
- **Example**: `[A,B] -> cost(0->A) + cost(A->B, load gA) + cost(B->0, load gA+gB)`.

### `refine_trip(seq, gold, D, SB, alpha, beta, Dp, SBp)`
Takes a single trip and **reorders its cities** to lower its cost, accounting
for the load. It is a 2-opt (tries reversing segments) that uses
`trip_cost_ordered` as its measure, so it favours orders where the heavy gold is
collected as late as possible (so it travels over fewer legs). Trips are short,
so it is cheap even trying all swaps.
- **In**: trip `seq`, gold, matrices.
- **Out**: the trip with reordered cities (cost <= the original).
- **Example**: `[B,A] -> [A,B]` if collecting A later reduces the penalty.

### `trips_from_back(order, back)`
Reconstructs the list of trips from the order and the `back` pointers produced
by `split`. It starts at the last position and walks back: `back[k]` says where
the last trip begins, then it jumps there and repeats, recovering the contiguous
blocks backwards. It is the classic "solution reconstruction" phase of a DP.
- **In**: order, `back` array.
- **Out**: list of trips (each a list of cities).
- **Example**: `back=[_,0,0,2,2,4] -> [[c0,c1],[c2,c3],[c4]]`.

---

## ACO (search for the order)

### `greedy_order(cities, D)`
Builds a reasonable initial visiting order with the nearest-neighbour rule: it
starts from the city closest to the base and always moves to the closest
not-yet-visited city. It serves as the ACO's **initial seed**: it gives the
colony a decent starting point and, combined with the split, guarantees the
solution is never worse than the baseline.
- **In**: city list, distance matrix.
- **Out**: a visiting order.
- **Example**: with the base in the centre, it chains the nearby cities one by one.

### `two_opt(order, D, deadline, max_passes=4)`
A local search that improves an order by shortening its length: it tries
**reversing segments** of the route and accepts a reversal if it reduces the
total distance, repeating while it improves (within a number of passes or a time
deadline). It is the classic 2-opt of the travelling-salesman problem. Here it
works on pure distance as a fast heuristic; the real load-aware cost is then
recomputed by the split.
- **In**: order, distances, deadline, max passes.
- **Out**: improved order.
- **Example**: `...A-D-C-B-E... -> ...A-B-C-D-E...` if it shortens.

### `build_candidate_lists(cities, D, k)`
For each city, precomputes the list of its `k` nearest cities. It speeds up
`ant_walk`: instead of evaluating all cities at each step, the ant only
considers the most promising neighbours. It sorts each row of the distance
matrix and keeps the first `k`.
- **In**: cities, distances, number `k`.
- **Out**: dictionary city -> list of its `k` neighbours.
- **Example**: with `k=10`, each city only "sees" the 10 closest.

### `ant_walk(cities, tau, eta, cand, a_exp, b_exp, rng)`
Simulates **one ant** building a complete visiting order, one city at a time. At
each step it picks the next city (among the nearby not-yet-visited candidates)
with a probabilistic "roulette" choice: the probability is proportional to
`tau**a_exp * eta**b_exp`, i.e. the pheromone `tau` (collective memory of what
worked) times the heuristic `eta` (1/distance, prefer nearby cities). It is the
exploratory heart of the ACO.
- **In**: cities, pheromone `tau`, heuristic `eta`, candidate lists, exponents, RNG.
- **Out**: a complete order (a candidate solution).
- **Example**: with high pheromone on `(A,B)`, after A it picks B with high probability.

### `aco(cities, gold, D, SB, alpha, beta, *, time_limit, seed, n_ants, a_exp, b_exp, rho, verbose, Dp, SBp)`
The engine that orchestrates everything. While there is time (`time_limit`),
each iteration has `n_ants` ants build an order, scores each order by its optimal
cost after the `split`, tracks the best, **evaporates** the pheromone and
**reinforces** it on the edges of the best order (Max-Min Ant System scheme, with
bounds on `tau` to avoid stagnation), and occasionally applies `two_opt` to
polish. It starts from the greedy seed as the incumbent. It finally returns the
best order, its cuts and the cost.
- **In**: problem data, matrices, ACO hyperparameters, time budget.
- **Out**: `(best_order, best_back, best_cost)`.
- **Example**: `order, back, cost = aco(..., time_limit=10)`.

---

## Output Reconstruction

### `expand(trips, gold, D, SB, Dp, SBp, PR, PRp, alpha, beta)` (with inner `route`)
Turns the abstract trips (lists of cities) into the **final route** in the
required format `[(city, gold), ..., (0, 0)]`. For each leg the inner `route`
function decides, based on the current load, whether to follow the
distance-shortest path or the penalty-optimal one (via `leg_use_penalty`), and
rebuilds it node by node with `reconstruct`. Transit nodes are inserted with
gold 0; gold is collected only at the scheduled stops; the load is dropped on
return to base. This way every consecutive pair is a real edge and the literal
path cost equals the optimised one.
- **In**: trips, gold, matrices and predecessors of both routings.
- **Out**: route `[(city, gold), ..., (0, 0)]`.
- **Example**: trip `[A,B] -> [(...transit...,0), (A,gA), ..., (B,gB), (0,0)]`.

### `true_cost(path, G, alpha, beta)`
Computes the real cost of the returned route, edge by edge, exactly as the
baseline does: it simulates from the base summing `e + (alpha*e*load)**beta` over
each edge, updates the load by collecting gold, and resets it every time it
passes back through the base. It is a check: the number it produces must match
the cost the solver claims to have optimised.
- **In**: path, graph `G`, `alpha/beta`.
- **Out**: real cost (float).
- **Example**: used in tests to verify `reported_cost == true_cost`.
