

import random
import time

import numpy as np
from scipy.sparse import csr_matrix 
from scipy.sparse.csgraph import dijkstra


#PRECOMPUTATION AND PATH COSTS
def precompute(G, beta):
    '''
    This function receives a Graph and the beta parameter, and precomputes the shortest path distances (using scipyDijkstra) 
    and the sum of edge^beta along those paths, for both the distance metric and the penalty-aware metric. 
    If beta == 1, the penalty-aware metric is the same as the distance metric, so it returns None for those values.
    '''
    n = G.number_of_nodes()
    m = G.number_of_edges()
    same = abs(beta - 1.0) < 1e-12

    # edge arrays
    ea = np.empty(m, dtype=np.int64)
    eb = np.empty(m, dtype=np.int64)
    ew = np.empty(m, dtype=float)
    for i, (a, b, w) in enumerate(G.edges(data="dist")):
        ea[i], eb[i], ew[i] = a, b, w
    ewb = ew ** beta

    Wdist = edge_lookup(ea, eb, ew, n)            # per-edge distance
   #routing for the distance metric (same as penalty metric when beta == 1)
    D, PR = all_pairs(n, ea, eb, ew)
    if same:                                       # beta == 1: nothing to gain
        return  D, D, PR, None, None, None
    Wpow = edge_lookup(ea, eb, ewb, n)            # per-edge distance**beta
    SB = sum_along(PR, Wpow, n)                   # sum edge**beta on distance paths
    #routing for the penalty-aware metric (different from distance when beta != 1)
    SBp, PRp = all_pairs(n, ea, eb, ewb)       # SBp is that metric's distance
    Dp = sum_along(PRp, Wdist, n)                 # sum edge distance on penalty paths
    return D, SB, PR, Dp, SBp, PRp


def edge_lookup(ea, eb, vals, n):
    """Dense symmetric per-edge lookup W[a, b] = vals (0 for non-adjacent)."""
    W = np.zeros((n, n))
    W[ea, eb] = vals
    W[eb, ea] = vals
    return W


def all_pairs(n, ea, eb, w):
    """All-pairs shortest paths (metric distance + predecessors) via SciPy's
    compiled Dijkstra, run on the sparse adjacency built from the edge weights w."""
    data = np.concatenate([w, w])
    rows = np.concatenate([ea, eb])
    cols = np.concatenate([eb, ea])
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    D, PR = dijkstra(A, directed=False, return_predecessors=True)
    return D, PR.astype(np.int32)


def sum_along(PR, Wedge, n):
    """For every pair (a, b), the sum of a per-edge quantity `Wedge` along the
    path encoded by predecessor matrix PR. Vectorised relaxation, a handful of
    passes (one per unit of graph diameter)."""
    S = np.zeros((n, n))
    idx = np.arange(n)
    valid = PR >= 0
    P = np.where(valid, PR, 0)                     # safe indices for the gather
    for _ in range(n):
        # S[a, b] = S[a, pred] + Wedge[pred, b]
        nxt = np.where(valid, S[idx[:, None], P] + Wedge[P, idx[None, :]], S)
        if np.array_equal(nxt, S):
            break
        S = nxt
    return S


def reconstruct(PR, a, b):
    """Rebuild the node sequence a -> ... -> b using the predecessor matrix."""
    path = [b]
    while b != a:
        b = int(PR[a, b])
        path.append(b)
    path.reverse()
    return path


#COST OF LEGS AND TRIPS
def legcost(a, b, w, D, SB, Dp, SBp, alpha, beta):
    """Cost of the leg a -> b carrying weight w, taking the cheaper of the
    distance routing and (when available) the penalty routing."""
    if w <= 0.0:
        return D[a, b]                       # empty: distance routing, no penalty
    pw = (alpha * w) ** beta
    c = D[a, b] + pw * SB[a, b]
    if Dp is not None:
        c2 = Dp[a, b] + pw * SBp[a, b]
        if c2 < c:
            c = c2
    return c


def leg_use_penalty(a, b, w, D, SB, Dp, SBp, alpha, beta):
    """True when the penalty routing is strictly cheaper for this leg/load
    (used at output time to expand the leg along the matching path)."""
    if Dp is None or w <= 0.0:
        return False
    pw = (alpha * w) ** beta
    return Dp[a, b] + pw * SBp[a, b] < D[a, b] + pw * SB[a, b]

#SPLIT AND REFINEMENT
def split(order, gold, D, SB, alpha, beta, bound=float('inf'), Dp=None, SBp=None):
    """Optimal partition of a visiting `order` into base-to-base trips.

    f[k] = min cost to serve the first k cities of `order`.
    A trip serving a contiguous block order[i:j] is a route
        0 -> order[i] -> ... -> order[j-1] -> 0
    with the gold accumulating along the way (heavier gold collected later in a
    trip is carried over fewer legs -- the order decides that).

    `bound` is the incumbent best full cost; branches that already exceed it are
    pruned (a trip's cost only grows as it gets longer).
    """
    m = len(order)
    f = [float('inf')] * (m + 1)
    back = [-1] * (m + 1)
    f[0] = 0.0

    for i in range(m):
        if f[i] == float('inf'):
            continue
        fi = f[i]
        load = 0.0          # weight carried *before* collecting the next city
        prev = 0            # last node visited (start of a trip = base)
        open_cost = 0.0     # cost of the open path 0 -> ... -> prev (no return yet)
        for j in range(i, m):
            city = order[j]
            # leg prev -> city, carried weight = current load (cheaper routing)
            if load > 0.0:
                pw = (alpha * load) ** beta
                lc = D[prev, city] + pw * SB[prev, city]
                if Dp is not None:
                    lc2 = Dp[prev, city] + pw * SBp[prev, city]
                    if lc2 < lc:
                        lc = lc2
                open_cost += lc
            else:
                open_cost += D[prev, city]
            if fi + open_cost >= bound:
                break  # this and any longer block can't beat the incumbent
            load += gold[city]            # collect gold at `city`
            # close the trip: leg city -> 0 carrying the full load
            pwf = (alpha * load) ** beta
            cl = D[city, 0] + pwf * SB[city, 0]
            if Dp is not None:
                cl2 = Dp[city, 0] + pwf * SBp[city, 0]
                if cl2 < cl:
                    cl = cl2
            total = fi + open_cost + cl
            if total < f[j + 1]:
                f[j + 1] = total
                back[j + 1] = i
            prev = city

    return f[m], back


def trip_cost_ordered(seq, gold, D, SB, alpha, beta, Dp=None, SBp=None):
    """Exact cost of a single trip 0 -> seq -> 0 with load accumulating."""
    load = 0.0
    prev = 0
    c = 0.0
    for city in seq:
        c += legcost(prev, city, load, D, SB, Dp, SBp, alpha, beta)
        load += gold[city]
        prev = city
    c += legcost(prev, 0, load, D, SB, Dp, SBp, alpha, beta)
    return c


def refine_trip(seq, gold, D, SB, alpha, beta, Dp=None, SBp=None):
    """Load-aware 2-opt on a single trip: reorder cities to cut the real
    (distance + load-penalty) trip cost. Trips are short, so this is cheap."""
    if len(seq) < 3:
        return seq
    seq = seq[:]
    best = trip_cost_ordered(seq, gold, D, SB, alpha, beta, Dp, SBp)
    improved = True
    while improved:
        improved = False
        for i in range(len(seq) - 1):
            for j in range(i + 1, len(seq)):
                cand = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                c = trip_cost_ordered(cand, gold, D, SB, alpha, beta, Dp, SBp)
                if c < best:
                    seq, best = cand, c
                    improved = True
    return seq


def trips_from_back(order, back):
    """Reconstruct the list of trips (each a list of cities) from split markers."""
    trips = []
    k = len(order)
    while k > 0:
        i = back[k]
        trips.append(order[i:k])
        k = i
    trips.reverse()
    return trips


#ACO LOGICS
def greedy_order(cities, D):
    """A quick nearest-neighbour visiting order, starting from the city
    closest to the base."""
    remaining = set(cities)
    cur = min(remaining, key=lambda c: D[0, c])
    order = []
    while remaining:
        order.append(cur)
        remaining.discard(cur)
        if not remaining:
            break
        cur = min(remaining, key=lambda c: D[cur, c])
    return order


def two_opt(order, D, deadline, max_passes=4):
    """Classic 2-opt: reverse segments that shorten the open tour distance.
    Restricted by a wall-clock deadline so it never blows the time budget."""
    n = len(order)
    if n < 4:
        return order
    order = order[:]
    improved = True
    passes = 0
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        for a in range(n - 1):
            ca, cb = order[a], order[a + 1]
            d_ab = D[ca, cb]
            for c in range(a + 2, n):
                cc = order[c]
                # reversing order[a+1 .. c] swaps edges
                #   (ca,cb) + (cc,cd)  ->  (ca,cc) + (cb,cd)
                if c + 1 < n:
                    cd = order[c + 1]
                    delta = (D[ca, cc] + D[cb, cd]) - (d_ab + D[cc, cd])
                else:
                    delta = D[ca, cc] - d_ab          # tail reversal (no cd)
                if delta < -1e-9:
                    order[a + 1:c + 1] = order[a + 1:c + 1][::-1]
                    improved = True
                    cb = order[a + 1]
                    d_ab = D[ca, cb]
            if time.time() >= deadline:
                break
    return order



def build_candidate_lists(cities, D, k):
    """For each city, the k nearest other cities (speeds up ant construction)."""
    cand = {}
    arr = np.array(cities)
    for c in cities:
        order = arr[np.argsort(D[c, arr])]
        order = [int(x) for x in order if x != c][:k]
        cand[c] = order
    return cand


def ant_walk(cities, tau, eta, cand, a_exp, b_exp, rng):
    """One ant builds a full visiting order using pheromone + heuristic."""
    unvisited = set(cities)
    start = rng.choice(cities)
    cur = start
    order = []
    while unvisited:
        order.append(cur)
        unvisited.discard(cur)
        if not unvisited:
            break
        # candidates = nearest unvisited; fall back to all unvisited
        choices = [c for c in cand[cur] if c in unvisited]
        if not choices:
            choices = list(unvisited)
        weights = [(tau[cur, c] ** a_exp) * (eta[cur, c] ** b_exp) for c in choices]
        tot = sum(weights)
        if tot <= 0.0:
            cur = rng.choice(choices)
        else:
            r = rng.random() * tot
            acc = 0.0
            cur = choices[-1]
            for c, w in zip(choices, weights):
                acc += w
                if acc >= r:
                    cur = c
                    break
    return order


#ACO ALGORITHM
def aco(cities, gold, D, SB, alpha, beta, *,
         time_limit, seed, n_ants, a_exp, b_exp, rho, verbose, Dp=None, SBp=None):
    rng = random.Random(seed)
    n_cities = len(cities)
    t0 = time.time()
    deadline = t0 + time_limit

    eta = np.zeros_like(D)
    with np.errstate(divide="ignore"):
        nz = D > 0
        eta[nz] = 1.0 / D[nz]

    k = min(n_cities - 1, max(8, int(0.2 * n_cities))) if n_cities > 1 else 1
    k = min(k, 25)
    cand = build_candidate_lists(cities, D, k)

    # --- greedy seed -> incumbent best ------------------------------------ #
    greedy = greedy_order(cities, D)
    best_cost, best_back = split(greedy, gold, D, SB, alpha, beta, Dp=Dp, SBp=SBp)
    best_order = greedy

    # --- pheromone (Max-Min Ant System bounds) ---------------------------- #
    tau_max = 1.0 / (rho * best_cost)
    tau_min = tau_max / (2.0 * n_cities)
    tau = np.full_like(D, tau_max)

    it = 0
    while time.time() < deadline:
        it += 1
        iter_best_cost = float('inf')
        iter_best_order = None
        for _ in range(n_ants):
            order = ant_walk(cities, tau, eta, cand, a_exp, b_exp, rng)
            # prune trips whose partial cost already exceeds the incumbent best:
            # an order worse than the incumbent returns inf and is simply ignored
            cost, _ = split(order, gold, D, SB, alpha, beta, bound=best_cost,
                             Dp=Dp, SBp=SBp)
            if cost < iter_best_cost:
                iter_best_cost = cost
                iter_best_order = order
            if time.time() >= deadline:
                break

        # an iteration may find nothing better than the incumbent: that's fine,
        # we keep going and reinforce the global best instead
        if iter_best_order is not None:
            # local-search boost on the iteration best
            improved = two_opt(iter_best_order, D, min(deadline, time.time() + 0.5))
            ic_cost, _ = split(improved, gold, D, SB, alpha, beta,
                                bound=iter_best_cost, Dp=Dp, SBp=SBp)
            if ic_cost < iter_best_cost:
                iter_best_cost, iter_best_order = ic_cost, improved
            if iter_best_cost < best_cost:
                best_cost, best_order = iter_best_cost, iter_best_order
                tau_max = 1.0 / (rho * best_cost)
                tau_min = tau_max / (2.0 * n_cities)

        # --- pheromone update (evaporate, then reinforce a good order) ---- #
        tau *= (1.0 - rho)
        # alternate iteration-best / global-best deposit (MMAS schedule)
        if iter_best_order is None or it % 4 == 0:
            deposit_order, deposit_cost = best_order, best_cost
        else:
            deposit_order, deposit_cost = iter_best_order, iter_best_cost
        amount = 1.0 / deposit_cost
        for u, v in zip(deposit_order, deposit_order[1:]):
            tau[u, v] += amount
            tau[v, u] += amount
        np.clip(tau, tau_min, tau_max, out=tau)

        if verbose and it % 10 == 0:
            print(f"  iter {it:4d}  best={best_cost:.2f}  ({time.time()-t0:.1f}s)")

    # final polish on the global best
    polished = two_opt(best_order, D, time.time() + min(2.0, max(0.0, deadline - time.time() + 2.0)))
    p_cost, _ = split(polished, gold, D, SB, alpha, beta, Dp=Dp, SBp=SBp)
    if p_cost < best_cost:
        best_cost, best_order = p_cost, polished

    _, best_back = split(best_order, gold, D, SB, alpha, beta, Dp=Dp, SBp=SBp)
    if verbose:
        print(f"  done: {it} iterations, best={best_cost:.2f}, {time.time()-t0:.1f}s")
    return best_order, best_back, best_cost

#OUTPUT EXPANSION AND TRUE COST
def expand(trips, gold, D, SB, Dp, SBp, PR, PRp, alpha, beta):
    """Turn the list of trips into the required output format:
        [(c1, g1), (c2, g2), ..., (0, 0)]
    Every consecutive pair is a real graph edge (intermediate transit cities are
    listed with gold 0), so the cost is unambiguous for any evaluator.
    Gold is collected only at the *scheduled* destination of each leg, exactly as
    assumed by the split DP. Each leg is expanded along whichever routing
    (distance or penalty) the cost model chose at that load, so the literal path
    cost equals the optimised cost.
    """
    def route(a, b, w):
        if leg_use_penalty(a, b, w, D, SB, Dp, SBp, alpha, beta):
            return reconstruct(PRp, a, b)
        return reconstruct(PR, a, b)

    path = []
    for trip in trips:
        prev = 0
        load = 0.0
        for city in trip:
            sp = route(prev, city, load)
            for nd in sp[1:-1]:            # transit nodes: no collection
                path.append((nd, 0))
            path.append((city, gold[city]))  # scheduled stop: collect its gold
            load += gold[city]
            prev = city
        # return to base to unload (carrying the full load)
        sp = route(prev, 0, load)
        for nd in sp[1:]:
            path.append((nd, 0))
    return path


def true_cost(path, G, alpha, beta):
    """Cost of the literal returned path, edge by edge (same model as baseline)."""
    cur = 0
    load = 0.0
    total = 0.0
    for node, g in path:
        e = G[cur][node]["dist"]          # consecutive entries are graph-adjacent
        total += e + (alpha * e * load) ** beta
        load += g
        if node == 0:
            load = 0.0                    # unload at the base
        cur = node
    return total


