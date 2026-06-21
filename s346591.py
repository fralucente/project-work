from Problem import Problem
from src.functions import precompute, aco, trips_from_back, refine_trip, expand

def solution(p:Problem):
    # TODO: implement your solution here
    G = p.graph
    alpha, beta = p.alpha, p.beta
    n=len(G.nodes())
    
    gold = {nd: G.nodes[nd]["gold"] for nd in G.nodes()}
    cities = [c for c in range(n) if c != 0]
    if not cities:
        return [(0, 0)]
    #n=len(cities),D=shortest_path distances,SB=shortest path sum of edge^beta,PR=predecessor matrix, The ones with p are the same but considering the shortest path distances with the load-aware cost function
    D, SB, PR, Dp, SBp, PRp = precompute(G, beta)
    
    n_ants = min(20, max(8, n // 5)) #ants proportional to the number of cities, but at least 8 and at most 20
    # run the ACO algorithm to get the best order of cities to visit, order is the best order of cities to visit, back is the trips cut, opt_cost is the optimal cost found by the ACO algorithm
    order, back, opt_cost = aco(
        cities, gold, D, SB, alpha, beta,
        time_limit=10, seed=42, n_ants=n_ants,
        a_exp=1.0, b_exp=3.0, rho=0.1, verbose=False, Dp=Dp, SBp=SBp,
    )
    trips = trips_from_back(order, back) #computes the trips from the order and back arrays returned by the ACO algorithm
    # final load-aware refinement of the order inside each trip
    trips = [refine_trip(t, gold, D, SB, alpha, beta, Dp, SBp) for t in trips]
    path = expand(trips, gold, D, SB, Dp, SBp, PR, PRp, alpha, beta) #expands the trips into a full path, using the precomputed shortest paths and load-aware shortest paths
    return path

for nc, dn, al, be in [(100, 0.2, 1, 1),(100, 1, 1, 1), (100, 0.2, 1, 2),(100, 1, 1, 2), (1000, 0.2, 1, 2), (1000, 0.2, 1, 3),(1000, 1, 1, 2), (1000, 1, 1, 3) ]:
    p = Problem(nc, density=dn, alpha=al, beta=be)
    base = p.baseline()
    path = solution(p)
    # cost of the literal returned path (edge by edge, as in the baseline)
    G = p.graph
    cur, load, cost = 0, 0.0, 0.0
    for node, g in path:
        e = G[cur][node]["dist"]
        cost += e + (al * e * load) ** be
        load += g
        if node == 0:
            load = 0.0
        cur = node
    print(f"n={nc} alpha={al} beta={be} | baseline={base:,.0f} "
            f"-> ACO={cost:,.0f}  ({100 * (base - cost) / base:.1f}% better)")
