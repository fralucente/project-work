# project-work

### Repository Setup

1. Create a Git repository named project-work.
2. Inside the repository, include:
    - A Python script named Problem.py which generates the problem through the class constructor and the baseline solution. 
    - A Python file named s<student_id>.py that contains a function named solution(p:Problem) which receives as input an instance of the class Problem which generates the problem.
    - A folder named src/ containing all additional code required to run your solution.
    - A TXT file named base_requirements.txt containing the basic python libraries that you need to run the code to generate the problem.


### Main File Requirements (s<student_id>.py)

1. Import the class responsible from Problem.py for generating the problem in your code.
2. Implement a method called solution() to place in s<student_id>.py that returns the optimal path in the following format: 
```python
[(c1, g1), (c2, g2), …, (cN, gN), (0, 0)]
```
where:
- c1, …, cN represent the sequence of cities visited.
- g1, …, gN represent the corresponding gold collected at each city.


### Rules
1. The thief must start and finish at (0, 0).
2. Returning to (0, 0) during the route is allowed to unload collected gold before continuing.
3. Don't forget to change the name of the file s123456.py provided as an example ;).

### Notes
- It is not necessary to push the report.pdf or log.pdf in this repo.
- It is mandatory to upload it in "materiale" section of "portale della didattica" at least 168 hours before the exam call.
- For well commented codes, I can't ensure a higher mark but they would be very welcome.
- In case you face any issue or you have any doubt text me at the email giuseppe.esposito@polito.it and professor Squillero giovanni.squillero@polito.it.
---

## How the solution works

### The problem
A thief starts and ends at the base (city `0`). Every other city holds some gold.
Moving along a graph edge of length `e` while carrying weight `w` costs
`e + (alpha * e * w) ** beta`, accounted **per edge**. Passing through the base
unloads the gold (the carried weight resets to 0). The goal is to collect all the
gold and bring it back to the base at minimum total cost. `Problem.baseline()`
(one independent round trip per city) is the reference to beat.

The work is split into five stages.

### 1. Precompute — the road map (`precompute`, `all_pairs`, `sum_along`)
Cities are not all directly connected, so first i compute, once, the shortest
paths between every pair of nodes with Dijkstra (SciPy). I use scipy instead 
of building it from scratch to reduce computation cost. For each pair i store the 
distance `D`, the penalty term `SB = sum(edge**beta)` along that path, and a
predecessor matrix `PR` to rebuild any route in `O(length)`.

### 2. Penalty-aware routing (the key idea)
The cost of a loaded leg is `distance + (alpha*w)**beta * sum(edge**beta)`. For
`beta > 1` the penalty does not care about total distance but about
`sum(edge**beta)`, which is smaller on a path made of many small edges. So i
also precompute a **second** routing that minimises `sum(edge**beta)` (`Dp`,
`SBp`, `PRp`), and for every leg i pick whichever of the two routings is cheaper
at the current load (`legcost`). This is where most of the improvement over the
baseline comes from when `beta > 1`. For `beta == 1` the two routings coincide,
so the second one is skipped.

### 3. ACO — search the visiting order (`aco`, `ant_walk`)
Finding the best order in which to visit the cities is a hard combinatorial
problem (TSP-like). I use Ant Colony Optimization (Max-Min Ant
System): a population of "ants" each builds an order step by step, choosing the
next city with probability proportional to `pheromone^a_exp * heuristic^b_exp`
(the heuristic being `1/distance`). After each iteration the pheromone evaporates
and the best orders reinforce it. The search is time-bounded and seeded with a
greedy nearest-neighbour solution.

### 4. Split DP — where to unload (`split`, `trips_from_back`)
Given an order, deciding where to return to the base to unload is solved
**exactly** with a dynamic program: `f[k]` = minimum cost to serve the first `k`
cities, built from `f[k] = min over i of f[i] + trip_cost(order[i:k])`. Because a
single-city trip is a special case, the optimal split is **provably never worse
than the baseline**. Each ant's order is scored by its optimal split cost, so the
ACO only has to learn a good order while the DP handles the load trade-off.

### 5. Refinement and output (`refine_trip`, `expand`, `true_cost`)
The best trips are reordered internally with a load-aware 2-opt (`refine_trip`,
which collects the heavy gold as late as possible), then expanded node by node
into the required format `[(c1, g1), ..., (0, 0)]` (`expand`). Each leg is
expanded along the routing the cost model chose, and transit cities are listed
with gold 0, so the literal cost of the returned path equals the optimised cost
(checked by `true_cost`).

### Results vs the baseline (seed 42)
| instance (n, density, alpha, beta) | baseline | this solver | improvement |
|---|---|---|---|
| n=100, d=0.2, alpha=1, beta=1 | 25,266 | 25,242 | 0.1% |
| n=100, d=1.0, alpha=1, beta=1 | 18,266 | 18,261 | 0.0% |
| n=100, d=0.2, alpha=1, beta=2 | 5,334,402 | 4,048,692 | 24.1% |
| n=100, d=1.0, alpha=1, beta=2 | 5,404,978 | 1,555,193 | 71.2% |
| n=1000, d=0.2, alpha=1, beta=2 | 37,545,928 | 8,607,032 | 77.1% |
| n=1000, d=0.2, alpha=1, beta=3 | 10,600,895,131 | 348,531,359 | 96.7% |
| n=1000, d=1.0, alpha=1, beta=2 | 57,580,019 | 4,424,377 | 92.3% |
| n=1000, d=1.0, alpha=1, beta=3 | 20,943,050,224 | 105,549,008 | 99.5% |

For `beta = 1` the baseline is already near-optimal (carrying each city's gold
straight home alone is the minimum possible penalty, and distance is a tiny part
of the cost), so the gain is ~0% — this is structural, not a limitation of the
search. The large gains appear for `beta > 1`, driven by the penalty-aware
routing. A function-by-function reference is in `src/FUNCTIONS.md`.
