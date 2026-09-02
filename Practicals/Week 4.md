# 🧩 Solving the 8-Puzzle — Week 4 Practical (CSE 276)

### *Practical 2a: Implement Hill Climbing → Practical 2b: Implement A\* Search → Compare Solution Quality, States Explored, and Success Rate, using Python + Matplotlib (all in Google Colab)*

> **What we're building today:** Project 1 (Week 3) was about finding *a* path with no extra information — just "is this neighbor visited or not." Today's search algorithms get a hint: a **heuristic**, an educated guess of "how far am I from the goal?" You'll implement **Hill Climbing** (Practical 2a) and **A\* Search** (Practical 2b) to solve the classic **8-puzzle** — slide numbered tiles around a 3×3 grid until they're in order — using the *exact same heuristic* for both. By the end you'll have hard evidence for why "having a heuristic" and "using a heuristic well" are two very different things.

> 🧑‍🎓 **This is Project 2.** It reuses Week 3's core shape — frontier, visited tracking, "who found me" bookkeeping — but every state now carries a **score**, and that score is what steers the search.

> 💻 **Runtime:** Google Colab → CPU (default). No files to upload — everything is generated in code, so there's nothing to lose if you're starting fresh today.

**Session plan (≈120 minutes, back-to-back):**

| Time | Block | Focus |
|------|-------|-------|
| 🕛 0:00 – 0:10 | **Recap** | Quick reconnect to Week 3 — what a frontier + visited set gave us, what it didn't |
| 🕧 0:10 – 0:25 | **Bridge** | Meet the 8-puzzle, meet the heuristic (Manhattan distance), meet "why blind search struggles here" |
| 🕐 0:25 – 1:00 | **Practical 2a** | Implement Hill Climbing as a puzzle solver; test, visualize, instrument |
| 🕜 1:00 – 1:40 | **Practical 2b** | Implement A\* as a puzzle solver; compare against Hill Climbing |
| 🕝 1:40 – 1:55 | **Benchmark Harness** | Run both across many random puzzles — success rate, solution length, states explored, time |
| 🕞 1:55 – 2:00 | **Wrap-up** | Recap, save your solver, preview of what's next |
| ➕ Extra | **Extend It** | Optional deeper exercises if we move faster than expected |

---

## 🗺️ The Big Picture (Read This First)

```mermaid
flowchart LR
    A["🧩 Meet the 8-puzzle<br/>+ Manhattan distance"] --> B["⛰️ Code Hill Climbing<br/>(greedy, no backtrack)"]
    B --> C["🚧 Watch it get stuck<br/>at a local optimum"]
    C --> D["⭐ Code A*<br/>(frontier + backtracking)"]
    D --> E["⚖️ Compare:<br/>success · length · states"]
    style A fill:#028090,color:#fff
    style B fill:#F26B0F,color:#fff
    style C fill:#B23A48,color:#fff
    style D fill:#4A4A4A,color:#fff
    style E fill:#3ECF8E,color:#053b26
```

**The one idea to hold onto all session:** both algorithms today use the *same* heuristic function to estimate "distance to goal." The only difference is **what they do with that number**. Hill Climbing looks at it once per step and never looks back. A\* remembers every option it has ever seen and always tries the most promising one next — even if that means backtracking. Same information, radically different outcomes. That's today's whole lesson.

---

## 📋 What You'll Need

1. Your Google account + a fresh (or continued) Colab notebook
2. Nothing to upload — we generate puzzles in code, so today is fully self-contained
3. Comfort with Week 3's `parent` dictionary trick (who discovered whom) — we reuse it today to reconstruct the sequence of moves

---

# 🕛 RECAP (0:00 – 0:10)

### 0.1 — What Week 3 gave us, and what it didn't

BFS and DFS both answered "is there a path, and if so, what is it?" — but neither one had any sense of *direction*. They explored blindly: BFS spread out evenly in all directions, DFS committed to one direction until it hit a wall. Neither could tell "this move is clearly getting me closer to the goal" from "this move is clearly getting me further away."

Today's problem — the 8-puzzle — has a state space far too large to explore blindly in real time (181,440 reachable arrangements, to be exact). We need algorithms that can **use information about the goal** to cut that search down. That information is called a **heuristic**.

### 0.2 — Set up the puzzle representation

```python
import random
import heapq
import itertools
import time
import matplotlib.pyplot as plt

# The goal: tiles 1-8 in reading order, blank (0) in the bottom-right corner
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

def print_puzzle(state):
    """Quick text view: 3 rows of 3, blank shown as underscore."""
    for r in range(3):
        row = state[r*3:(r+1)*3]
        print(" ".join(str(v) if v != 0 else "_" for v in row))
    print()

print("Goal state:")
print_puzzle(GOAL)
```

> 🔑 We represent the puzzle as a flat tuple of 9 numbers (0 = blank), read left-to-right, top-to-bottom — position 0 is the top-left square, position 8 is the bottom-right square. Tuples (not lists) because we'll be putting states into sets and dictionaries in a minute, exactly like `visited` in Week 3 — and only tuples (not lists) can go into a Python set.

### 0.3 — Build `get_neighbors()` — this week's version of the adjacency list

In Week 3, `adj[node]` told you where you could go next. Today there's no pre-built graph — neighbors are *computed on the fly* by sliding the blank tile up, down, left, or right.

```python
def swap(state, i, j):
    """Return a new state with positions i and j swapped."""
    lst = list(state)
    lst[i], lst[j] = lst[j], lst[i]
    return tuple(lst)

def get_neighbors(state):
    """All states reachable by one legal slide of the blank tile."""
    neighbors = []
    blank = state.index(0)
    row, col = divmod(blank, 3)

    if row > 0: neighbors.append(swap(state, blank, blank - 3))   # slide blank up
    if row < 2: neighbors.append(swap(state, blank, blank + 3))   # slide blank down
    if col > 0: neighbors.append(swap(state, blank, blank - 1))   # slide blank left
    if col < 2: neighbors.append(swap(state, blank, blank + 1))   # slide blank right

    return neighbors

# Sanity check — the goal state should have exactly 2 legal moves (blank is in a corner)
print(f"Neighbors of GOAL: {len(get_neighbors(GOAL))}")
for n in get_neighbors(GOAL):
    print_puzzle(n)
```

---

# 🕧 BRIDGE (0:10 – 0:25): Meet the Heuristic

*(Discussion + light typing — sets up both practicals.)*

### 1.1 — Manhattan distance: "how far is every tile from home?"

A heuristic is just a **fast, approximate guess**. For the 8-puzzle, the standard heuristic is **Manhattan distance**: for every tile, count how many rows + columns it is away from where it belongs in the goal, then add all of those up. It ignores the fact that tiles block each other — that's exactly what makes it fast to compute and only an *estimate*, not the true answer.

```mermaid
flowchart LR
    A["Tile '5' is here"] -->|"1 row + 2 cols away"| B["Tile '5' belongs here"]
    style A fill:#B23A48,color:#fff
    style B fill:#3ECF8E,color:#053b26
```

```python
GOAL_POS = {val: idx for idx, val in enumerate(GOAL)}

def manhattan_distance(state, goal=GOAL):
    """Sum, over every tile, of how many rows+cols it is from its goal position."""
    goal_pos = {val: idx for idx, val in enumerate(goal)}
    total = 0
    for idx, val in enumerate(state):
        if val == 0:
            continue   # we don't score the blank
        goal_idx = goal_pos[val]
        row1, col1 = divmod(idx, 3)
        row2, col2 = divmod(goal_idx, 3)
        total += abs(row1 - row2) + abs(col1 - col2)
    return total

print("Heuristic of GOAL itself:", manhattan_distance(GOAL))   # should be 0 — already home
```

> 🧠 **Why 0 matters:** a heuristic of 0 should mean "you have arrived." If `manhattan_distance(GOAL)` printed anything other than 0, something in the goal-position mapping would be broken — a useful sanity check to keep in your back pocket for any heuristic you write in this course.

### 1.2 — Generate a solvable puzzle honestly

Not every random arrangement of 1–8 and a blank is solvable — exactly half of them are dead ends, mathematically. Rather than generate a random layout and hope, we start **from the goal and shuffle backwards** using only legal slides. Every state reached this way is guaranteed solvable, because we can always walk the same moves back.

```python
def make_random_puzzle(num_shuffles=25, seed=None):
    """Start at GOAL and take random legal moves — always solvable, by construction."""
    rng = random.Random(seed)
    state = GOAL
    last_state = None
    for _ in range(num_shuffles):
        neighbors = get_neighbors(state)
        if last_state in neighbors and len(neighbors) > 1:
            neighbors.remove(last_state)   # don't immediately undo the last move — keeps it scrambled
        last_state = state
        state = rng.choice(neighbors)
    return state

puzzle = make_random_puzzle(num_shuffles=25, seed=42)
print("Starting puzzle:")
print_puzzle(puzzle)
print("Heuristic estimate (Manhattan distance to goal):", manhattan_distance(puzzle))
```

> 🎯 **This is today's `parent`-dictionary equivalent moment:** just like Week 3's "how do I turn visit order into a path," today's new piece is "how do I turn a random shuffle into a guaranteed-solvable starting puzzle." Both are bookkeeping tricks, not new algorithms.

---

# 🕐 PRACTICAL 2a (0:25 – 1:00): Implement Hill Climbing

## Goal: a function that takes (start, goal) and greedily slides toward a lower heuristic score, one step at a time

### 2.1 — Write `hill_climbing()`

```python
def hill_climbing(start, goal=GOAL, max_steps=1000):
    """
    Greedy local search: at every step, move to whichever neighbor has the
    LOWEST heuristic score — but only if it's better than staying put.
    No visited set, no backtracking, no memory of anywhere it's already been.

    Returns: (path, states_explored, success)
    path = list of states from start to goal (or to wherever it got stuck)
    success = True only if it actually reached the goal
    """
    current = start
    path = [current]
    states_explored = 0

    for _ in range(max_steps):
        states_explored += 1
        if current == goal:
            return path, states_explored, True

        current_h = manhattan_distance(current, goal)
        best_neighbor = None
        best_h = current_h   # a neighbor only counts if it beats THIS score

        for neighbor in get_neighbors(current):
            h = manhattan_distance(neighbor, goal)
            if h < best_h:
                best_h = h
                best_neighbor = neighbor

        if best_neighbor is None:
            # every neighbor is equal or worse — we're stuck (a local optimum or plateau)
            return path, states_explored, False

        current = best_neighbor
        path.append(current)

    return path, states_explored, False   # ran out of steps without reaching the goal
```

> 🔍 **Compare this to `bfs_route` from Week 3.** There's no `queue`, no `visited` set, no `parent` dictionary for backtracking — `path` is just "everywhere we've been, in order," because Hill Climbing never needs to ask "who discovered me?" It only ever has one current state, and it only ever moves forward. That simplicity is also its weakness, which you're about to see.

### 2.2 — Run it on the puzzle from section 1.2

```python
hc_path, hc_states, hc_success = hill_climbing(puzzle)

print("Hill Climbing reached the goal:", hc_success)
print("Steps taken:", len(hc_path) - 1)
print("States explored:", hc_states)
print("\nFinal state reached:")
print_puzzle(hc_path[-1])
```

> 🧪 **Before running:** look at the starting puzzle's heuristic score from section 1.2. Predict — will Hill Climbing solve it, or get stuck? There's no way to know for sure without running it (that unpredictability is itself part of the lesson) — but form a guess anyway.

### 2.3 — Visualize the attempt

```python
def draw_puzzle(state, ax, title=""):
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for idx, val in enumerate(state):
        row, col = divmod(idx, 3)
        x, y = col, 2 - row   # flip rows so the grid reads top-to-bottom
        color = "white" if val == 0 else "lightblue"
        ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor="gray"))
        if val != 0:
            ax.text(x + 0.5, y + 0.5, str(val), ha="center", va="center",
                     fontsize=16, fontweight="bold")
    ax.set_title(title, fontsize=9)

def pick_frames(path_len, max_frames=6):
    """Evenly spaced indices into a path, so long paths don't need 40 subplots."""
    if path_len <= max_frames:
        return list(range(path_len))
    step = (path_len - 1) / (max_frames - 1)
    return sorted(set(round(i * step) for i in range(max_frames)))

def draw_path(path, title_prefix="", max_frames=6):
    frames = pick_frames(len(path), max_frames)
    fig, axes = plt.subplots(1, len(frames), figsize=(2.6 * len(frames), 2.8))
    if len(frames) == 1:
        axes = [axes]
    for ax, idx in zip(axes, frames):
        draw_puzzle(path[idx], ax, title=f"{title_prefix} step {idx}")
    plt.tight_layout()
    plt.show()

draw_path(hc_path, title_prefix="Hill Climbing")
```

### 2.4 — Instrument it: why did it stop where it stopped?

```python
final_h = manhattan_distance(hc_path[-1])
print(f"Heuristic score at the final state: {final_h}")

if hc_success:
    print("Reached the goal — heuristic hit 0.")
else:
    print("Stuck: every neighbor of the final state has an equal or worse heuristic score.")
    print("This is a LOCAL OPTIMUM — the algorithm has no way to know a better path")
    print("exists somewhere it would have to temporarily move AWAY from the goal to reach.")
```

> 🧠 **Why this matters:** Hill Climbing is fast and uses almost no memory — it only ever tracks one state. But it has no concept of "worth a temporary step backward for a bigger payoff later." Once every visible neighbor looks equal or worse, it simply stops, whether or not the goal was actually reachable from there. That's the trade-off we're about to measure honestly.

### 2.5 — 🎮 Your turn: run Hill Climbing on 4 more random puzzles

```python
for seed in [1, 2, 3, 4]:
    p = make_random_puzzle(num_shuffles=25, seed=seed)
    path, states, success = hill_climbing(p)
    print(f"seed={seed}: success={success}, steps={len(path)-1}, states_explored={states}")
```

> 🧪 **~10 minutes.** Tally how many of your 4 runs succeeded. Then try `num_shuffles=10` (an easier, closer-to-solved puzzle) versus `num_shuffles=40` (a harder one) — does the success rate change? Keep your tally; we'll build a proper version of this in the Benchmark Harness.

---

## 🛠️ Troubleshooting — Practical 2a

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `hill_climbing` always "succeeds" in 0 steps | `start` puzzle equals `GOAL` — `make_random_puzzle` wasn't actually called, or `num_shuffles=0` | Re-check section 1.2; confirm `print_puzzle(puzzle)` shows a scrambled grid before solving it |
| Every run gets stuck immediately (`states_explored` is very small) | `manhattan_distance` returning the same value for every neighbor — often a bug in `GOAL_POS`/`goal_pos` construction | Print `manhattan_distance(GOAL)` — it must be exactly `0`; if not, the goal-position mapping is broken |
| `hc_path` visualization shows only 1–2 tiles moving | Totally normal — Hill Climbing only slides the blank one square per step, same as a real puzzle move | Not a bug; increase `max_frames` in `draw_path` to see more intermediate steps |
| Function runs but never terminates | `max_steps` too high combined with an unlucky infinite plateau-chasing shuffle bug | Shouldn't happen with the guard in 2.1 (`best_neighbor is None` returns immediately) — if it does, double check no code was removed from that block |

---

# 🕜 PRACTICAL 2b (1:00 – 1:40): Implement A\* Search

## Same heuristic, opposite philosophy — remember everything, never fully commit

### 3.1 — Write `astar()`

```python
def astar(start, goal=GOAL):
    """
    Keeps a priority queue ("open set") of every state it has ever discovered,
    always expanding whichever one currently looks most promising:
        f(state) = g(state) + h(state)
        g = actual steps taken so far to reach this state
        h = heuristic estimate of steps still remaining (Manhattan distance)

    Unlike Hill Climbing, A* can "change its mind" — a state that looked
    promising can be set aside in favor of a newly discovered better one,
    and picked back up later if nothing beats it.

    Returns: (path, states_explored, success)
    """
    counter = itertools.count()   # tie-breaker so heapq never has to compare puzzle states directly
    start_h = manhattan_distance(start, goal)
    open_set = [(start_h, 0, next(counter), start, [start])]
    best_g = {start: 0}   # cheapest steps-so-far found for each state seen
    states_explored = 0

    while open_set:
        f, g, _, current, path = heapq.heappop(open_set)
        states_explored += 1

        if current == goal:
            return path, states_explored, True

        for neighbor in get_neighbors(current):
            new_g = g + 1
            if neighbor not in best_g or new_g < best_g[neighbor]:
                best_g[neighbor] = new_g
                new_f = new_g + manhattan_distance(neighbor, goal)
                heapq.heappush(open_set, (new_f, new_g, next(counter), neighbor, path + [neighbor]))

    return None, states_explored, False   # open_set emptied without finding the goal (shouldn't happen if solvable)
```

> 🔑 **Two deliberate differences from `hill_climbing`, both structural:**
> 1. `open_set` is a **priority queue** (Python's `heapq`) holding *every* undecided state, not just the current one — this is Week 3's `queue`/`stack` idea, but sorted by score instead of insertion order.
> 2. `best_g` plays the same role as Week 3's `visited` set, but stores *how cheaply* we reached each state — so if a shorter route to an already-seen state turns up later, A\* is allowed to reconsider it. Hill Climbing could never do this; it doesn't remember anywhere it's been.
> 3. The `counter` tie-breaker exists purely so that if two states ever have the exact same `f` and `g` score, Python compares the counter (always unique) instead of trying to compare puzzle states or paths — a small but important plumbing detail with `heapq`.

### 3.2 — Run it on the same puzzle Hill Climbing was given

```python
astar_path, astar_states, astar_success = astar(puzzle)

print("A* reached the goal:", astar_success)
print("Steps taken:", len(astar_path) - 1)
print("States explored:", astar_states)
```

### 3.3 — Side-by-side visualization

```python
print("Hill Climbing:")
draw_path(hc_path, title_prefix="HC")

print("A*:")
draw_path(astar_path, title_prefix="A*")
```

### 3.4 — Compare directly

```python
print(f"{'Metric':<22}{'Hill Climbing':<18}{'A*':<18}")
print("-" * 58)
print(f"{'Reached goal?':<22}{str(hc_success):<18}{str(astar_success):<18}")
print(f"{'Steps taken':<22}{len(hc_path)-1:<18}{len(astar_path)-1:<18}")
print(f"{'States explored':<22}{hc_states:<18}{astar_states:<18}")
```

> 🧠 **What to expect, and why:** if the puzzle is solvable (and `make_random_puzzle` guarantees it is), A\* is *guaranteed* to find a path — and with Manhattan distance specifically, that path is guaranteed to be the **shortest possible** one, because Manhattan distance never overestimates the true remaining distance (this property is called being **admissible**). Hill Climbing has no such guarantee either way. It usually explores far fewer states when it works — but "usually works, cheaply, when it works" is a very different promise than "always works, optimally." That gap is the whole point of today's comparison.

---

## 🛠️ Troubleshooting — Practical 2b

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `TypeError: '<' not supported between instances of 'tuple' and 'tuple'` | The `itertools.count()` tie-breaker was removed or misplaced in the pushed tuple | Re-check section 3.1 — `next(counter)` must be the 3rd element, before `current` and `path` |
| A\* runs but seems slow / notebook pauses noticeably | Normal for harder puzzles (`num_shuffles=40+`) — the open set can grow into the thousands of entries | Not a bug; this is exactly the "states explored" number growing, which we'll measure honestly in the benchmark |
| `astar_path` is longer than `hc_path` even though A\* "should" be optimal | Compare *fairly* — A\* is optimal only when Hill Climbing actually reached the goal at all; if `hc_success` is `False`, there's nothing to compare against, since Hill Climbing never finished | Check `hc_success` before comparing path lengths |
| `states_explored` for A\* seems too low compared to Hill Climbing | Genuinely possible on easy, near-solved puzzles — the heuristic does most of the work and A\* homes in fast | Try a harder puzzle (`num_shuffles=40`) to see the more typical pattern re-emerge |

---

# 🕝 BENCHMARK HARNESS (1:40 – 1:55): Measuring Heuristic Efficiency

## Same heuristic, two very different report cards

### 4.1 — Why timing needs the same "many repeats" honesty as Week 3

```python
def time_it(func, start, repeats=50):
    t0 = time.perf_counter()
    for _ in range(repeats):
        func(start)
    t1 = time.perf_counter()
    return ((t1 - t0) / repeats) * 1000   # average ms per call

hc_avg_ms = time_it(hill_climbing, puzzle)
astar_avg_ms = time_it(astar, puzzle)

print(f"Hill Climbing average time: {hc_avg_ms:.4f} ms")
print(f"A* average time:            {astar_avg_ms:.4f} ms")
```

> ⚠️ On easy puzzles both will look nearly instant — the honest comparison isn't a single millisecond number, it's what happens as puzzles get *harder*, which the full harness below tests directly.

### 4.2 — Full comparison across many random puzzles

```python
def run_benchmark(num_puzzles=20, num_shuffles=25):
    rows = []
    for seed in range(num_puzzles):
        p = make_random_puzzle(num_shuffles=num_shuffles, seed=seed)

        hc_p, hc_s, hc_ok = hill_climbing(p)
        a_p, a_s, a_ok = astar(p)

        rows.append({
            "seed": seed,
            "hc_success": hc_ok,
            "hc_steps": len(hc_p) - 1 if hc_ok else None,
            "hc_states": hc_s,
            "astar_success": a_ok,
            "astar_steps": len(a_p) - 1 if a_ok else None,
            "astar_states": a_s,
        })
    return rows

results = run_benchmark(num_puzzles=20, num_shuffles=25)

hc_success_rate = sum(r["hc_success"] for r in results) / len(results) * 100
astar_success_rate = sum(r["astar_success"] for r in results) / len(results) * 100
avg_hc_states = sum(r["hc_states"] for r in results) / len(results)
avg_astar_states = sum(r["astar_states"] for r in results) / len(results)

print(f"{'Metric':<28}{'Hill Climbing':<18}{'A*':<18}")
print("-" * 64)
print(f"{'Success rate':<28}{f'{hc_success_rate:.0f}%':<18}{f'{astar_success_rate:.0f}%':<18}")
print(f"{'Avg states explored':<28}{avg_hc_states:<18.1f}{avg_astar_states:<18.1f}")
```

> 🎯 **This table is your Project 2 deliverable in miniature.** Screenshot or save this output. The headline number to discuss: Hill Climbing's success rate will almost never be 100% — A\*'s always will be, for any solvable puzzle. That single fact, next to how many states each one explored to earn its result, *is* "heuristic efficiency" made concrete.

---

## 🚀 Extend It (Buffer content — use if we're ahead of schedule)

1. **Random-restart Hill Climbing:** wrap `hill_climbing` in a loop — if it gets stuck, shuffle to a *new* random puzzle attempt starting from the same original state (or just restart from a random nearby state) and try again, up to N restarts. Does the success rate improve? At what cost in total states explored?
2. **Weighted A\*:** multiply the heuristic by a constant `w > 1` in the `f = g + h` calculation (i.e. `f = g + w*h`). Compare states explored and solution length against plain A\* (`w = 1`) — you're trading guaranteed optimality for speed. How large can `w` get before solutions start visibly getting longer?
3. **Harder puzzles:** raise `num_shuffles` to 60 or 80 in the benchmark. Does A\*'s states-explored number grow roughly linearly, or does it blow up? What does that suggest about how well Manhattan distance scales?
4. **A worse heuristic, for contrast:** write `misplaced_tiles(state, goal)` — just count how many tiles are in the wrong position (ignoring *how far* wrong). Rerun A\* with this heuristic instead — it's still admissible, but far less informative. Compare states explored against Manhattan distance on the same puzzles.
5. **Uniform-cost search bridge:** delete the `+ manhattan_distance(...)` term from A\*'s `new_f` calculation (i.e. rank purely by `g`). This turns A\* into plain uniform-cost search — effectively Week 3's BFS, generalized. Confirm it still finds the optimal path, just by exploring far more states.

---

# 🕞 WRAP-UP (1:55 – 2:00)

### ✅ What You Learned Today

- 🧭 Learned what a **heuristic** is — a fast, approximate "distance to goal" estimate — and implemented Manhattan distance for the 8-puzzle
- ⛰️ Implemented **`hill_climbing()`**: a greedy, memory-light search that follows the heuristic downhill one step at a time, with no way to backtrack
- 🧱 Watched Hill Climbing get stuck at a **local optimum** — and understood *why*, not just *that* it happened
- ⭐ Implemented **`astar()`**: a search that keeps every option on the table via a priority queue, using `f = g + h` to always expand the most promising state next
- 📏 Learned the concept of an **admissible heuristic** — one that never overestimates — and why that guarantees A\*'s solutions are optimal
- 🔢 Measured **success rate** and **states explored** as honest, side-by-side efficiency numbers for two algorithms sharing one heuristic
- 📊 Built a **benchmark harness** across 20 random puzzles — your first real evidence for "more informed search beats blind local search," backed by numbers instead of intuition

### 👀 Preview of What's Next

Search so far has been about finding paths and solving puzzles with hard, deterministic rules for "what's a legal move." Week 5 turns to a different kind of reasoning entirely: **rule-based expert systems** — encoding human expert knowledge as if-then production rules — and **knowledge representation**, giving a program a structured way to *know* things about a real-world domain, not just search through states.

> Save your notebook (`File → Save a copy in Drive`) with `get_neighbors`, `manhattan_distance`, `hill_climbing`, and `astar` intact — `manhattan_distance` in particular is a pattern (estimate-then-guide) you'll see again.

---

## 🧰 Quick Reference Card — Full Session

```python
# ── PUZZLE BASICS ──
import random, heapq, itertools

GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

def swap(state, i, j):
    lst = list(state); lst[i], lst[j] = lst[j], lst[i]
    return tuple(lst)

def get_neighbors(state):
    neighbors = []; blank = state.index(0); row, col = divmod(blank, 3)
    if row > 0: neighbors.append(swap(state, blank, blank - 3))
    if row < 2: neighbors.append(swap(state, blank, blank + 3))
    if col > 0: neighbors.append(swap(state, blank, blank - 1))
    if col < 2: neighbors.append(swap(state, blank, blank + 1))
    return neighbors

def manhattan_distance(state, goal=GOAL):
    goal_pos = {val: idx for idx, val in enumerate(goal)}
    total = 0
    for idx, val in enumerate(state):
        if val == 0: continue
        r1, c1 = divmod(idx, 3); r2, c2 = divmod(goal_pos[val], 3)
        total += abs(r1 - r2) + abs(c1 - c2)
    return total

def make_random_puzzle(num_shuffles=25, seed=None):
    rng = random.Random(seed); state = GOAL; last_state = None
    for _ in range(num_shuffles):
        neighbors = get_neighbors(state)
        if last_state in neighbors and len(neighbors) > 1:
            neighbors.remove(last_state)
        last_state = state; state = rng.choice(neighbors)
    return state

# ── HILL CLIMBING ──
def hill_climbing(start, goal=GOAL, max_steps=1000):
    current = start; path = [current]; states = 0
    for _ in range(max_steps):
        states += 1
        if current == goal: return path, states, True
        current_h = manhattan_distance(current, goal)
        best_neighbor, best_h = None, current_h
        for nb in get_neighbors(current):
            h = manhattan_distance(nb, goal)
            if h < best_h: best_h, best_neighbor = h, nb
        if best_neighbor is None: return path, states, False
        current = best_neighbor; path.append(current)
    return path, states, False

# ── A* SEARCH ──
def astar(start, goal=GOAL):
    counter = itertools.count()
    open_set = [(manhattan_distance(start, goal), 0, next(counter), start, [start])]
    best_g = {start: 0}; states = 0
    while open_set:
        f, g, _, current, path = heapq.heappop(open_set)
        states += 1
        if current == goal: return path, states, True
        for nb in get_neighbors(current):
            new_g = g + 1
            if nb not in best_g or new_g < best_g[nb]:
                best_g[nb] = new_g
                new_f = new_g + manhattan_distance(nb, goal)
                heapq.heappush(open_set, (new_f, new_g, next(counter), nb, path + [nb]))
    return None, states, False
```

| Concept | One-liner |
|---------|-----------|
| **Heuristic** | A fast, approximate "distance to goal" estimate — Manhattan distance sums each tile's row+col offset from home |
| **Admissible heuristic** | Never overestimates the true remaining cost — this is what makes A\*'s answer guaranteed-optimal |
| **Local optimum** | A state where every neighbor looks equal or worse — Hill Climbing has no way to see past it |
| **`g` vs `h` vs `f`** | `g` = actual cost so far, `h` = estimated cost remaining, `f = g + h` = A\*'s ranking score |
| **Priority queue (`heapq`)** | Week 3's queue/stack, but always pops the *lowest-scored* item instead of the oldest/newest |
| **Hill Climbing vs A\*, structurally** | HC tracks one current state and never looks back; A\* tracks every state it's ever seen and can reconsider any of them |
| **Golden rule of today** | Two algorithms can use the *identical* heuristic and get wildly different guarantees — the guarantee comes from what you do with the number, not the number itself |
