import argparse
from dataclasses import dataclass
import heapq
from itertools import count
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter


SAMPLE_INPUT = {
    "grid": [
        ["A", 1, 1, "X", 2, 1, "T1"],
        [1, "X", 6, "X", 1, "X", 1],
        [1, 1, "T2", 1, 1, 1, 1],
        ["X", 1, "X", 8, "X", 1, "X"],
        [1, 1, 1, 1, "T3", 1, 1],
        [1, "X", 9, "X", 1, "X", 1],
        ["T4", 1, 1, "X", 1, 1, "B"],
    ],
    "starts": {"A": [0, 0], "B": [6, 6]},
    "treasures": [
        {"name": "T1", "position": [0, 6], "value": 9},
        {"name": "T2", "position": [2, 2], "value": 6},
        {"name": "T3", "position": [4, 4], "value": 7},
        {"name": "T4", "position": [6, 0], "value": 7},
    ],
}
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


@dataclass(frozen=True)
class GameState:
    position_a: int
    position_b: int
    remaining: tuple[int, ...]
    score_a: int = 0
    score_b: int = 0
    player: str = "A"

    @property
    def utility(self):
        return self.score_a - self.score_b


@dataclass
class SearchStats:
    visited_nodes: int = 0
    evaluated_leaves: int = 0
    cutoffs: int = 0


@dataclass
class Decision:
    target: int | None
    value: int


def terrain_cost(cell):
    if cell == "X":
        return None
    if cell in ("A", "B") or str(cell).startswith("T"):
        return 1
    return int(cell)


def astar(start, end, grid):
    rows, cols = len(grid), len(grid[0])
    costs = [[terrain_cost(cell) for cell in row] for row in grid]
    min_cost = min(cost for row in costs for cost in row if cost is not None)

    def estimate(position):
        return (abs(position[0] - end[0]) + abs(position[1] - end[1])) * min_cost

    serial = count()
    h = estimate(start)
    frontier = [(h, h, next(serial), 0, start)]
    g_score = {start: 0}
    while frontier:
        _, _, _, current_g, current = heapq.heappop(frontier)
        if current_g != g_score[current]:
            continue
        if current == end:
            return current_g
        for dr, dc in DIRECTIONS:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or costs[nr][nc] is None:
                continue
            neighbor = (nr, nc)
            candidate_g = current_g + costs[nr][nc]
            if candidate_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = candidate_g
                h = estimate(neighbor)
                heapq.heappush(
                    frontier, (candidate_g + h, h, next(serial), candidate_g, neighbor)
                )
    return float("inf")


def build_distance_matrix(grid, positions):
    return [[astar(start, end, grid) for end in positions] for start in positions]


def move(state, target, distances, values):
    remaining = tuple(t for t in state.remaining if t != target)
    if state.player == "A":
        gain = values[target] - distances[state.position_a][target]
        return GameState(target, state.position_b, remaining,
                         state.score_a + gain, state.score_b, "B")
    gain = values[target] - distances[state.position_b][target]
    return GameState(state.position_a, target, remaining,
                     state.score_a, state.score_b + gain, "A")


def ordered_targets(state, distances, values, order):
    targets = list(state.remaining)
    if order == "reverse":
        targets.reverse()
    elif order == "gain":
        origin = state.position_a if state.player == "A" else state.position_b
        targets.sort(key=lambda t: values[t] - distances[origin][t], reverse=True)
    return targets


def minimax(state, distances, values, depth, *, stats=None, order="fixed"):
    if stats is None:
        stats = SearchStats()
    stats.visited_nodes += 1
    if depth == 0 or not state.remaining:
        stats.evaluated_leaves += 1
        return Decision(None, state.utility)

    maximizing = state.player == "A"
    best = Decision(None, -float("inf") if maximizing else float("inf"))
    for target in ordered_targets(state, distances, values, order):
        child = move(state, target, distances, values)
        value = minimax(child, distances, values, depth - 1, stats=stats, order=order).value
        better = value > best.value if maximizing else value < best.value
        if better:
            best = Decision(target, value)
    return best


def alpha_beta(
    state, distances, values, depth, alpha=-float("inf"), beta=float("inf"),
    *, stats=None, order="fixed",
):
    if stats is None:
        stats = SearchStats()
    stats.visited_nodes += 1
    if depth == 0 or not state.remaining:
        stats.evaluated_leaves += 1
        return Decision(None, state.utility)

    maximizing = state.player == "A"
    best = Decision(None, -float("inf") if maximizing else float("inf"))
    targets = ordered_targets(state, distances, values, order)
    for index, target in enumerate(targets):
        child = move(state, target, distances, values)
        value = alpha_beta(
            child, distances, values, depth - 1, alpha, beta, stats=stats, order=order,
        ).value
        better = value > best.value if maximizing else value < best.value
        if better:
            best = Decision(target, value)
        if maximizing:
            alpha = max(alpha, best.value)
        else:
            beta = min(beta, best.value)
        if alpha >= beta and index + 1 < len(targets):
            stats.cutoffs += 1
            break
    return best


def read_problem(data):
    treasures = data["treasures"]
    names = ["A", "B"] + [t["name"] for t in treasures]
    positions = [tuple(data["starts"][p]) for p in ("A", "B")]
    positions += [tuple(t["position"]) for t in treasures]
    values = [0, 0] + [t["value"] for t in treasures]
    return data["grid"], names, positions, values


def print_matrix(distances, names):
    print("distance_matrix:")
    print("    " + "".join(f"{name:>5}" for name in names))
    for name, row in zip(names, distances):
        print(f"{name:<4}" + "".join(f"{value:>5}" for value in row))


def show_line(state, distances, values, names, depth, order):
    print("root_action_values:")
    for target in state.remaining:
        child = move(state, target, distances, values)
        value = minimax(child, distances, values, depth - 1).value
        print(f"  {names[target]}: {value}")
    # 用完整 Minimax 重建最优应对，避免将剪枝回传的界当作精确分支值。
    for ply in range(1, depth + 1):
        if not state.remaining:
            break
        target = minimax(state, distances, values, depth - ply + 1, order=order).target
        origin = state.position_a if state.player == "A" else state.position_b
        distance = distances[origin][target]
        gain = values[target] - distance
        child = move(state, target, distances, values)
        print(
            f"  {ply}. {state.player} -> {names[target]}: distance={distance}, gain={gain}, "
            f"score_A={child.score_a}, score_B={child.score_b}, U={child.utility}"
        )
        state = child


def compare(state, distances, values, names, depths):
    print("algorithm   H  order    target  value  visited  leaves  cutoffs  median_ms")
    for depth in depths:
        for algorithm in (minimax, alpha_beta):
            for order in ("fixed", "reverse", "gain"):
                timings = []
                for _ in range(101):
                    stats = SearchStats()
                    begin = perf_counter()
                    result = algorithm(state, distances, values, depth, stats=stats, order=order)
                    timings.append((perf_counter() - begin) * 1000)
                print(
                    f"{algorithm.__name__:<11} {depth:>1}  {order:<7} "
                    f"{names[result.target]:>6} {result.value:>6} {stats.visited_nodes:>8} "
                    f"{stats.evaluated_leaves:>7} {stats.cutoffs:>8} {median(timings):>10.4f}"
                )


def main():
    parser = argparse.ArgumentParser(description="题目四：竞争环境下的宝藏分配")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path)
    source.add_argument("--stdin", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--matrix", action="store_true", help="只输出距离矩阵")
    mode.add_argument("--trace", action="store_true", help="展示最优应对与收益累计")
    mode.add_argument("--compare", action="store_true", help="比较算法、深度和访问顺序")
    parser.add_argument("--depth", type=int, choices=(2, 4))
    parser.add_argument("--algorithm", choices=("minimax", "alpha_beta"), default="alpha_beta")
    parser.add_argument("--order", choices=("fixed", "reverse", "gain"), default="fixed")
    args = parser.parse_args()

    if args.input:
        data = json.loads(args.input.read_text(encoding="utf-8-sig"))
    elif args.stdin:
        data = json.load(sys.stdin)
    else:
        data = SAMPLE_INPUT
    grid, names, positions, values = read_problem(data)
    distances = build_distance_matrix(grid, positions)
    state = GameState(0, 1, tuple(range(2, len(names))))
    depths = [args.depth] if args.depth is not None else [2, 4]
    print_matrix(distances, names)
    if args.matrix:
        return
    if args.compare:
        compare(state, distances, values, names, depths)
        return

    algorithm = minimax if args.algorithm == "minimax" else alpha_beta
    for depth in depths:
        result = algorithm(state, distances, values, depth, order=args.order)
        output = {"best_target": names[result.target], "best_value": result.value}
        print(f"H={depth}: {json.dumps(output, ensure_ascii=False)}")
        if args.trace:
            show_line(state, distances, values, names, depth, args.order)


if __name__ == "__main__":
    main()
