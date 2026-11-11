import argparse
import ast
from dataclasses import dataclass
import heapq
from itertools import accumulate, count
from pathlib import Path
from statistics import median
import sys
from time import perf_counter


Position = tuple[int, int]
SAMPLE_START = (0, 0)
SAMPLE_END = (4, 4)
SAMPLE_GRID = [
    ["S", 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, "E"],
]
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


@dataclass
class SearchResult:
    cost: int
    path: list[Position]
    expanded_nodes: int


def terrain_cost(cell: int | str) -> int | None:
    if cell == "X":
        return None
    if cell == "S":
        return 1
    if cell == "E":
        return 0
    return int(cell)


def search(start, end, grid, heuristic="manhattan") -> SearchResult:
    """地形代价非负；进入 E 的代价为 0，起点不计费。"""
    rows, cols = len(grid), len(grid[0])
    costs = [[terrain_cost(cell) for cell in row] for row in grid]
    if costs[start[0]][start[1]] is None or costs[end[0]][end[1]] is None:
        return SearchResult(-1, [], 0)
    min_cost = min(
        (cost for r, row in enumerate(costs) for c, cost in enumerate(row)
         if cost is not None and (r, c) != end),
        default=0,
    )

    def estimate(position):
        if heuristic == "zero":
            return 0
        distance = abs(position[0] - end[0]) + abs(position[1] - end[1])
        # 最后一步进入 E 免费，只对其余至少 distance-1 步估计代价。
        return max(distance - 1, 0) * min_cost

    serial = count()
    h = estimate(start)
    frontier = [(h, h, next(serial), 0, start)]
    g_score = {start: 0}
    parent = {}
    expanded = 0

    while frontier:
        _, _, _, current_g, current = heapq.heappop(frontier)
        if current_g != g_score[current]:
            continue
        if current == end:
            path = [end]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return SearchResult(current_g, path[::-1], expanded)

        expanded += 1
        for dr, dc in DIRECTIONS:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            step_cost = costs[nr][nc]
            if step_cost is None:
                continue
            neighbor = (nr, nc)
            candidate_g = current_g + step_cost
            if candidate_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = candidate_g
                parent[neighbor] = current
                h = estimate(neighbor)
                heapq.heappush(
                    frontier, (candidate_g + h, h, next(serial), candidate_g, neighbor)
                )

    return SearchResult(-1, [], expanded)


def astar(start, end, grid) -> int:
    return search(start, end, grid).cost


def read_problem(text):
    lines = text.strip().splitlines()
    start = tuple(ast.literal_eval(lines[0]))
    end = tuple(ast.literal_eval(lines[1]))
    grid = ast.literal_eval("\n".join(lines[2:]))
    return start, end, grid


def compare(start, end, grid):
    print("heuristic   cost  expanded  median_ms")
    for heuristic in ("zero", "manhattan"):
        timings = []
        for _ in range(101):
            begin = perf_counter()
            result = search(start, end, grid, heuristic)
            timings.append((perf_counter() - begin) * 1000)
        print(
            f"{heuristic:<10} {result.cost:>5} {result.expanded_nodes:>9} "
            f"{median(timings):>10.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="题目二：加权地形 A* 搜索")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path)
    source.add_argument("--stdin", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--trace", action="store_true", help="展示路径与代价累计")
    mode.add_argument("--compare", action="store_true", help="对照两种启发函数")
    args = parser.parse_args()

    if args.input:
        problem = read_problem(args.input.read_text(encoding="utf-8-sig"))
    elif args.stdin:
        problem = read_problem(sys.stdin.read())
    else:
        problem = SAMPLE_START, SAMPLE_END, SAMPLE_GRID

    if args.compare:
        compare(*problem)
        return
    result = search(*problem)
    print(result.cost)
    if args.trace:
        grid = problem[2]
        entry_costs = [terrain_cost(grid[r][c]) for r, c in result.path[1:]]
        print("path:", result.path)
        print("entry_costs:", entry_costs)
        print("cumulative_costs:", list(accumulate(entry_costs, initial=0)) if result.path else [])
        print("expanded_nodes:", result.expanded_nodes)


if __name__ == "__main__":
    main()
