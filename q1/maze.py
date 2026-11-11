import argparse
import ast
from dataclasses import dataclass
import heapq
from itertools import count
from pathlib import Path
from statistics import median
import sys
from time import perf_counter


Position = tuple[int, int]

SAMPLE_START = (0, 0)
SAMPLE_END = (3, 4)
SAMPLE_GRID = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))

@dataclass
class SearchStats:
    expanded_nodes: int = 0  # 实际展开邻居的节点数，不包含终点
    visited_nodes: int = 0   # 从堆中取出的有效节点数，包含终点
    discovered_nodes: int = 0  # 已发现的不同坐标数，包含起点


def astar(
    start: Position,
    end: Position,
    grid: list[list[str]],
    heuristic: str = "manhattan",
    *,
    stats: SearchStats | None = None,
) -> list[Position]:
    """返回包含起终点的最短路径；空地图、端点无效或不可达时返回 []。

    heuristic 可选 manhattan 或 zero。后者令 h=0，作为对照基线。
    地图必须为矩形；X 不可通行，其余格子的进入代价均为 1。
    """
    if heuristic not in ("manhattan", "zero"):
        raise ValueError("heuristic 必须是 manhattan 或 zero")
    if stats is None:
        stats = SearchStats()
    stats.expanded_nodes = stats.visited_nodes = stats.discovered_nodes = 0

    if not grid:
        return []
    rows, cols = len(grid), len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("地图必须为矩形，每行长度应相同")
    if cols == 0:
        return []

    def passable(position: Position) -> bool:
        row, col = position
        return 0 <= row < rows and 0 <= col < cols and grid[row][col] != "X"

    if not passable(start) or not passable(end):
        return []

    def estimate(position: Position) -> int:
        if heuristic == "zero":
            return 0
        return abs(position[0] - end[0]) + abs(position[1] - end[1])

    # 条目为 (f, h, 序号, 入队时的 g, 坐标)。同 f 时优先较小 h，
    # 再按入队顺序处理；这些规则只影响搜索顺序，不改变最优代价。
    serial = count()
    initial_h = estimate(start)
    frontier = [(initial_h, initial_h, next(serial), 0, start)]
    g_score = {start: 0}
    parent: dict[Position, Position] = {}
    stats.discovered_nodes = 1

    while frontier:
        _, _, _, current_g, current = heapq.heappop(frontier)
        # 同一坐标可能被多次入队，忽略已被更低代价替代的旧条目。
        if current_g != g_score[current]:
            continue
        stats.visited_nodes += 1

        # 在终点出队时结束，而非首次发现终点时结束。
        if current == end:
            path = [end]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return path

        stats.expanded_nodes += 1
        for dr, dc in DIRECTIONS:
            neighbor = (current[0] + dr, current[1] + dc)
            if not passable(neighbor):
                continue
            candidate_g = current_g + 1
            if candidate_g >= g_score.get(neighbor, float("inf")):
                continue
            if neighbor not in g_score:
                stats.discovered_nodes += 1
            g_score[neighbor] = candidate_g
            parent[neighbor] = current
            h = estimate(neighbor)
            heapq.heappush(
                frontier, (candidate_g + h, h, next(serial), candidate_g, neighbor)
            )

    return []


def read_problem(text: str) -> tuple[Position, Position, list[list[str]]]:
    """读取题目格式：前两行分别为起终点，后续内容为二维列表。"""
    lines = text.strip().splitlines()
    if len(lines) < 3:
        raise ValueError("输入需要起点、终点两行，以及后续的地图列表")
    start = ast.literal_eval(lines[0])
    end = ast.literal_eval(lines[1])
    grid = ast.literal_eval("\n".join(lines[2:]))
    for position in (start, end):
        if (
            not isinstance(position, (tuple, list))
            or len(position) != 2
            or any(type(value) is not int for value in position)
        ):
            raise ValueError("起终点必须是包含两个整数的坐标")
    if not isinstance(grid, list) or any(not isinstance(row, list) for row in grid):
        raise ValueError("地图必须为二维列表")
    if any(not isinstance(cell, str) for row in grid for cell in row):
        raise ValueError("地图格子必须使用字符串，例如 'S'、'.'、'X'、'E'")
    return tuple(start), tuple(end), grid


def compare(start: Position, end: Position, grid: list[list[str]]) -> None:
    """固定输入和优先队列规则，只改变 h；耗时取 101 次运行中位数。"""
    print("heuristic  steps  expanded  visited  discovered  median_ms")
    for heuristic in ("zero", "manhattan"):
        stats = SearchStats()
        timings = []
        for _ in range(101):
            begin = perf_counter()
            path = astar(start, end, grid, heuristic, stats=stats)
            timings.append((perf_counter() - begin) * 1000)
        steps = len(path) - 1 if path else -1
        print(
            f"{heuristic:<10} {steps:>5} {stats.expanded_nodes:>9} "
            f"{stats.visited_nodes:>8} {stats.discovered_nodes:>11} "
            f"{median(timings):>10.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="读取 UTF-8 格式的输入文件")
    source.add_argument("--stdin", action="store_true", help="从标准输入读取题目")
    parser.add_argument("--compare", action="store_true", help="比较 h=0 与曼哈顿距离")
    args = parser.parse_args()

    try:
        if args.input:
            problem = read_problem(args.input.read_text(encoding="utf-8-sig"))
        elif args.stdin:
            problem = read_problem(sys.stdin.read())
        else:
            problem = (SAMPLE_START, SAMPLE_END, SAMPLE_GRID)
        if args.compare:
            compare(*problem)
        else:
            print(astar(*problem))
    except (ValueError, SyntaxError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
