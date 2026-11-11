import argparse
from collections import deque
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import heapq
from itertools import accumulate, permutations
import json
import os
from pathlib import Path
import platform
from random import Random
from statistics import median
from time import perf_counter

from q1 import maze
from q2 import weighted_astar as weighted
from q3 import game_tree as tree
from q4 import treasure_game as treasure


ROOT = Path(__file__).resolve().parent
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def measure(solve, repeats):
    solve()
    times = []
    for _ in range(repeats):
        begin = perf_counter()
        result = solve()
        times.append((perf_counter() - begin) * 1000)
    return result, {"median_ms": median(times), "times_ms": times}


def neighbors(position, grid):
    for dr, dc in DIRECTIONS:
        r, c = position[0] + dr, position[1] + dc
        if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] != "X":
            yield r, c


def bfs_distance(start, end, grid):
    pending = deque([(start, 0)])
    seen = {start}
    while pending:
        position, distance = pending.popleft()
        if position == end:
            return distance
        for neighbor in neighbors(position, grid):
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append((neighbor, distance + 1))
    return -1


def dijkstra_distances(start, grid, entry_cost, reverse=False):
    distances = {start: 0}
    pending = [(0, start)]
    while pending:
        cost, position = heapq.heappop(pending)
        if cost != distances[position]:
            continue
        for neighbor in neighbors(position, grid):
            # 反向搜索沿原图的入边走，边权仍为进入原终点的代价。
            r, c = position if reverse else neighbor
            candidate = cost + entry_cost(grid[r][c])
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                heapq.heappush(pending, (candidate, neighbor))
    return distances


def check_path(path, start, end, grid, expected):
    assert bool(path) == (expected >= 0)
    if path:
        assert path[0] == start and path[-1] == end
        assert all(0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] != "X"
                   for r, c in path)
        assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                   for a, b in zip(path, path[1:]))


def maze_cases(seed, trials):
    yield {"name": "sample", "start": maze.SAMPLE_START, "end": maze.SAMPLE_END,
           "grid": maze.SAMPLE_GRID}
    yield {"name": "unreachable", "start": (0, 0), "end": (0, 2),
           "grid": [["S", "X", "E"]]}
    yield {"name": "same_endpoint", "start": (0, 0), "end": (0, 0), "grid": [["S"]]}
    for size in (20, 40, 80):
        for trial in range(trials):
            case_seed = seed + size * 100 + trial
            rng = Random(case_seed)
            noise = [[rng.random() for _ in range(size)] for _ in range(size)]
            for density in (0.0, 0.15, 0.30):
                grid = [["X" if value < density else "." for value in row] for row in noise]
                grid[0][0], grid[-1][-1] = "S", "E"
                yield {
                    "name": f"maze_{size}_p{density:.2f}_trial{trial}",
                    "start": (0, 0), "end": (size - 1, size - 1), "grid": grid,
                    "seed": case_seed, "trial": trial, "size": size,
                    "obstacle_probability": density,
                    "actual_obstacle_density": sum(row.count("X") for row in grid) / size**2,
                }


def run_maze(seed, trials, repeats):
    cases = []
    for case in maze_cases(seed, trials):
        start, end, grid = case["start"], case["end"], case["grid"]
        expected = bfs_distance(start, end, grid)
        case["reference_steps"] = expected
        case["reachable"] = expected >= 0
        case["runs"] = []
        for heuristic in ("zero", "manhattan"):
            def solve():
                stats = maze.SearchStats()
                return maze.astar(start, end, grid, heuristic, stats=stats), stats

            (path, stats), timing = measure(solve, repeats)
            check_path(path, start, end, grid, expected)
            steps = len(path) - 1 if path else -1
            assert steps == expected
            case["runs"].append({"heuristic": heuristic, "steps": steps, "path": path,
                                 **asdict(stats), **timing})
        cases.append(case)
    return {"cases": cases}


def weighted_cases(seed, trials):
    yield {"name": "sample", "start": weighted.SAMPLE_START, "end": weighted.SAMPLE_END,
           "grid": weighted.SAMPLE_GRID}
    for name, grid in [
        ("free_terminal", [["S", "E"]]),
        ("zero_terrain", [["S", 0, 0, "E"]]),
        ("unreachable", [["S", "X", "E"]]),
    ]:
        yield {"name": name, "start": (0, 0), "end": (0, len(grid[0]) - 1), "grid": grid}
    for weight in (1, 3, 9):
        yield {"name": f"detour_weight_{weight}", "start": (0, 0), "end": (0, 3),
               "grid": [["S", weight, weight, "E"], [1, 1, 1, 1]]}
    for size in (20, 40):
        for trial in range(trials):
            case_seed = seed + 20000 + size * 100 + trial
            rng = Random(case_seed)
            grid = [["X" if rng.random() < 0.15 else rng.randint(1, 9)
                     for _ in range(size)] for _ in range(size)]
            grid[0][0], grid[-1][-1] = "S", "E"
            yield {"name": f"weighted_{size}_trial{trial}", "start": (0, 0),
                   "end": (size - 1, size - 1), "grid": grid, "seed": case_seed,
                   "trial": trial, "size": size, "obstacle_probability": 0.15}


def heuristic_audit(grid, end):
    distances = dijkstra_distances(end, grid, weighted.terrain_cost, reverse=True)
    min_cost = min((weighted.terrain_cost(cell)
                    for r, row in enumerate(grid) for c, cell in enumerate(row)
                    if cell != "X" and (r, c) != end), default=0)
    overestimates = []
    for position, exact in distances.items():
        raw = abs(position[0] - end[0]) + abs(position[1] - end[1])
        corrected = max(raw - 1, 0) * min_cost
        assert corrected <= exact
        if raw > exact:
            overestimates.append({"position": position, "raw_h": raw,
                                  "corrected_h": corrected, "exact_remaining_cost": exact})
    return {"reachable_positions_checked": len(distances), "min_step_cost": min_cost,
            "raw_overestimate_count": len(overestimates), "corrected_overestimate_count": 0,
            "raw_overestimate_examples": overestimates[:3]}


def run_weighted(seed, trials, repeats):
    cases = []
    for case in weighted_cases(seed, trials):
        start, end, grid = case["start"], case["end"], case["grid"]
        expected = dijkstra_distances(start, grid, weighted.terrain_cost).get(end, -1)
        case["reference_cost"] = expected
        case["heuristic_audit"] = heuristic_audit(grid, end)
        case["runs"] = []
        for heuristic in ("zero", "manhattan"):
            result, timing = measure(lambda: weighted.search(start, end, grid, heuristic), repeats)
            assert result.cost == expected
            check_path(result.path, start, end, grid, expected)
            entries = [weighted.terrain_cost(grid[r][c]) for r, c in result.path[1:]]
            if result.path:
                assert sum(entries) == result.cost
            case["runs"].append({"heuristic": heuristic, **asdict(result),
                                 "steps": len(result.path) - 1 if result.path else -1,
                                 "entry_costs": entries,
                                 "cumulative_costs": list(accumulate(entries, initial=0)) if result.path else [],
                                 **timing})
        cases.append(case)
    return {"cases": cases}


def random_tree(rng, branching, depth):
    if depth == 0:
        return {"value": rng.randint(-20, 20)}
    return {"children": [random_tree(rng, branching, depth - 1) for _ in range(branching)]}


def reference_tree(data, maximizing=True):
    if not data.get("children"):
        return data["value"]
    values = [reference_tree(child, not maximizing) for child in data["children"]]
    return (max if maximizing else min)(values)


def tree_cases(seed, trials):
    for name, filename in (("sample", "sample_input.json"), ("order_demo", "order_input.json")):
        data = json.loads((ROOT / "q3" / filename).read_text(encoding="utf-8"))
        yield {"name": name, "tree_structure": data["tree_structure"]}
    for index, (branching, depth) in enumerate(((2, 6), (3, 4), (3, 6))):
        for trial in range(trials):
            case_seed = seed + 30000 + index * 100 + trial
            yield {"name": f"tree_b{branching}_d{depth}_trial{trial}", "seed": case_seed,
                   "trial": trial, "branching_factor": branching, "depth": depth,
                   "tree_structure": random_tree(Random(case_seed), branching, depth)}


def run_trees(seed, trials, repeats):
    cases = []
    for case in tree_cases(seed, trials):
        root = tree.build_tree(case["tree_structure"])
        expected = reference_tree(case["tree_structure"])
        case["reference_value"] = expected
        case["runs"] = []
        for order in ("left", "reverse"):
            for algorithm in (tree.minimax, tree.alpha_beta):
                def solve():
                    stats = tree.SearchStats()
                    return algorithm(root, stats=stats, order=order), stats

                (value, stats), timing = measure(solve, repeats)
                assert value == expected
                case["runs"].append({"algorithm": algorithm.__name__, "order": order,
                                     "value": value, **asdict(stats), **timing})
            full, pruned = case["runs"][-2:]
            assert pruned["visited_nodes"] <= full["visited_nodes"]
        cases.append(case)
    return {"cases": cases}


def exhaustive_actions(distances, values, depth):
    outcomes = {}
    for actions in permutations(range(2, len(values)), depth):
        positions, scores = [0, 1], [0, 0]
        for ply, target in enumerate(actions):
            player = ply % 2
            scores[player] += values[target] - distances[positions[player]][target]
            positions[player] = target
        outcomes[actions] = scores[0] - scores[1]
    for level in range(depth - 1, 0, -1):
        grouped = {}
        for actions, value in outcomes.items():
            grouped.setdefault(actions[:level], []).append(value)
        outcomes = {prefix: (max if level % 2 == 0 else min)(scores)
                    for prefix, scores in grouped.items()}
    return {actions[0]: value for actions, value in outcomes.items()}


def treasure_line(state, distances, values, names, depth):
    line = []
    for ply in range(depth):
        target = treasure.minimax(state, distances, values, depth - ply).target
        origin = state.position_a if state.player == "A" else state.position_b
        distance = distances[origin][target]
        child = treasure.move(state, target, distances, values)
        line.append({"player": state.player, "target": names[target], "distance": distance,
                     "gain": values[target] - distance, "score_a": child.score_a,
                     "score_b": child.score_b, "utility": child.utility})
        state = child
    return line


def run_treasure(seed, trials, repeats):
    grid, names, positions, original_values = treasure.read_problem(treasure.SAMPLE_INPUT)
    begin = perf_counter()
    distances = treasure.build_distance_matrix(grid, positions)
    matrix_ms = (perf_counter() - begin) * 1000
    expected_matrix = [[dijkstra_distances(p, grid, treasure.terrain_cost)[q]
                        for q in positions] for p in positions]
    assert distances == expected_matrix
    value_cases = [("sample", original_values), ("equal_values", [0, 0, 7, 7, 7, 7]),
                   ("high_T1", [0, 0, 20, 6, 7, 7]), ("high_T4", [0, 0, 9, 6, 7, 20])]
    for trial in range(trials):
        rng = Random(seed + 40000 + trial)
        value_cases.append((f"random_values_trial{trial}", [0, 0] + [rng.randint(0, 20) for _ in range(4)]))
    state = treasure.GameState(0, 1, (2, 3, 4, 5))
    cases = []
    for name, values in value_cases:
        case = {"name": name, "values": dict(zip(names[2:], values[2:])), "depths": []}
        for depth in (2, 4):
            exact = exhaustive_actions(distances, values, depth)
            expected = max(exact.values())
            optimal = [target for target, value in exact.items() if value == expected]
            detail = {"depth": depth, "reference_value": expected,
                      "optimal_targets": [names[t] for t in optimal],
                      "root_action_values": {names[t]: value for t, value in sorted(exact.items())},
                      "optimal_line": treasure_line(state, distances, values, names, depth), "runs": []}
            for order in ("fixed", "reverse", "gain"):
                for algorithm in (treasure.minimax, treasure.alpha_beta):
                    def solve():
                        stats = treasure.SearchStats()
                        return algorithm(state, distances, values, depth, stats=stats, order=order), stats

                    (result, stats), timing = measure(solve, repeats)
                    assert result.value == expected and result.target in optimal
                    detail["runs"].append({"algorithm": algorithm.__name__, "order": order,
                                           "best_target": names[result.target], "value": result.value,
                                           **asdict(stats), **timing})
                full, pruned = detail["runs"][-2:]
                assert pruned["visited_nodes"] <= full["visited_nodes"]
            if name == "sample":
                sample_answer = (["T2"], -1) if depth == 2 else (["T3"], 3)
                assert (detail["optimal_targets"], expected) == sample_answer
            case["depths"].append(detail)
        cases.append(case)
    return {"input": treasure.SAMPLE_INPUT, "names": names, "distance_matrix": distances,
            "distance_matrix_ms": matrix_ms, "cases": cases}


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def summary(data):
    meta = data["metadata"]
    sections = ["# 实验二批量实验结果", f"运行时间：{meta['started_at']}；Python {meta['python']}。",
                f"随机种子 {meta['seed']}；每种配置预热 1 次，计时 {meta['repeats']} 次，使用中位数。",
                "全部配置已通过参考结果校验。耗时单位为毫秒，包含求解及统计，不含输入构造、"
                "参考算法校验和文件读写。第四题的博弈耗时不包含距离矩阵预计算。",
                "完整输入、每次计时、路径及校验结果见 experiments.json。",
                "## 一：启发函数、地图大小与障碍密度",
                "相同尺寸和 trial 的地图共享随机数阵，用不同阈值设置障碍。没有强制保通或丢弃无解地图；"
                "无解时两种启发函数都需搜索起点可达区域。表中 p 为障碍生成概率，实际密度另存于 JSON。"]
    rows = []
    for case in data["q1"]["cases"]:
        zero, guided = case["runs"]
        rows.append([case["name"], case.get("obstacle_probability", "—"), case["reference_steps"],
                     zero["expanded_nodes"], guided["expanded_nodes"],
                     f"{zero['median_ms']:.4f}", f"{guided['median_ms']:.4f}"])
    sections.append(table(["输入", "p", "最短步数", "h=0 扩展", "曼哈顿扩展", "h=0 ms", "曼哈顿 ms"], rows))
    sections += ["## 二：加权地形与启发函数可采纳性",
                 "Dijkstra 校验最小代价，反向 Dijkstra 给出各位置到 E 的真实剩余代价。"
                 "原始 h 超估数是启发函数审计结果，未运行原始曼哈顿启发的 A*；不能据此声称它在这些输入上返回了错误答案。"]
    rows = []
    for case in data["q2"]["cases"]:
        zero, guided = case["runs"]
        rows.append([case["name"], case["reference_cost"], guided["steps"],
                     zero["expanded_nodes"], guided["expanded_nodes"],
                     case["heuristic_audit"]["raw_overestimate_count"],
                     f"{zero['median_ms']:.4f}", f"{guided['median_ms']:.4f}"])
    sections.append(table(["输入", "最小代价", "修正 h 路径步数", "h=0 扩展", "修正 h 扩展",
                           "原始 h 超估数", "h=0 ms", "修正 h ms"], rows))
    sections += ["## 三：博弈树访问顺序与剪枝", "visited 包含内部节点和叶子；剪枝次数只计实际跳过剩余子节点的截断。"]
    rows = []
    for case in data["q3"]["cases"]:
        for index in (0, 2):
            full, pruned = case["runs"][index:index + 2]
            rows.append([case["name"], full["order"], full["value"], full["visited_nodes"],
                         pruned["visited_nodes"], pruned["cutoffs"],
                         f"{full['median_ms']:.4f}", f"{pruned['median_ms']:.4f}"])
    sections.append(table(["输入", "顺序", "最优值", "Minimax 节点", "Alpha-beta 节点", "剪枝次数",
                           "Minimax ms", "Alpha-beta ms"], rows))
    sections += ["## 四：宝藏价值、搜索深度与行动排序",
                 "保持地图不变，只调整宝藏价值。各首步的精确值由行动序列穷举得到；同分首步均视为正确。",
                 table(["起点/终点"] + data["q4"]["names"],
                       [[name] + row for name, row in zip(data["q4"]["names"], data["q4"]["distance_matrix"])])]
    rows = []
    for case in data["q4"]["cases"]:
        for detail in case["depths"]:
            runs = detail["runs"]
            rows.append([case["name"], detail["depth"], ", ".join(detail["optimal_targets"]),
                         detail["reference_value"], runs[0]["visited_nodes"],
                         runs[1]["visited_nodes"], runs[3]["visited_nodes"], runs[5]["visited_nodes"]])
    sections.append(table(["价值设置", "H", "最优首步", "收益差", "Minimax 节点",
                           "AB fixed", "AB reverse", "AB gain"], rows))
    sections += ["## 指标解释", "最短步数或最小代价为 -1 表示无解。节点数由输入和访问规则决定，"
                 "耗时会受机器状态影响；减少节点不保证在很小的样例上更快。"
                 "即时收益排序只是访问顺序，最优值仍由完整深度的博弈搜索确定。"]
    return "\n\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description="实验二统一对照实验")
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--trials", type=int, default=3, help="每组随机输入的数量")
    parser.add_argument("--repeats", type=int, default=31, help="每种配置的计时次数")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.trials < 1 or args.repeats < 1:
        parser.error("trials 和 repeats 必须为正整数")
    data = {"metadata": {
        "started_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "logical_processors": os.cpu_count(),
        "seed": args.seed, "trials": args.trials, "repeats": args.repeats, "warmup_runs": 1,
    }}
    for key, run in (("q1", run_maze), ("q2", run_weighted), ("q3", run_trees), ("q4", run_treasure)):
        begin = perf_counter()
        data[key] = run(args.seed, args.trials, args.repeats)
        print(f"{key}: {len(data[key]['cases'])} cases checked ({perf_counter() - begin:.2f}s)", flush=True)
    data["metadata"]["validation_passed"] = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_file = args.output_dir / "experiments.json"
    summary_file = args.output_dir / "summary.md"
    raw_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_file.write_text(summary(data), encoding="utf-8")
    print(raw_file.resolve())
    print(summary_file.resolve())


if __name__ == "__main__":
    main()
