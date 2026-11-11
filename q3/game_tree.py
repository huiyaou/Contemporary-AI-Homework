import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter


class TreeNode:
    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children if children is not None else []


@dataclass
class SearchStats:
    visited_nodes: int = 0
    evaluated_leaves: int = 0
    cutoffs: int = 0


def ordered_children(node, order):
    children = list(enumerate(node.children))
    return children if order == "left" else children[::-1]


def minimax(node, maximizing=True, *, stats=None, order="left"):
    if stats is None:
        stats = SearchStats()
    stats.visited_nodes += 1
    if not node.children:
        stats.evaluated_leaves += 1
        return node.value

    value = -float("inf") if maximizing else float("inf")
    for _, child in ordered_children(node, order):
        child_value = minimax(child, not maximizing, stats=stats, order=order)
        value = max(value, child_value) if maximizing else min(value, child_value)
    return value


def alpha_beta(
    node, alpha=-float("inf"), beta=float("inf"), maximizing=True,
    *, stats=None, order="left", trace=None, path="root",
):
    if stats is None:
        stats = SearchStats()
    stats.visited_nodes += 1
    if not node.children:
        stats.evaluated_leaves += 1
        if trace is not None:
            trace.append(f"{path}: leaf={node.value}")
        return node.value

    value = -float("inf") if maximizing else float("inf")
    children = ordered_children(node, order)
    role = "MAX" if maximizing else "MIN"
    for position, (index, child) in enumerate(children):
        child_value = alpha_beta(
            child, alpha, beta, not maximizing,
            stats=stats, order=order, trace=trace, path=f"{path}.{index}",
        )
        if maximizing:
            value = max(value, child_value)
            alpha = max(alpha, value)
        else:
            value = min(value, child_value)
            beta = min(beta, value)

        # 仅在确实跳过剩余子节点时计为一次剪枝。
        if alpha >= beta and position + 1 < len(children):
            stats.cutoffs += 1
            if trace is not None:
                skipped = [f"{path}.{i}" for i, _ in children[position + 1:]]
                trace.append(
                    f"{path} {role}: alpha={alpha}, beta={beta}, skip={skipped}"
                )
            break

    if trace is not None:
        trace.append(f"{path} {role}: return={value}")
    return value


def build_tree(data):
    return TreeNode(
        value=data.get("value"),
        children=[build_tree(child) for child in data.get("children", [])],
    )


def sample_tree():
    max1 = TreeNode(children=[TreeNode(3), TreeNode(5)])
    max2 = TreeNode(children=[TreeNode(6), TreeNode(9)])
    max3 = TreeNode(children=[TreeNode(1), TreeNode(2)])
    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])
    return TreeNode(children=[min1, min2])


def compare(root):
    print("algorithm   order     value  visited  leaves  cutoffs  median_ms")
    for algorithm in (minimax, alpha_beta):
        for order in ("left", "reverse"):
            timings = []
            for _ in range(101):
                stats = SearchStats()
                begin = perf_counter()
                value = algorithm(root, stats=stats, order=order)
                timings.append((perf_counter() - begin) * 1000)
            print(
                f"{algorithm.__name__:<11} {order:<8} {value:>6} "
                f"{stats.visited_nodes:>8} {stats.evaluated_leaves:>7} "
                f"{stats.cutoffs:>8} {median(timings):>10.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description="题目三：极大极小博弈树")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path)
    source.add_argument("--stdin", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--trace", action="store_true", help="展示回传与剪枝过程")
    mode.add_argument("--compare", action="store_true", help="比较算法与访问顺序")
    parser.add_argument("--algorithm", choices=("minimax", "alpha_beta"), default="alpha_beta")
    parser.add_argument("--order", choices=("left", "reverse"), default="left")
    args = parser.parse_args()

    if args.input:
        root = build_tree(json.loads(args.input.read_text(encoding="utf-8-sig"))["tree_structure"])
    elif args.stdin:
        root = build_tree(json.load(sys.stdin)["tree_structure"])
    else:
        root = sample_tree()

    if args.compare:
        compare(root)
    elif args.trace:
        trace = []
        stats = SearchStats()
        print(alpha_beta(root, stats=stats, order=args.order, trace=trace))
        print("\n".join(trace))
        print("visited_nodes:", stats.visited_nodes)
        print("evaluated_leaves:", stats.evaluated_leaves)
        print("cutoffs:", stats.cutoffs)
    else:
        algorithm = minimax if args.algorithm == "minimax" else alpha_beta
        print(algorithm(root, order=args.order))


if __name__ == "__main__":
    main()
