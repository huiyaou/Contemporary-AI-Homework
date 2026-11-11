# 搜索算法

使用 Python 3.10 或更高版本。算法与实验脚本仅依赖标准库，配图脚本使用 NumPy 和 Matplotlib。

## 代码与结果文件

| 文件 | 内容 |
| --- | --- |
| `q1/maze.py` | 迷宫 A* 与路径恢复 |
| `q2/weighted_astar.py` | 加权 A* 与代价累计 |
| `q3/game_tree.py` | 博弈树 Minimax、Alpha-beta 与剪枝跟踪 |
| `q4/treasure_game.py` | 距离矩阵、宝藏状态转移与博弈搜索 |
| `experiments.py` | 统一实验、结果校验与计时 |
| `results/experiments.json` | 完整输入、原始计时、路径、节点统计与参考结果 |
| `results/summary.md` | 批量实验汇总表 |
| `plot_results.py` | 从保存数据生成报告配图 |
| `figures/` | 报告配图的 PNG 与 SVG 文件 |

生成配图前安装依赖：

```powershell
python -m pip install numpy matplotlib
```


运行实验与生成配图：
```powershell
python experiments.py --seed 20261111 --trials 3 --repeats 31
python plot_results.py
```

算法与实验脚本使用 Python 标准库，配图脚本使用 NumPy 和 Matplotlib。本报告图表均取自 `results/experiments.json` 保存的本次实验数据。实验设计、具体数据与结果分析见 [report.md](report.md)。统一实验生成的数据保存在 `results/`，配图保存在 `figures/`。

## 运行

```powershell
python q1/maze.py
python q2/weighted_astar.py
python q3/game_tree.py
python q4/treasure_game.py
```

默认运行给定样例。读取自定义输入文件，或使用 `--stdin` 从标准输入读取：

```powershell
python q1/maze.py --input q1/sample_input.txt
python q2/weighted_astar.py --input q2/sample_input.txt
```

前两题的输入文件为 UTF-8 编码，前两行分别为起点和终点坐标，其余内容为地图的 Python 二维列表。`S`、`E`、`X` 等字符需要加引号。例如：

```text
(0, 0)
(1, 2)
[
    ['S', 1, 9],
    [1, 1, 'E']
]
```

函数接口分别为 `maze.astar(start, end, grid)` 和 `weighted_astar.astar(start, end, grid)`。第二题的 `search` 函数另外返回路径及扩展节点数，供实验分析使用。

## 其他参数

四题均支持 `--compare` 运行对照；q2、q3、q4 支持 `--trace` 展示搜索过程。q3、q4 可用 `--algorithm minimax` 切换算法，q4 可用 `--depth 2` 或 `--depth 4` 指定深度。完整参数通过各脚本的 `--help` 查看。

