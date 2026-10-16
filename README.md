# 新闻文本分类实验

使用 TF-IDF 提取文本特征，分别使用线性支持向量机（SVM）和逻辑回归（Logistic Regression）进行十分类。

## 1. 运行环境

实验使用 Python 3.12.7、pandas 2.2.2、scikit-learn 1.5.1。

安装依赖：

```powershell
python -m pip install pandas==2.2.2 scikit-learn==1.5.1
```

目录结构：

```text
Project1分支/
├── my_experiment.py
└── data/
    ├── train_data.csv
    └── test_data_unlabeled.csv
```

训练数据包含 `text` 和 `target` 字段，测试数据包含 `text` 字段，类别标签为 0—9。

## 2. 运行规则

以下命令均在根目录执行。

程序将有标签数据按 **8:2 分层划分**为训练集和验证集，固定随机种子为 `42`。TF-IDF 仅在训练部分拟合，再用于转换验证集。

程序输出训练准确率、验证准确率和验证 Macro-F1，并按验证准确率选择当前运行中的最佳方案；准确率相同时优先选择较小的 C。

未添加 `--validation-only` 时，程序会使用全部有标签数据重新训练选定方案，生成测试集预测。

### 运行基础实验

```powershell
python my_experiment.py --validation-only --output-dir outputs/baseline
```

默认比较两个模型，使用最多 5000 个单词特征，尝试 `C=0.1、1、10、100`。

### 复现最终模型

最终采用逻辑回归，配置为：`C=100`、最多 20000 个单词特征、对数词频缩放、`min_df=2`。

```powershell
python my_experiment.py --models LogisticRegression --max-features 20000 --ngram-range 1 1 --sublinear-tf --min-df 2 --c-values 100 --output-dir outputs/final
```

该命令先验证指定配置，再使用全部 **7368 条**有标签数据重新拟合 TF-IDF 和模型，对 **2457 条**测试文本进行预测。

## 3. 参数说明

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--models` | 选择 `SVM`、`LogisticRegression`，可同时指定 | 两个模型 |
| `--max-features` | TF-IDF 特征数量上限 | `5000` |
| `--c-values` | 要比较的 C，可指定多个值 | `0.1 1 10 100` |
| `--ngram-range` | 词组长度；`1 1` 为单词，`1 2` 包含双词组合 | `1 1` |
| `--sublinear-tf` | 启用对数词频缩放 | 关闭 |
| `--min-df` | 词语至少出现于多少篇训练文本 | `1` |
| `--validation-only` | 仅验证，跳过最终重训和测试预测 | 关闭 |
| `--output-dir` | 输出目录，相对路径以脚本所在目录为基准 | `outputs` |

SVM 使用 `SVC(kernel="linear")`；逻辑回归使用 `lbfgs` 求解器，最大迭代次数为 `2000`。