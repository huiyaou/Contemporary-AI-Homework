import argparse
import math
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# 默认基线配置，也可以通过命令行参数调整特征设置和 C。
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
C_VALUES = (0.1, 1, 10, 100)
MAX_FEATURES = 5000
MODEL_NAMES = ("SVM", "LogisticRegression")


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data_unlabeled.csv")
    return (
        train_df["text"].astype(str).tolist(),
        train_df["target"].to_numpy(),
        test_df["text"].astype(str).tolist(),
    )


def split_data(texts, labels):
    return train_test_split(
        texts,
        labels,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


def build_vectorizer(
    max_features=MAX_FEATURES, ngram_range=(1, 1), sublinear_tf=False, min_df=1
):
    # 默认仅用单词；(1, 2) 同时包含单词和相邻双词组合。
    return TfidfVectorizer(
        max_features=max_features, ngram_range=ngram_range,
        sublinear_tf=sublinear_tf, min_df=min_df,
    )


def build_model(model_name, c_value):
    if model_name == "SVM":
        return SVC(kernel="linear", C=c_value, random_state=RANDOM_STATE)
    if model_name == "LogisticRegression":
        return LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"未知模型：{model_name}")


def run_experiments(
    x_train, x_val, y_train, y_val, c_values=C_VALUES, max_features=MAX_FEATURES,
    ngram_range=(1, 1), sublinear_tf=False, min_df=1, models=MODEL_NAMES,
):
    """在同一份数据划分和 TF-IDF 特征上比较模型与 C。"""
    results = []
    print("\n模型                   C    训练准确率  验证准确率  验证Macro-F1")
    for model_name in models:
        for c_value in c_values:
            model = build_model(model_name, c_value)
            started = perf_counter()
            model.fit(x_train, y_train)
            fit_seconds = perf_counter() - started

            train_predictions = model.predict(x_train)
            val_predictions = model.predict(x_val)
            train_accuracy = accuracy_score(y_train, train_predictions)
            val_accuracy = accuracy_score(y_val, val_predictions)
            val_macro_f1 = f1_score(
                y_val, val_predictions, average="macro", zero_division=0
            )
            results.append(
                {
                    "model": model_name,
                    "C": c_value,
                    "max_features": max_features,
                    "ngram_min": ngram_range[0],
                    "ngram_max": ngram_range[1],
                    "sublinear_tf": sublinear_tf,
                    "min_df": min_df,
                    "n_features": x_train.shape[1],
                    "random_state": RANDOM_STATE,
                    "validation_size": VALIDATION_SIZE,
                    "train_accuracy": train_accuracy,
                    "validation_accuracy": val_accuracy,
                    "validation_macro_f1": val_macro_f1,
                    "fit_seconds": fit_seconds,
                    # SVC 有多个二分类器，记录其中最大的求解器迭代次数。
                    "solver_iterations_max": int(model.n_iter_.max()),
                }
            )
            print(
                f"{model_name:<20} {c_value:>5g}"
                f"    {train_accuracy:.4f}      {val_accuracy:.4f}       {val_macro_f1:.4f}",
                flush=True,
            )

    results_df = pd.DataFrame(results)
    # 按验证准确率选方案；分数相同时优先较小的 C。
    best = results_df.sort_values(
        ["validation_accuracy", "C"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0]
    return results_df, best


def train_final_and_predict(
    texts, labels, test_texts, best, max_features=MAX_FEATURES, output_dir=OUTPUT_DIR,
    ngram_range=(1, 1), sublinear_tf=False, min_df=1,
):
    """锁定方案后，用全部有标签数据重新拟合特征和模型。"""
    vectorizer = build_vectorizer(max_features, ngram_range, sublinear_tf, min_df)
    x_full = vectorizer.fit_transform(texts)
    model = build_model(best["model"], best["C"])
    model.fit(x_full, labels)
    predictions = model.predict(vectorizer.transform(test_texts))
    # 保持示例提交格式：单列，无表头、无索引，顺序与测试数据一致。
    pd.DataFrame(predictions).to_csv(
        output_dir / "predictions.csv", index=False, header=False
    )
    return len(predictions)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="线性 SVM 与逻辑回归文本分类实验")
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES,
        help="要训练的模型，默认两种都训练",
    )
    parser.add_argument(
        "--max-features", type=int, default=MAX_FEATURES,
        help="TF-IDF 特征数量上限，默认 5000",
    )
    parser.add_argument(
        "--c-values", type=float, nargs="+", default=C_VALUES,
        help="要比较的 C，默认 0.1 1 10 100",
    )
    parser.add_argument(
        "--ngram-range", type=int, nargs=2, default=(1, 1), metavar=("MIN_N", "MAX_N"),
        help="词组长度范围，默认 1 1；1 2 表示单词和双词组合",
    )
    parser.add_argument(
        "--sublinear-tf", action="store_true",
        help="启用对数词频缩放，减弱重复词的影响；默认关闭",
    )
    parser.add_argument(
        "--min-df", type=int, default=1,
        help="词语至少出现于多少篇训练文本，默认 1",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="结果目录；相对路径以脚本目录为基准，默认 outputs",
    )
    parser.add_argument(
        "--validation-only", action="store_true",
        help="只比较验证结果，跳过最终重训和测试集预测",
    )
    args = parser.parse_args(argv)
    if args.max_features <= 0:
        parser.error("--max-features 必须为正整数")
    if args.min_df <= 0:
        parser.error("--min-df 必须为正整数")
    if any(not math.isfinite(c) or c <= 0 for c in args.c_values):
        parser.error("--c-values 必须为有限的正数")
    if not 1 <= args.ngram_range[0] <= args.ngram_range[1]:
        parser.error("--ngram-range 必须满足 1 <= MIN_N <= MAX_N")
    args.ngram_range = tuple(args.ngram_range)
    if not args.output_dir.is_absolute():
        args.output_dir = BASE_DIR / args.output_dir
    return args


def main(argv=None):
    args = parse_args(argv)
    texts, labels, test_texts = load_data()
    train_texts, val_texts, y_train, y_val = split_data(texts, labels)
    print(
        f"数据：训练 {len(train_texts)} | 验证 {len(val_texts)}"
        f" | 测试 {len(test_texts)}",
        flush=True,
    )

    # 先划分，后拟合；验证集只使用训练部分学到的词表和 IDF。
    vectorizer = build_vectorizer(
        args.max_features, args.ngram_range, args.sublinear_tf, args.min_df
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    results_df, best = run_experiments(
        x_train, x_val, y_train, y_val, args.c_values, args.max_features,
        args.ngram_range, args.sublinear_tf, args.min_df, args.models,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "validation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(
        f"\n最佳方案：{best['model']}，C={best['C']:g}，"
        f"验证准确率={best['validation_accuracy']:.4f}，"
        f"验证Macro-F1={best['validation_macro_f1']:.4f}",
        flush=True,
    )
    if args.validation_only:
        print(f"已保存：{results_path}")
        return
    prediction_count = train_final_and_predict(
        texts, labels, test_texts, best, args.max_features, args.output_dir,
        args.ngram_range, args.sublinear_tf, args.min_df,
    )
    print(
        f"已保存：{results_path}、{args.output_dir / 'predictions.csv'}"
        f"（{prediction_count} 条）"
    )


if __name__ == "__main__":
    main()
