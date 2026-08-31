# -*- coding: utf-8 -*-
"""
E心助贷 —— 模型评估模块
========================
实现计划书第五章"模型评估指标与实验设计"所述的三项核心指标：
    - AUC (Area Under Curve)：整体区分能力
    - KS (Kolmogorov-Smirnov)：好坏客户的最大区分能力
    - PSI (Population Stability Index)：跨时间窗口的稳定性

以及按时间序列划分训练/验证/测试集的实验设计辅助函数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC：y_score 为"坏客户概率"或"风险得分"（越大风险越高）。"""
    return float(roc_auc_score(y_true, y_score))


def compute_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS值：好坏客户累积分布之间的最大差值。"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """PSI：比较两个时间窗口（如训练期 vs 验证期）评分分布的稳定性。

    经验判断标准（行业常用口径）：
        PSI < 0.1   基本无偏移
        0.1 <= PSI < 0.25  存在一定偏移，需关注
        PSI >= 0.25  分布发生显著偏移，模型可能需要重新训练
    """
    breakpoints = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    breakpoints = np.unique(breakpoints)

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def psi_verdict(psi: float) -> str:
    if psi < 0.1:
        return "基本无偏移"
    elif psi < 0.25:
        return "存在一定偏移，需关注"
    else:
        return "分布发生显著偏移，建议重新训练"


def time_based_split(df: pd.DataFrame, date_col: str, train_ratio: float = 0.8,
                      valid_ratio: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按时间顺序切分训练集/验证集/测试集，避免"时间穿越"导致的过拟合假象。

    对应计划书第五章"实验设计"：前80%时间为训练集，
    中间10%为验证集，最后10%作为样本外测试集。
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    train = df_sorted.iloc[:n_train]
    valid = df_sorted.iloc[n_train:n_train + n_valid]
    test = df_sorted.iloc[n_train + n_valid:]
    return train, valid, test


def evaluation_report(y_true: np.ndarray, y_score: np.ndarray,
                       y_true_ref: np.ndarray | None = None,
                       y_score_ref: np.ndarray | None = None) -> dict:
    """生成完整评估报告；若提供参考期(ref)数据，同时计算PSI。"""
    report = {
        "AUC": round(compute_auc(y_true, y_score), 4),
        "KS": round(compute_ks(y_true, y_score), 4),
    }
    if y_score_ref is not None:
        psi = compute_psi(y_score_ref, y_score)
        report["PSI"] = round(psi, 4)
        report["PSI判定"] = psi_verdict(psi)
    return report


if __name__ == "__main__":
    import os
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_here, "../../data"))
    sys.path.insert(0, os.path.join(_here, "../features"))
    from generate_mock_data import generate_shop_features
    from feature_engineering import ALL_FEATURES
    from scorecard import ScorecardModel
    from sklearn.model_selection import train_test_split

    df = generate_shop_features()
    X, y = df[ALL_FEATURES], df["is_bad"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = ScorecardModel().fit(X_train, y_train)
    p_train = model.predict_proba_bad(X_train)
    p_test = model.predict_proba_bad(X_test)

    report = evaluation_report(y_test.values, p_test, y_score_ref=p_train)
    print("=== 评分卡模型评估报告（测试集） ===")
    for k, v in report.items():
        print(f"{k}: {v}")
