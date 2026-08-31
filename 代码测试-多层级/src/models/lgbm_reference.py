# -*- coding: utf-8 -*-
"""
E心助贷 —— LightGBM 参考模型
==============================
实现计划书第五章"模型选型对比"中的参考模型：以 LightGBM 捕捉
非线性风险信号，并用 SHAP 值做事后解释，验证评分卡主模型是否
遗漏了重要的非线性特征交互。

依赖：lightgbm, shap（本地无网络环境未安装，接入真实数据时请先
`pip install lightgbm shap`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


@dataclass
class LightGBMReferenceModel:
    """LightGBM 参考模型，用于与评分卡主模型做"双模型对照"。"""

    params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 15,
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbose": -1,
    })
    model_: "lgb.LGBMClassifier" = field(default=None, repr=False)
    feature_names_: list = field(default_factory=list, repr=False)
    explainer_: "shap.TreeExplainer" = field(default=None, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMReferenceModel":
        if lgb is None:
            raise ImportError("请先 `pip install lightgbm` 后再训练LightGBM参考模型。")
        self.feature_names_ = list(X.columns)
        self.model_ = lgb.LGBMClassifier(**self.params)
        self.model_.fit(X, y)
        return self

    def predict_proba_bad(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]

    def fit_explainer(self, X_background: pd.DataFrame) -> None:
        """基于背景样本构建SHAP TreeExplainer（树模型可直接用TreeExplainer，速度快）。"""
        if shap is None:
            raise ImportError("请先 `pip install shap` 后再计算SHAP解释。")
        self.explainer_ = shap.TreeExplainer(self.model_)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        if self.explainer_ is None:
            self.fit_explainer(X)
        sv = self.explainer_.shap_values(X)
        # 新版shap对二分类可能返回 (n_samples, n_features, 2)，统一取正类
        if isinstance(sv, list):
            return sv[1]
        if sv.ndim == 3:
            return sv[:, :, 1]
        return sv

    def top_features_by_shap(self, X: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """计算全局特征重要度（平均|SHAP值|），用于图5-2风格的排序图。"""
        sv = self.shap_values(X)
        importance = np.abs(sv).mean(axis=0)
        df = pd.DataFrame({"特征": self.feature_names_, "平均|SHAP值|": importance})
        return df.sort_values("平均|SHAP值|", ascending=False).head(top_n).reset_index(drop=True)

    def explain_single(self, X_row: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """输出单个样本的Top-N正/负向风险因子，对应看板"风险归因Top5"模块。"""
        sv = self.shap_values(X_row)[0]
        df = pd.DataFrame({
            "特征": self.feature_names_,
            "取值": X_row.iloc[0].values,
            "SHAP贡献": sv,
        })
        df["方向"] = np.where(df["SHAP贡献"] > 0, "正向（提升风险）", "负向（降低风险）")
        return df.reindex(df["SHAP贡献"].abs().sort_values(ascending=False).index).head(top_n)


if __name__ == "__main__":
    import os
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_here, "../../data"))
    sys.path.insert(0, os.path.join(_here, "../features"))
    from generate_mock_data import generate_shop_features
    from feature_engineering import ALL_FEATURES
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    if lgb is None or shap is None:
        print("未检测到 lightgbm / shap，跳过本地运行示例。")
        print("请在具备网络的环境中执行： pip install lightgbm shap")
    else:
        df = generate_shop_features()
        X, y = df[ALL_FEATURES], df["is_bad"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = LightGBMReferenceModel().fit(X_train, y_train)
        p_test = model.predict_proba_bad(X_test)
        print("LightGBM 测试集AUC:", round(roc_auc_score(y_test, p_test), 4))

        model.fit_explainer(X_train)
        print("\n=== SHAP Top10 特征重要度 ===")
        print(model.top_features_by_shap(X_test).to_string(index=False))

        print("\n=== 单样本风险归因示例 ===")
        print(model.explain_single(X_test.iloc[[0]]).to_string(index=False))
