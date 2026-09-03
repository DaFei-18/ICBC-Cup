# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

@dataclass
class WOEBinner:
    n_bins: int = 5
    bins_: np.ndarray = field(default=None, repr=False)
    woe_map_: dict = field(default_factory=dict, repr=False)
    iv_: float = 0.0

    def fit(self, x: pd.Series, y: pd.Series) -> "WOEBinner":
        valid = x.notna()
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        edges = np.unique(x[valid].quantile(quantiles).values)
        if len(edges) < 3:
            edges = np.array([x[valid].min() - 1e-6, x[valid].max() + 1e-6])
        self.bins_ = edges
        bin_idx = np.digitize(x[valid], edges[1:-1], right=True)
        df = pd.DataFrame({"bin": bin_idx, "y": y[valid]})
        
        bins_present = df["bin"].unique()
        k = len(bins_present)
        # 严谨归一化：分母计入平滑增量，确保概率和恒为 1.0
        total_good_smooth = (y == 0).sum() + 0.5 * k
        total_bad_smooth = (y == 1).sum() + 0.5 * k
        iv = 0.0
        woe_map = {}
        
        for b, g in df.groupby("bin"):
            n_good = (g["y"] == 0).sum() + 0.5
            n_bad = (g["y"] == 1).sum() + 0.5
            good_rate = n_good / total_good_smooth
            bad_rate = n_bad / total_bad_smooth
            woe = np.log(good_rate / bad_rate)
            iv += (good_rate - bad_rate) * woe
            woe_map[b] = woe
            
        self.woe_map_ = woe_map
        self.iv_ = float(iv)
        return self

    def transform(self, x: pd.Series) -> pd.Series:
        val = x.fillna(x.median() if pd.notna(x.median()) else 0.0)
        bin_idx = np.digitize(val, self.bins_[1:-1], right=True)
        return pd.Series(bin_idx, index=x.index).map(self.woe_map_).fillna(0.0)

@dataclass
class ScorecardModel:
    base_score: int = 600
    base_odds: float = 1 / 20
    pdo: int = 20
    n_bins: int = 5
    iv_threshold: float = 0.02
    binners_: dict = field(default_factory=dict, repr=False)
    lr_: LogisticRegression = field(default=None, repr=False)
    feature_names_: list = field(default_factory=list, repr=False)
    dropped_features_: dict = field(default_factory=dict, repr=False)
    A_: float = field(default=0.0, repr=False)
    B_: float = field(default=0.0, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ScorecardModel":
        all_candidates = list(X.columns)
        candidate_binners = {}
        kept_features = []
        dropped_features = {}

        for col in all_candidates:
            binner = WOEBinner(n_bins=self.n_bins).fit(X[col], y)
            candidate_binners[col] = binner
            if binner.iv_ >= self.iv_threshold:
                kept_features.append(col)
            else:
                dropped_features[col] = binner.iv_

        if len(kept_features) == 0:
            raise ValueError("IV阈值过高，所有特征都被剔除，请降低 iv_threshold")

        self.feature_names_ = kept_features
        self.binners_ = {col: candidate_binners[col] for col in kept_features}
        self.dropped_features_ = dropped_features

        woe_df = pd.DataFrame(index=X.index)
        for col in self.feature_names_:
            woe_df[col] = self.binners_[col].transform(X[col])
            
        self.lr_ = LogisticRegression(max_iter=1000)
        self.lr_.fit(woe_df, y)
        
        self.B_ = self.pdo / np.log(2)
        self.A_ = self.base_score - self.B_ * np.log(self.base_odds)
        return self

    def dropped_features_report(self) -> pd.DataFrame:
        rows = [{"剔除特征": col, "IV值": round(iv, 4)} for col, iv in self.dropped_features_.items()]
        df = pd.DataFrame(rows)
        if not df.empty:
            return df.sort_values("IV值", ascending=False).reset_index(drop=True)
        return pd.DataFrame(columns=["剔除特征", "IV值"])

    def _woe_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        woe_df = pd.DataFrame(index=X.index)
        for col in self.feature_names_:
            woe_df[col] = self.binners_[col].transform(X[col])
        return woe_df

    def predict_proba_bad(self, X: pd.DataFrame) -> np.ndarray:
        woe_df = self._woe_transform(X[self.feature_names_])
        return self.lr_.predict_proba(woe_df)[:, 1]

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        p_bad = self.predict_proba_bad(X)
        p_bad = np.clip(p_bad, 1e-6, 1 - 1e-6)
        odds = (1 - p_bad) / p_bad
        score = self.A_ + self.B_ * np.log(odds)
        return np.round(score).astype(int)

    def score_breakdown(self, x_row: pd.Series) -> pd.DataFrame:
        rows = []
        base_component = self.A_ - self.B_ * self.lr_.intercept_[0]
        for i, col in enumerate(self.feature_names_):
            woe = self.binners_[col].transform(pd.Series([x_row[col]]))[0]
            coef = self.lr_.coef_[0][i]
            contribution = -self.B_ * coef * woe
            rows.append({
                "特征": col, 
                "原始值": x_row.get(col, np.nan), 
                "WOE": round(woe, 4),
                "对评分的贡献": round(contribution, 1),
                "分值贡献": round(contribution, 1)
            })
        out = pd.DataFrame(rows).sort_values("对评分的贡献", ascending=False)
        out.loc[len(out)] = {"特征": "基准截距分", "原始值": "-", "WOE": "-", "对评分的贡献": round(base_component, 1), "分值贡献": round(base_component, 1)}
        return out

    def risk_grade(self, score: int) -> str:
        if score >= 720:
            return "A级 (低风险)"
        elif score >= 650:
            return "B级 (中低风险)"
        elif score >= 580:
            return "C级 (中风险)"
        else:
            return "D级 (高风险)"

    def iv_report(self) -> pd.DataFrame:
        rows = [{"特征": col, "IV": round(binner.iv_, 4)} for col, binner in self.binners_.items()]
        return pd.DataFrame(rows).sort_values("IV", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import os, sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from data.generate_mock_data import generate_shop_features
    from src.features.feature_engineering import ALL_FEATURES

    # 1. 生成仿真数据集（内部种子为 42）
    df = generate_shop_features(n_shops=1000, seed=42)

    # 2. 引入一个纯白噪声特征
    # 彻底解决：采用独立种子 999（或 9999），杜绝与内部 latent_risk 重叠导致标签泄露
    rng_noise = np.random.default_rng(999)
    df["pure_random_noise"] = rng_noise.normal(0, 1, len(df))

    candidate_cols = ALL_FEATURES + ["pure_random_noise"]
    X = df[candidate_cols]
    y = df["is_bad"]

    print(">>> 开始训练评分卡模型（启用 IV 动态筛选，阈值 0.02）...")
    model = ScorecardModel(iv_threshold=0.02).fit(X, y)

    print(f"\n[1] 筛选入模特征数: {len(model.feature_names_)} / {len(candidate_cols)}")
    print("\n[2] 入模特征 IV 排行 (Top 5):")
    print(model.iv_report().head(5).to_string(index=False))

    print("\n[3] 因 IV < 0.02 被自动剔除的特征报表 (dropped_features_report):")
    dropped_df = model.dropped_features_report()
    if not dropped_df.empty:
        print(dropped_df.to_string(index=False))
    else:
        print("无特征被剔除")