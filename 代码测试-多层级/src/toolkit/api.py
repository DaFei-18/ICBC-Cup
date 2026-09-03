# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import pandas as pd

# 防跨目录串包保护
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.scorecard import ScorecardModel
from src.rules.anomaly_detection import AnomalyRuleEngine
from src.llm.explain import explain_score, LLMConfig
from src.features.feature_engineering import TIER_FEATURES, determine_model_tier, aggregate_from_orders_vectorized
from src.features.data_cleaner import ECommerceDataCleaner

@dataclass
class AssessmentResult:
    shop_id: str
    score: int
    risk_grade: str
    p_bad: float
    model_tier: str
    score_breakdown: pd.DataFrame
    anomaly_result: Optional[dict] = None
    natural_language_explanation: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "shop_id": self.shop_id,
            "score": self.score,
            "risk_grade": self.risk_grade,
            "p_bad": round(self.p_bad, 4),
            "model_tier": self.model_tier,
            "top_positive_factors": self.score_breakdown.head(3).to_dict("records"),
            "top_negative_factors": self.score_breakdown.tail(3).to_dict("records"),
            "anomaly": self.anomaly_result,
            "explanation": self.natural_language_explanation,
        }

class CreditToolkit:
    """E心助贷 - 统一多层路由风控评估引擎（含数据清洗与容错）"""
    def __init__(self, scorecards: Dict[str, ScorecardModel],
                 rule_engine: Optional[AnomalyRuleEngine] = None,
                 cleaner: Optional[ECommerceDataCleaner] = None):
        self.scorecards = scorecards
        self.rule_engine = rule_engine or AnomalyRuleEngine()
        self.cleaner = cleaner or ECommerceDataCleaner()

    @classmethod
    def train_new(cls, df: pd.DataFrame, y: pd.Series, **scorecard_kwargs) -> "CreditToolkit":
        scorecards = {}
        for tier, feats in TIER_FEATURES.items():
            valid_cols = [f for f in feats if f in df.columns]
            scorecards[tier] = ScorecardModel(**scorecard_kwargs).fit(df[valid_cols], y)
        return cls(scorecards=scorecards)

    @classmethod
    def load_pretrained(cls) -> "CreditToolkit":
        from data.generate_mock_data import generate_shop_features
        df = generate_shop_features()
        y = df["is_bad"]
        return cls.train_new(df, y)

    def assess(self, shop_features: dict | pd.Series,
               shop_id: str = "UNKNOWN",
               tier: Optional[str] = None,
               orders: Optional[pd.DataFrame] = None,
               explain: bool = False,
               llm_config: Optional[LLMConfig] = None) -> AssessmentResult:
        row = pd.Series(shop_features) if isinstance(shop_features, dict) else shop_features
        selected_tier = tier or determine_model_tier(row)
        scorecard = self.scorecards.get(selected_tier, self.scorecards["L1"])

        # 缺失特征填补为 NaN 兜底，防止 KeyError
        aligned = {col: row.get(col, np.nan) for col in scorecard.feature_names_}
        X_row = pd.DataFrame([aligned])[scorecard.feature_names_]

        score = int(scorecard.predict_score(X_row)[0])
        p_bad = float(scorecard.predict_proba_bad(X_row)[0])
        risk_grade = scorecard.risk_grade(score)
        breakdown = scorecard.score_breakdown(pd.Series(aligned))

        anomaly_result = self.rule_engine.evaluate_shop(orders) if orders is not None else None
        explanation = None
        if explain:
            explanation = explain_score(
                shop_id=shop_id, score=score, risk_grade=risk_grade,
                score_breakdown=breakdown, config=llm_config,
            )

        return AssessmentResult(
            shop_id=shop_id, score=score, risk_grade=risk_grade, p_bad=p_bad,
            model_tier=selected_tier, score_breakdown=breakdown,
            anomaly_result=anomaly_result, natural_language_explanation=explanation,
        )

    def batch_assess(self, X: pd.DataFrame, tier: Optional[str] = None, shop_ids: Optional[list] = None) -> pd.DataFrame:
        """批量评估：支持全自动自适应路由与缺失列容错"""
        if shop_ids is not None:
            ids = shop_ids
        elif "shop_id" in X.columns:
            ids = X["shop_id"].tolist()
        else:
            ids = [f"SHOP_{i:04d}" for i in range(len(X))]

        results = []
        for i, (_, row) in enumerate(X.iterrows()):
            selected_tier = tier if tier and tier != "auto" else determine_model_tier(row)
            scorecard = self.scorecards.get(selected_tier, self.scorecards["L1"])

            aligned = {col: row.get(col, np.nan) for col in scorecard.feature_names_}
            X_row = pd.DataFrame([aligned])[scorecard.feature_names_]

            score = int(scorecard.predict_score(X_row)[0])
            p_bad = float(scorecard.predict_proba_bad(X_row)[0])
            risk_grade = scorecard.risk_grade(score)

            results.append({
                "shop_id": ids[i],
                "score": score,
                "p_bad": round(p_bad, 4),
                "risk_grade": risk_grade,
                "model_tier": selected_tier
            })
        return pd.DataFrame(results)

    def assess_raw_orders(self, raw_orders_df: pd.DataFrame, tier: Optional[str] = None) -> pd.DataFrame:
        """端到端评估：清洗 -> 规则排查 -> 向量化特征提炼 -> 自动路由打分"""
        clean_orders = self.cleaner.clean_orders(raw_orders_df)
        anomaly_report = self.rule_engine.batch_evaluate(clean_orders, shop_id_col="shop_id")
        features = aggregate_from_orders_vectorized(clean_orders)
        scores_df = self.batch_assess(features, tier=tier)
        return pd.merge(scores_df, anomaly_report[["shop_id", "触发规则数", "风险建议"]], on="shop_id", how="left")