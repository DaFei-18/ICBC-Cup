# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd

# 强制将当前项目根目录置于 sys.path 首位，防止 PyCharm 路径串包
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.models.scorecard import ScorecardModel
from src.rules.anomaly_detection import AnomalyRuleEngine
from src.llm.explain import explain_score, LLMConfig
from src.features.feature_engineering import TIER_FEATURES, determine_model_tier


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
    """E心助贷 - 统一多层路由风控评估引擎"""

    def __init__(self, scorecards: Dict[str, ScorecardModel], rule_engine: Optional[AnomalyRuleEngine] = None):
        self.scorecards = scorecards
        self.rule_engine = rule_engine or AnomalyRuleEngine()

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

        # 1. 自动路由或指定层级
        selected_tier = tier or determine_model_tier(row)
        scorecard = self.scorecards.get(selected_tier, self.scorecards["L1"])

        # 2. 特征对齐与评分预测
        X_row = pd.DataFrame([row])[scorecard.feature_names_]
        score = int(scorecard.predict_score(X_row)[0])
        p_bad = float(scorecard.predict_proba_bad(X_row)[0])
        risk_grade = scorecard.risk_grade(score)
        breakdown = scorecard.score_breakdown(row)

        # 3. 规则检测与大模型解释
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

    def batch_assess(self, X: pd.DataFrame, tier: str = "L3", shop_ids: Optional[list] = None) -> pd.DataFrame:
        scorecard = self.scorecards.get(tier, self.scorecards["L1"])
        X_sub = X[scorecard.feature_names_]
        scores = scorecard.predict_score(X_sub)
        p_bad = scorecard.predict_proba_bad(X_sub)
        grades = [scorecard.risk_grade(s) for s in scores]
        ids = shop_ids if shop_ids is not None else [f"ROW{i}" for i in range(len(X))]
        return pd.DataFrame({
            "shop_id": ids, "score": scores, "p_bad": p_bad.round(4),
            "risk_grade": grades, "model_tier": tier,
        })


if __name__ == "__main__":
    print("正在初始化并训练多层级风控模型...")
    toolkit = CreditToolkit.load_pretrained()

    demo_shop = {
        "open_months": 18, "avg_monthly_orders": 1240, "order_volatility": 0.22,
        "avg_order_value": 128.5, "gmv_mom_volatility": 0.10, "revenue_trend_slope": 0.8,
        "repurchase_rate": 0.31, "refund_rate": 0.045, "dispute_rate": 0.012,
        "gmv_yoy_growth": 0.15, "customer_hhi": 0.09, "customer_geo_entropy": 2.3,
        "interruption_count": 0, "positive_review_rate": 0.93, "sku_change_rate": 0.05,
        "new_customer_ratio": 0.4, "avg_settlement_days": 4.2,
        "platform_penalty_count": 0, "abnormal_large_txn_ratio": 0.015,
    }

    # 1. 测试全特征自动路由评估
    result = toolkit.assess(demo_shop, shop_id="SHOP_DEMO_01", explain=True)
    print("\n========== 评估结果 ==========")
    print(f"商户编号: {result.shop_id}")
    print(f"匹配模型层级: {result.model_tier}")
    print(f"综合信用评分: {result.score} 分")
    print(f"风险等级: {result.risk_grade}")
    print(f"违约概率预测: {result.p_bad:.2%}")
    print("\n[分值贡献明细 Top5]")
    print(result.score_breakdown.head(6).to_string(index=False))
    print("\n[智能风控诊断意见]")
    print(result.natural_language_explanation)
