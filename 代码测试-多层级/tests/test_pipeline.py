# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.generate_mock_data import generate_shop_features
from src.features.feature_engineering import ALL_FEATURES, TIER_FEATURES
from src.rules.anomaly_detection import AnomalyRuleEngine, _generate_demo_orders
from src.toolkit.api import CreditToolkit

@pytest.fixture(scope="module")
def mock_df():
    return generate_shop_features(n_shops=1000, seed=42)

def test_mock_data_integrity(mock_df):
    """1. 验证模拟数据集结构与字段完整度"""
    assert len(mock_df) == 1000
    for f in ALL_FEATURES:
        assert f in mock_df.columns
    assert set(mock_df["is_bad"].unique()) <= {0, 1}

def test_anomaly_rule_engine():
    """2. 验证防刷单规则引擎的拦截能力"""
    demo_orders = _generate_demo_orders(n_normal_shops=5, n_suspicious_shops=3, seed=42)
    engine = AnomalyRuleEngine()
    result = engine.batch_evaluate(demo_orders)
    assert len(result) == 8
    assert "触发规则数" in result.columns
    suspicious = result[result["shop_id"].str.startswith("SUSPECT")]
    assert (suspicious["触发规则数"] >= 1).all()

def test_tiered_model_routing_and_assessment(mock_df):
    """3. 验证三层模型路由（L1/L2/L3）的自动降级与打分"""
    y = mock_df["is_bad"]
    toolkit = CreditToolkit.train_new(mock_df, y)
    row = mock_df.iloc[0]

    # 全量特征 -> 路由至 L3
    res_l3 = toolkit.assess(row[ALL_FEATURES], shop_id="TEST_SHOP_L3")
    assert res_l3.model_tier == "L3"
    assert 300 <= res_l3.score <= 900

    # 仅基础流水特征 -> 自动降级至 L1
    l1_data = row[TIER_FEATURES["L1"]].to_dict()
    res_l1 = toolkit.assess(l1_data, shop_id="TEST_SHOP_L1")
    assert res_l1.model_tier == "L1"
    assert 300 <= res_l1.score <= 900

    # 手动指定层级
    res_manual_l2 = toolkit.assess(row[ALL_FEATURES], shop_id="TEST_SHOP_L2", tier="L2")
    assert res_manual_l2.model_tier == "L2"

def test_batch_assess_with_tiers(mock_df):
    """4. 验证批量商户授信评估"""
    toolkit = CreditToolkit.train_new(mock_df, mock_df["is_bad"])
    batch_res = toolkit.batch_assess(mock_df.head(50), tier="L2")
    assert len(batch_res) == 50
    assert (batch_res["model_tier"] == "L2").all()
    assert set(batch_res.columns) == {"shop_id", "score", "p_bad", "risk_grade", "model_tier"}

def test_scorecard_monotonicity(mock_df):
    """5. 验证好坏客户评分的区分度（好客户平均分应高于坏客户）"""
    toolkit = CreditToolkit.train_new(mock_df, mock_df["is_bad"])
    scores = toolkit.scorecards["L3"].predict_score(mock_df[ALL_FEATURES])
    good_mean = scores[mock_df["is_bad"] == 0].mean()
    bad_mean = scores[mock_df["is_bad"] == 1].mean()
    assert good_mean > bad_mean
