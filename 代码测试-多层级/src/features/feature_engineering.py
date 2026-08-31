# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd

# ===========================================================================
# 1. 特征体系与五大维度定义
# ===========================================================================
FEATURE_GROUPS = {
    "经营稳定性": [
        "open_months", "avg_monthly_orders", "order_volatility", "interruption_count",
    ],
    "交易质量": [
        "avg_order_value", "repurchase_rate", "refund_rate", "dispute_rate",
        "positive_review_rate",
    ],
    "规模与成长性": [
        "gmv_yoy_growth", "gmv_mom_volatility", "sku_change_rate", "revenue_trend_slope",
    ],
    "客户结构": [
        "customer_hhi", "new_customer_ratio", "customer_geo_entropy",
    ],
    "资金流与合规": [
        "avg_settlement_days", "platform_penalty_count", "abnormal_large_txn_ratio",
    ],
}

ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]

# ===========================================================================
# 2. 三层模型特征子集定义 (L1 基础准入 -> L2 标准经营 -> L3 全维增信)
# ===========================================================================
TIER_FEATURES = {
    "L1": [
        "open_months", "avg_monthly_orders", "order_volatility",
        "avg_order_value", "gmv_mom_volatility", "revenue_trend_slope",
    ],
    "L2": [
        "open_months", "avg_monthly_orders", "order_volatility",
        "avg_order_value", "gmv_mom_volatility", "revenue_trend_slope",
        "repurchase_rate", "refund_rate", "dispute_rate",
        "gmv_yoy_growth", "customer_hhi", "customer_geo_entropy",
    ],
    "L3": ALL_FEATURES,
}

FEATURE_DISPLAY_NAMES = {
    "open_months": "开店时长", "avg_monthly_orders": "月均订单量",
    "order_volatility": "订单量波动系数", "interruption_count": "经营中断次数",
    "avg_order_value": "客单价", "repurchase_rate": "复购率", "refund_rate": "退款率",
    "dispute_rate": "纠纷投诉率", "positive_review_rate": "好评率",
    "gmv_yoy_growth": "GMV同比增速", "gmv_mom_volatility": "GMV环比波动",
    "sku_change_rate": "SKU变化率", "revenue_trend_slope": "营收趋势斜率",
    "customer_hhi": "客户集中度(HHI)", "new_customer_ratio": "新客户占比",
    "customer_geo_entropy": "客户地域分散度", "avg_settlement_days": "平均回款周期",
    "platform_penalty_count": "平台处罚数", "abnormal_large_txn_ratio": "异常大额交易占比",
}

FEATURE_DIRECTION = {
    "open_months": 1, "avg_monthly_orders": 1, "order_volatility": -1,
    "interruption_count": -1, "avg_order_value": 1, "repurchase_rate": 1,
    "refund_rate": -1, "dispute_rate": -1, "positive_review_rate": 1,
    "gmv_yoy_growth": 1, "gmv_mom_volatility": -1, "sku_change_rate": 1,
    "revenue_trend_slope": 1, "customer_hhi": -1, "new_customer_ratio": -1,
    "customer_geo_entropy": 1, "avg_settlement_days": -1,
    "platform_penalty_count": -1, "abnormal_large_txn_ratio": -1,
}

def determine_model_tier(features: dict | pd.Series | pd.DataFrame) -> str:
    """根据传入数据的非空字段，自动路由判定适用的模型层级"""
    if isinstance(features, pd.DataFrame):
        available = set(features.dropna(axis=1, how="all").columns)
    else:
        available = {k for k, v in features.items() if pd.notna(v)}

    if set(TIER_FEATURES["L3"]).issubset(available):
        return "L3"
    elif set(TIER_FEATURES["L2"]).issubset(available):
        return "L2"
    else:
        return "L1"

def health_scores(df: pd.DataFrame, row: pd.Series) -> dict:
    scores = {}
    for f in ALL_FEATURES:
        if f not in df.columns or pd.isna(row.get(f)):
            continue
        pct_rank = (df[f] <= row[f]).mean() * 100
        direction = FEATURE_DIRECTION.get(f, 1)
        scores[f] = pct_rank if direction == 1 else 100 - pct_rank
    return scores

def group_health_scores(df: pd.DataFrame, row: pd.Series) -> dict:
    single = health_scores(df, row)
    group_scores = {}
    for group, feats in FEATURE_GROUPS.items():
        vals = [single[f] for f in feats if f in single]
        group_scores[group] = round(float(np.mean(vals)), 1) if vals else 50.0
    return group_scores

def aggregate_from_orders_vectorized(orders: pd.DataFrame, as_of_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """基于订单流水的向量化特征聚合计算"""
    orders = orders.copy()
    orders["order_time"] = pd.to_datetime(orders["order_time"])
    if as_of_date is None:
        as_of_date = orders["order_time"].max()

    window_start = as_of_date - pd.Timedelta(days=365)
    recent = orders[orders["order_time"] >= window_start]

    base_features = recent.groupby("shop_id").agg(
        open_months=("order_time", lambda x: (as_of_date - x.min()).days / 30),
        avg_order_value=("order_amount", "mean"),
        refund_rate=("is_refund", "mean") if "is_refund" in recent.columns else ("order_time", lambda x: np.nan),
        dispute_rate=("is_dispute", "mean") if "is_dispute" in recent.columns else ("order_time", lambda x: np.nan),
    )

    monthly = recent.groupby([
        "shop_id",
        pd.Grouper(key="order_time", freq="MS"),
    ])["order_amount"].agg(monthly_count="count", monthly_sum="sum").reset_index()

    monthly_features = monthly.groupby("shop_id").agg(
        avg_monthly_orders=("monthly_count", "mean"),
        order_volatility=("monthly_count", lambda x: x.std() / (x.mean() + 1e-6)),
        gmv_mom_volatility=("monthly_sum", lambda x: x.pct_change().std()),
    )

    cust_amt = recent.groupby(["shop_id", "customer_id"])["order_amount"].sum().reset_index(name="cust_sum")
    cust_amt["shop_sum"] = cust_amt.groupby("shop_id")["cust_sum"].transform("sum")
    cust_amt["hhi_part"] = (cust_amt["cust_sum"] / (cust_amt["shop_sum"] + 1e-6)) ** 2
    hhi_features = cust_amt.groupby("shop_id").agg(customer_hhi=("hhi_part", "sum"))

    return pd.concat([base_features, monthly_features, hhi_features], axis=1).reset_index()
