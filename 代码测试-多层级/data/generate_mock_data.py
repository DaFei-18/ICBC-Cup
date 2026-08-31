# -*- coding: utf-8 -*-
"""
E心助贷 —— 模拟数据生成模块
============================
本模块生成用于原型验证的电商小微商户模拟经营数据，字段设计对齐
计划书第四章"特征体系设计"中定义的五大类共18项核心特征。

重要声明：
本模块生成的全部数据均为规则化模拟数据，不采集、不使用任何真实
商户信息，仅用于演示信用评估建模流程，不构成真实信贷决策依据。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SHOPS = 3000
BAD_RATE = 0.18  # 模拟坏客户占比，仅用于演示，非真实经验值


def _rng(seed: int = RANDOM_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_shop_features(n_shops: int = N_SHOPS, bad_rate: float = BAD_RATE,
                            seed: int = RANDOM_SEED, signal_strength: float = 0.6) -> pd.DataFrame:
    """生成 n_shops 家模拟店铺的18项核心特征 + 标签(is_bad)。

    生成逻辑（避免"完美可分"的不真实模拟）：
    1. 先为每家店铺抽取一个潜在风险变量 latent_risk ~ N(0, 1)；
    2. 每项特征 = 潜在风险的线性函数（信号强度由 signal_strength 控制）
       + 与风险无关的独立噪声，模拟真实业务中"特征只是弱信号、
       并非标签本身"的情况；
    3. 标签 is_bad 由潜在风险通过逻辑函数生成 Bernoulli 抽样，
       保留不可约的标签噪声，而非按风险值确定性划分。
    这样得到的模拟数据集在建模后AUC通常落在0.75~0.85区间，
    与计划书图5-3展示的模拟评估结果量级一致，而非人为的完美可分。
    """
    rng = _rng(seed)
    n = n_shops
    latent_risk = rng.normal(0, 1, n)  # 越大风险越高

    # 通过逻辑函数 + 随机抽样生成标签，保留标签噪声
    logit = signal_strength * 2.2 * latent_risk - np.log((1 - bad_rate) / bad_rate)
    p_bad = 1 / (1 + np.exp(-logit))
    is_bad = rng.binomial(1, p_bad)

    df = pd.DataFrame({"shop_id": [f"SHOP{i:06d}" for i in range(n)]})
    df["is_bad"] = is_bad
    df["_latent_risk"] = latent_risk  # 仅用于生成过程，建模时不作为特征使用

    def mix(good_params, bad_params, clip_min=None, clip_max=None):
        """保留原接口：good_params/bad_params 用于给出该特征"低风险端"与
        "高风险端"的目标分布参数；实际取值由潜在风险变量插值决定，
        并叠加独立噪声，而不是按标签做确定性二选一。
        """
        # 以 latent_risk 的分位数位置在 good/bad 两组参考样本间做插值混合，
        # 相当于让该特征与真实风险呈现"有相关但不完美"的关系
        u = 1 / (1 + np.exp(-latent_risk))  # 映射到 (0,1)，作为混合权重
        good_sample = good_params(n)
        bad_sample = bad_params(n)
        vals = (1 - u) * good_sample + u * bad_sample
        # 叠加额外独立噪声，进一步弱化单特征的区分力
        noise_scale = (np.std(good_sample) + np.std(bad_sample)) / 2 * 0.6
        vals = vals + rng.normal(0, noise_scale, n)
        if clip_min is not None or clip_max is not None:
            vals = np.clip(vals, clip_min, clip_max)
        return vals

    # ---------- 1. 经营稳定性特征 ----------
    df["open_months"] = mix(
        lambda n: rng.normal(24, 8, n), lambda n: rng.normal(14, 7, n), 1, None
    ).round(0)
    df["avg_monthly_orders"] = mix(
        lambda n: rng.lognormal(mean=5.2, sigma=0.5, size=n),
        lambda n: rng.lognormal(mean=4.4, sigma=0.6, size=n),
        1, None,
    ).round(0)
    df["order_volatility"] = mix(
        lambda n: rng.normal(0.25, 0.08, n), lambda n: rng.normal(0.55, 0.18, n), 0, None
    ).round(3)
    df["interruption_count"] = mix(
        lambda n: rng.poisson(0.3, n), lambda n: rng.poisson(2.1, n), 0, None
    ).round(0).astype(int)

    # ---------- 2. 交易质量特征 ----------
    df["avg_order_value"] = mix(
        lambda n: rng.lognormal(mean=4.5, sigma=0.4, size=n),
        lambda n: rng.lognormal(mean=4.2, sigma=0.5, size=n),
        10, None,
    ).round(1)
    df["repurchase_rate"] = mix(
        lambda n: rng.normal(0.38, 0.10, n), lambda n: rng.normal(0.18, 0.09, n), 0, 1
    ).round(3)
    df["refund_rate"] = mix(
        lambda n: rng.normal(0.03, 0.015, n), lambda n: rng.normal(0.11, 0.045, n), 0, None
    ).round(3)
    df["dispute_rate"] = mix(
        lambda n: rng.normal(0.01, 0.006, n), lambda n: rng.normal(0.045, 0.02, n), 0, None
    ).round(3)
    df["positive_review_rate"] = mix(
        lambda n: rng.normal(0.96, 0.03, n), lambda n: rng.normal(0.82, 0.08, n), 0, 1
    ).round(3)

    # ---------- 3. 规模与成长性特征 ----------
    df["gmv_yoy_growth"] = mix(
        lambda n: rng.normal(0.22, 0.15, n), lambda n: rng.normal(-0.15, 0.20, n), None, None
    ).round(3)
    df["gmv_mom_volatility"] = mix(
        lambda n: rng.normal(0.12, 0.05, n), lambda n: rng.normal(0.28, 0.10, n), 0, None
    ).round(3)
    df["sku_change_rate"] = mix(
        lambda n: rng.normal(0.10, 0.08, n), lambda n: rng.normal(-0.05, 0.15, n), None, None
    ).round(3)
    df["revenue_trend_slope"] = mix(
        lambda n: rng.normal(1.5, 1.0, n), lambda n: rng.normal(-1.2, 1.3, n), None, None
    ).round(3)

    # ---------- 4. 客户结构特征 ----------
    df["customer_hhi"] = mix(
        lambda n: rng.normal(0.08, 0.03, n), lambda n: rng.normal(0.22, 0.08, n), 0, 1
    ).round(3)
    df["new_customer_ratio"] = mix(
        lambda n: rng.normal(0.35, 0.10, n), lambda n: rng.normal(0.55, 0.15, n), 0, 1
    ).round(3)
    df["customer_geo_entropy"] = mix(
        lambda n: rng.normal(2.6, 0.4, n), lambda n: rng.normal(1.6, 0.5, n), 0, None
    ).round(3)

    # ---------- 5. 资金流与合规特征 ----------
    df["avg_settlement_days"] = mix(
        lambda n: rng.normal(3.5, 1.2, n), lambda n: rng.normal(6.5, 2.5, n), 0, None
    ).round(1)
    df["platform_penalty_count"] = mix(
        lambda n: rng.poisson(0.1, n), lambda n: rng.poisson(1.3, n), 0, None
    ).round(0).astype(int)
    df["abnormal_large_txn_ratio"] = mix(
        lambda n: rng.normal(0.01, 0.008, n), lambda n: rng.normal(0.06, 0.03, n), 0, None
    ).round(3)

    return df


def generate_monthly_gmv_series(n_shops: int = 200, n_months: int = 12,
                                 seed: int = RANDOM_SEED) -> pd.DataFrame:
    """生成用于趋势类图表（如图4-3）演示的月度GMV时间序列样例数据。"""
    rng = _rng(seed + 1)
    records = []
    for i in range(n_shops):
        is_bad = 1 if i % 5 == 0 else 0  # 20%比例标记为经营异常，仅作演示
        base = rng.normal(55, 8)
        trend = rng.normal(-2.6, 1.0) if is_bad else rng.normal(3.2, 1.0)
        for m in range(1, n_months + 1):
            gmv = base + trend * m + rng.normal(0, 3)
            records.append({
                "shop_id": f"SHOP{i:06d}",
                "month": m,
                "gmv_index": round(max(gmv, 0), 1),
                "is_distressed": is_bad,
            })
    return pd.DataFrame(records)


if __name__ == "__main__":
    shop_df = generate_shop_features()
    # _latent_risk 仅为内部生成过程使用的潜在变量，导出前移除，
    # 避免下游误当作特征使用造成标签泄漏（data leakage）。
    export_df = shop_df.drop(columns=["_latent_risk"])
    export_df.to_csv("mock_shop_features.csv", index=False, encoding="utf-8-sig")
    print(f"已生成 {len(export_df)} 条模拟店铺特征数据 -> mock_shop_features.csv")
    print(f"模拟坏客户占比: {export_df['is_bad'].mean():.3f}")
    print(export_df.head())

    gmv_df = generate_monthly_gmv_series()
    gmv_df.to_csv("mock_gmv_timeseries.csv", index=False, encoding="utf-8-sig")
    print(f"已生成 {len(gmv_df)} 条模拟GMV时间序列数据 -> mock_gmv_timeseries.csv")
