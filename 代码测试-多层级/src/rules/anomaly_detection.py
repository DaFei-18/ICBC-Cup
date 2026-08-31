# -*- coding: utf-8 -*-
"""
异常交易与轻量级防刷单规则引擎
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class AnomalyRuleEngine:
    time_concentration_threshold: float = 0.35
    customer_repeat_threshold: float = 0.50
    logistics_missing_threshold: float = 0.15

    def check_time_concentration(self, orders: pd.DataFrame) -> dict:
        orders = orders.copy()
        orders["order_date"] = pd.to_datetime(orders["order_time"]).dt.date
        daily_counts = orders.groupby("order_date").size()
        if len(daily_counts) == 0:
            return {"triggered": False, "metric": 0.0, "detail": "无订单数据"}
        max_day_share = daily_counts.max() / daily_counts.sum()
        triggered = max_day_share > self.time_concentration_threshold
        return {
            "triggered": bool(triggered),
            "metric": round(float(max_day_share), 4),
            "detail": f"单日最高订单集中度 {max_day_share:.1%}" + (" (触发异常)" if triggered else " (正常)")
        }

    def check_customer_repeat(self, orders: pd.DataFrame) -> dict:
        cust_counts = orders["customer_id"].value_counts()
        if len(cust_counts) == 0:
            return {"triggered": False, "metric": 0.0, "detail": "无客户数据"}
        top5_share = cust_counts.head(5).sum() / cust_counts.sum()
        triggered = top5_share > self.customer_repeat_threshold
        return {
            "triggered": bool(triggered),
            "metric": round(float(top5_share), 4),
            "detail": f"Top5客户订单占比 {top5_share:.1%}" + (" (触发异常)" if triggered else " (正常)")
        }

    def check_logistics_missing(self, orders: pd.DataFrame) -> dict:
        if "has_logistics_record" not in orders.columns:
            return {"triggered": False, "metric": 0.0, "detail": "无物流记录字段"}
        missing_rate = 1 - orders["has_logistics_record"].mean()
        triggered = missing_rate > self.logistics_missing_threshold
        return {
            "triggered": bool(triggered),
            "metric": round(float(missing_rate), 4),
            "detail": f"物流轨迹缺失率 {missing_rate:.1%}" + (" (触发异常)" if triggered else " (正常)")
        }

    def evaluate_shop(self, orders: pd.DataFrame) -> dict:
        r1 = self.check_time_concentration(orders)
        r2 = self.check_customer_repeat(orders)
        r3 = self.check_logistics_missing(orders)
        n_triggered = sum([r1["triggered"], r2["triggered"], r3["triggered"]])
        return {
            "时间集中度": r1,
            "客户重复度": r2,
            "物流缺失率": r3,
            "触发规则数": n_triggered,
            "风险建议": "疑似刷单高危商户" if n_triggered >= 2 else (
                "存在异常波动" if n_triggered == 1 else "经营流水正常")
        }

    def batch_evaluate(self, orders: pd.DataFrame, shop_id_col: str = "shop_id") -> pd.DataFrame:
        rows = []
        for shop_id, g in orders.groupby(shop_id_col):
            res = self.evaluate_shop(g)
            rows.append({
                "shop_id": shop_id,
                "时间集中度指标": res["时间集中度"]["metric"],
                "客户重复度指标": res["客户重复度"]["metric"],
                "物流缺失率指标": res["物流缺失率"]["metric"],
                "触发规则数": res["触发规则数"],
                "风险建议": res["风险建议"]
            })
        return pd.DataFrame(rows).sort_values("触发规则数", ascending=False)


def _generate_demo_orders(n_normal_shops: int = 20, n_suspicious_shops: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    # 正常商户
    for i in range(n_normal_shops):
        shop_id = f"NORMAL{i:03d}"
        n_orders = rng.integers(80, 200)
        dates = pd.date_range("2025-01-01", periods=90, freq="D")
        order_dates = rng.choice(dates, size=n_orders)
        customers = [f"CUST{rng.integers(0, n_orders * 3):05d}" for _ in range(n_orders)]
        for d, c in zip(order_dates, customers):
            records.append({
                "shop_id": shop_id, "order_time": d, "customer_id": c,
                "has_logistics_record": rng.random() > 0.03,
            })

    # 疑似刷单商户
    for i in range(n_suspicious_shops):
        shop_id = f"SUSPECT{i:03d}"
        n_orders = rng.integers(80, 200)
        burst_dates = pd.date_range("2025-02-10", periods=3, freq="D")
        order_dates = rng.choice(burst_dates, size=n_orders)
        few_customers = [f"CUSTX{j:03d}" for j in range(5)]
        customers = rng.choice(few_customers, size=n_orders)
        for d, c in zip(order_dates, customers):
            records.append({
                "shop_id": shop_id, "order_time": d, "customer_id": c,
                "has_logistics_record": rng.random() > 0.7,
            })

    return pd.DataFrame(records)
