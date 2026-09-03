# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Optional, List
import numpy as np
import pandas as pd


class ECommerceDataCleaner:
    """电商数据接入与清洗标准化组件"""
    ALIAS_MAP = {
        "shop_id": ["shop_id", "seller_id", "merchant_id", "store_id", "vender_id", "shopId"],
        "order_time": ["order_time", "order_purchase_timestamp", "time_stamp", "create_time", "pay_time", "created_at"],
        "customer_id": ["customer_id", "user_id", "buyer_id", "visitorid", "customerId"],
        "order_amount": ["order_amount", "price", "amount", "payment_value", "total_amount", "gmv"],
        "order_status": ["order_status", "status", "action_type", "state"],
        "logistics_time": ["order_delivered_customer_date", "shipping_time", "delivered_at", "logistics_date"]
    }

    def __init__(self, custom_mapping: Optional[Dict[str, str]] = None):
        self.custom_mapping = custom_mapping or {}

    def _find_column(self, df_cols: List[str], target_field: str) -> Optional[str]:
        if target_field in self.custom_mapping and self.custom_mapping[target_field] in df_cols:
            return self.custom_mapping[target_field]
        for alias in self.ALIAS_MAP.get(target_field, []):
            if alias in df_cols:
                return alias
        return None

    def clean_orders(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        cols = list(raw_df.columns)
        shop_c = self._find_column(cols, "shop_id")
        time_c = self._find_column(cols, "order_time")
        cust_c = self._find_column(cols, "customer_id")
        amt_c = self._find_column(cols, "order_amount")
        stat_c = self._find_column(cols, "order_status")
        logi_c = self._find_column(cols, "logistics_time")

        if not shop_c:
            raise ValueError(f"原始数据未检测到商户字段 (如 seller_id/merchant_id)")

        clean_df = pd.DataFrame()
        clean_df["shop_id"] = raw_df[shop_c].astype(str).str.strip()
        clean_df["customer_id"] = raw_df[cust_c].astype(str).str.strip() if cust_c else "UNKNOWN"

        clean_df["order_time"] = pd.to_datetime(raw_df[time_c], errors="coerce") if time_c else pd.Timestamp.now()

        #  金额去噪清洗（替换为更稳健的正则提取数字方式）
        if amt_c:
            # 提取所有连续的数字与小数点，剔除一切货币符号（如 $、¥、逗号）
            cleaned_amt = raw_df[amt_c].astype(str).str.extract(r'(\d+(?:\.\d+)?)')[0]
            clean_df["order_amount"] = pd.to_numeric(cleaned_amt, errors="coerce").fillna(0.0)
        else:
            clean_df["order_amount"] = 100.0

        if stat_c:
            clean_df["is_refund"] = raw_df[stat_c].astype(str).str.lower().isin(
                ["canceled", "cancelled", "refunded", "unavailable", "退款", "关闭"]
            )
        else:
            clean_df["is_refund"] = False

        if logi_c:
            clean_df["has_logistics_record"] = raw_df[logi_c].notna()
        else:
            clean_df["has_logistics_record"] = True

        return clean_df.dropna(subset=["order_time"]).sort_values("order_time").reset_index(drop=True)

