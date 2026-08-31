# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features.feature_engineering import ALL_FEATURES, TIER_FEATURES, group_health_scores
from src.toolkit.api import CreditToolkit
from src.llm.explain import LLMConfig, PRESET_PROVIDERS

st.set_page_config(page_title="E心助贷 - 智能风控看板", layout="wide")


@st.cache_resource
def get_toolkit() -> CreditToolkit:
    return CreditToolkit.load_pretrained()


@st.cache_data
def get_demo_shops() -> pd.DataFrame:
    from data.generate_mock_data import generate_shop_features
    return generate_shop_features(n_shops=50, seed=7)


def render_radar(scores: dict) -> go.Figure:
    categories = list(scores.keys())
    values = list(scores.values())
    values += values[:1]
    categories_closed = categories + categories[:1]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories_closed, fill="toself", name="店铺画像",
        line_color="#2E6FB0",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, margin=dict(l=30, r=30, t=30, b=30), height=380,
    )
    return fig


def main():
    st.title("E心助贷 · 电商小微企业风控评估平台")
    toolkit = get_toolkit()
    demo_shops = get_demo_shops()

    with st.sidebar:
        st.header("店铺与数据源配置")
        shop_idx = st.selectbox(
            "选择评估店铺", options=demo_shops.index,
            format_func=lambda i: demo_shops.loc[i, "shop_id"]
        )

        st.markdown("---")
        st.header("模型路由设置")
        selected_tier = st.selectbox(
            "授信模型层级",
            options=["Auto (自动路由)", "L1 (极简准入)", "L2 (标准经营)", "L3 (全维增信)"]
        )
        tier_param = None if "Auto" in selected_tier else selected_tier.split(" ")[0]

        st.markdown("---")
        st.header("AI 智能解释")
        enable_explain = st.checkbox("开启风控报告解读", value=False)
        llm_config = None
        if enable_explain:
            provider = st.selectbox("大模型提供商", options=list(PRESET_PROVIDERS.keys()))
            api_key_input = st.text_input(f"{provider} API Key", type="password")
            llm_config = LLMConfig(provider=provider, api_key=api_key_input or None)

    row = demo_shops.loc[shop_idx]

    # 模拟数据降级测试
    eval_row = row.copy()
    if tier_param == "L1":
        missing_cols = set(ALL_FEATURES) - set(TIER_FEATURES["L1"])
        eval_row[list(missing_cols)] = None
    elif tier_param == "L2":
        missing_cols = set(ALL_FEATURES) - set(TIER_FEATURES["L2"])
        eval_row[list(missing_cols)] = None

    result = toolkit.assess(
        eval_row, shop_id=row["shop_id"], tier=tier_param,
        explain=enable_explain, llm_config=llm_config
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("店铺基本信息")
        st.write(f"**店铺编号**: {row['shop_id']}")
        st.write(f"**开店时长**: {int(row['open_months'])} 个月")
        st.write(f"**月均订单**: {int(row['avg_monthly_orders'])} 笔")
    with col2:
        st.subheader("评估结论")
        st.metric("信用评分", result.score)
        st.write(f"**风险等级**: {result.risk_grade}")
        st.write(
            f"**匹配模型层级**: `{result.model_tier}` ({'极简准入版' if result.model_tier == 'L1' else '标准经营版' if result.model_tier == 'L2' else '全维增信版'})")
    with col3:
        st.subheader("核心质量指标")
        st.write(f"**退款率**: {row['refund_rate']:.2%}")
        st.write(f"**违规处罚数**: {int(row['platform_penalty_count'])}")
        st.write(f"**客户HHI指数**: {row['customer_hhi']:.3f}")

    st.markdown("---")
    col4, col5 = st.columns(2)
    with col4:
        st.subheader("五维经营健康度画像")
        radar_scores = group_health_scores(demo_shops, row)
        st.plotly_chart(render_radar(radar_scores), use_container_width=True)
    with col5:
        st.subheader(f"分值拆解 (Top 因素 - {result.model_tier} 模型)")
        breakdown = result.score_breakdown
        top_factors = breakdown[breakdown["特征"] != "基准截距分"].reindex(
            breakdown["分值贡献"].abs().sort_values(ascending=False).index
        ).head(5)
        st.dataframe(top_factors, use_container_width=True, hide_index=True)

    if enable_explain:
        st.markdown("---")
        st.subheader("AI 智能风控诊断意见")
        st.info(result.natural_language_explanation)


if __name__ == "__main__":
    main()
