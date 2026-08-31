# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
import pandas as pd

try:
    from src.features.feature_engineering import FEATURE_DISPLAY_NAMES
except ImportError:
    FEATURE_DISPLAY_NAMES = {}

try:
    from openai import OpenAI

    HAS_OPENAI_SDK = True
except ImportError:
    OpenAI = None
    HAS_OPENAI_SDK = False


def _to_display_name(feature_code: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(feature_code, feature_code)


def _get_contrib_col(df: pd.DataFrame) -> str:
    """自适应获取贡献度列名"""
    if "对评分的贡献" in df.columns:
        return "对评分的贡献"
    elif "分值贡献" in df.columns:
        return "分值贡献"
    raise KeyError("未在 DataFrame 中找到评分贡献列 ('对评分的贡献' 或 '分值贡献')")


PRESET_PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.5-flash",
        "api_key_env": "GEMINI_API_KEY",
        "doc": "https://aistudio.google.com/app/apikey",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
        "doc": "https://help.aliyun.com/zh/model-studio/compatible-mode",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "doc": "https://api-docs.deepseek.com/",
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-8k",
        "api_key_env": "MOONSHOT_API_KEY",
        "doc": "https://platform.moonshot.cn/docs",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4-flash",
        "api_key_env": "ZHIPU_API_KEY",
        "doc": "https://open.bigmodel.cn/dev/api",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "doc": "https://platform.openai.com/docs",
    },
}


@dataclass
class LLMConfig:
    provider: str = "gemini"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    def resolve(self) -> dict:
        if self.provider not in PRESET_PROVIDERS and self.base_url is None:
            raise ValueError(
                f"未知 provider '{self.provider}'，请提供 base_url 或从以下选择: {list(PRESET_PROVIDERS.keys())}"
            )
        preset = PRESET_PROVIDERS.get(self.provider, {})
        base_url = self.base_url or preset.get("base_url")
        model = self.model or preset.get("default_model")
        api_key = self.api_key or os.environ.get(preset.get("api_key_env", ""), "")
        if not api_key:
            raise ValueError(
                f"未找到 API Key。请设置环境变量 {preset.get('api_key_env')} 或在配置中传入 api_key。"
            )
        return {"base_url": base_url, "model": model, "api_key": api_key}


def _build_prompt(shop_id: str, score: int, risk_grade: str,
                  score_breakdown: pd.DataFrame, top_n: int = 5) -> str:
    contrib_col = _get_contrib_col(score_breakdown)
    breakdown = score_breakdown[score_breakdown["特征"] != "基准截距分"].copy()
    breakdown = breakdown.reindex(breakdown[contrib_col].abs().sort_values(ascending=False).index)
    top = breakdown.head(top_n)

    lines = []
    for _, row in top.iterrows():
        direction = "加分" if row[contrib_col] > 0 else "扣分"
        display_name = _to_display_name(row["特征"])
        val = row["原始值"]
        val_str = f"{val:.2%}" if isinstance(val, (int, float)) and "率" in display_name else f"{val}"
        lines.append(f"- {display_name} (当前值: {val_str}): {direction} {abs(row[contrib_col]):.1f} 分")

    factor_text = "\n".join(lines)
    return f"""你是一名工商银行普惠金融部门的专业信贷风控专家。
请根据以下评分卡模型的输出结果，为信贷经理生成一段简明扼要、专业客观的商户信用诊断报告（控制在 150 字以内）：

【店铺编号】: {shop_id}
【综合评分】: {score} 分
【风险等级】: {risk_grade}
【核心关键影响特征 (Top {top_n})】:
{factor_text}

报告要求：
1. 先给出总体风险评价与授信建议；
2. 指出主要的加分优势项及主要扣分风险项；
3. 给出 1 条具体的经营改善或风控核实建议。
"""


def explain_score(shop_id: str, score: int, risk_grade: str,
                  score_breakdown: pd.DataFrame,
                  config: Optional[LLMConfig] = None,
                  top_n: int = 5) -> str:
    if config is None:
        config = LLMConfig()

    if not HAS_OPENAI_SDK:
        return (_fallback_explain(shop_id, score, risk_grade, score_breakdown, top_n)
                + "\n\n[提示: 未安装 openai 库，当前使用模板降级输出。]")

    try:
        resolved = config.resolve()
    except ValueError as e:
        return (_fallback_explain(shop_id, score, risk_grade, score_breakdown, top_n)
                + f"\n\n[提示: {e}，当前使用模板降级输出。]")

    prompt = _build_prompt(shop_id, score, risk_grade, score_breakdown, top_n)

    try:
        client = OpenAI(api_key=resolved["api_key"], base_url=resolved["base_url"])
        response = client.chat.completions.create(
            model=resolved["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return (_fallback_explain(shop_id, score, risk_grade, score_breakdown, top_n)
                + f"\n\n[提示: 模型接口调用失败 ({str(e)})，已自动切换为规则模板。]")


def _fallback_explain(shop_id: str, score: int, risk_grade: str,
                      score_breakdown: pd.DataFrame, top_n: int = 5) -> str:
    contrib_col = _get_contrib_col(score_breakdown)
    breakdown = score_breakdown[score_breakdown["特征"] != "基准截距分"].copy()
    breakdown = breakdown.reindex(breakdown[contrib_col].abs().sort_values(ascending=False).index)
    top = breakdown.head(top_n)

    pos = top[top[contrib_col] > 0]
    neg = top[top[contrib_col] < 0]

    parts = [f"店铺 {shop_id} 综合信用评分 {score} 分，评定为 {risk_grade}。"]
    if len(pos) > 0:
        pos_names = "、".join(_to_display_name(f) for f in pos["特征"].tolist())
        parts.append(f"该店铺在【{pos_names}】方面表现优秀，构成主要增信项。")
    if len(neg) > 0:
        neg_names = "、".join(_to_display_name(f) for f in neg["特征"].tolist())
        parts.append(f"但在【{neg_names}】方面存在一定风险敞口，构成主要扣分项。")
    parts.append("建议信贷团队结合该店铺实际流水做进一步复核。")
    return "".join(parts)
