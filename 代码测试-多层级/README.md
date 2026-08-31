# E心助贷 —— 面向电商小微企业的大数据智能风控与融资服务平台

第十七届"工行杯"全国大学生金融科技创新大赛 · 参赛项目代码仓库

> **数据合规声明**：本仓库中的全部数据均为规则化模拟数据（见
> `data/generate_mock_data.py`），不采集、不使用任何真实商户信息，
> 仅用于演示信用评估建模的完整流程，**不构成实际信贷决策依据**。

## 项目简介

电商小微企业普遍缺乏抵押物，长期面临"融资难、融资贵"的问题。本项目提出
以电商经营数据（订单、交易、客户、物流）替代传统抵押物的信用评估思路，
采用"数据层—特征层—模型层—应用层"四层架构，并在模型层面采取"双模型
对照"策略：

- **主模型**：逻辑回归 + WOE分箱评分卡，具备完全可拆解的可解释性，
  满足金融监管场景对"看得懂、可追溯"的要求；
- **参考模型**：LightGBM + SHAP，用于验证评分卡是否遗漏重要的非线性
  风险信号，并给出特征重要度排序。

完整的项目设计文档（含背景分析、需求调研、竞品对比、商业模式设计等）
见参赛计划书（不在本仓库范围内）。本仓库仅包含技术演示所需的可运行代码。

## 项目结构

```
e-xin-zhu-dai/
├── data/
│   └── generate_mock_data.py      # 模拟数据生成（18项特征 + GMV时间序列）
├── src/
│   ├── features/
│   │   └── feature_engineering.py # 特征体系定义 + 从订单明细聚合特征
│   ├── models/
│   │   ├── scorecard.py           # 评分卡主模型（WOE分箱 + 逻辑回归）
│   │   ├── lgbm_reference.py      # LightGBM参考模型 + SHAP解释
│   │   └── evaluate.py            # AUC / KS / PSI 评估指标
│   ├── rules/
│   │   └── anomaly_detection.py   # 防刷单异常交易识别规则引擎
│   └── toolkit/
│       └── api.py                 # 对外统一调用接口（工具包封装）
├── dashboard/
│   └── app.py                     # Streamlit 可视化看板原型
├── tests/
│   └── test_pipeline.py           # 端到端单元测试
├── requirements.txt
└── README.md
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成模拟数据（可选，工具包会在无预训练模型时自动生成）
python data/generate_mock_data.py

# 3. 训练并测试评分卡主模型
python src/models/scorecard.py

# 4. 训练并测试LightGBM参考模型（需先安装 lightgbm、shap）
python src/models/lgbm_reference.py

# 5. 测试防刷单规则引擎
python src/rules/anomaly_detection.py

# 6. 测试工具包统一API
python -m src.toolkit.api

# 7. 启动可视化看板
streamlit run dashboard/app.py

# 8. 运行单元测试
pytest tests/ -v
```

## 核心设计说明

### 1. 为什么模拟数据不是"完美可分"的

早期版本的模拟数据生成逻辑直接按标签（好/坏客户）分别采样两组差异
很大的分布，导致训练出的模型 AUC 接近 1.0——这在真实业务中几乎不可能
出现，属于不真实的"builder's overfitting"。

现在的生成逻辑（`generate_shop_features`）引入了一个连续的潜在风险变量
`_latent_risk`，每个特征都是该变量的**含噪声弱信号**，标签本身也通过
逻辑函数 + 随机抽样生成（保留不可约的标签噪声）。最终训练出的评分卡
AUC 稳定在 0.75~0.80 区间，与计划书中给出的模拟评估结果（AUC≈0.78）
量级一致。

### 2. 评分卡实现未依赖第三方评分卡库

`src/models/scorecard.py` 中的WOE分箱、IV计算、标准评分卡转换公式
（`Score = A - B·ln(odds)`）均为原生 Python/NumPy 实现，没有使用
`scorecardpy` 等封装库，便于逐行讲解每一步的业务含义（尤其适合答辩
现场展示"这一分是怎么来的"）。

### 3. 工具包 API 的设计意图

`src/toolkit/api.py` 中的 `CreditToolkit` 类将评分卡、规则引擎的调用
细节完全封装，对接方（如银行风控系统或本仓库的 Streamlit 看板）只需
调用 `toolkit.assess(shop_features)` 即可拿到结构化的评估结果，不需要
关心内部的 WOE 分箱、LightGBM 参数等实现细节——这是为了在计划书中论证
"可复用性、可迁移至不同电商平台数据结构"这一差异化定位而设计的接口
形态。

## 已知限制 / TODO

- `lgbm_reference.py` 依赖 `lightgbm` 与 `shap`，本仓库开发环境为
  离线沙箱，未能实际运行训练与 SHAP 计算，已做导入失败时的优雅降级
  处理；建议在有网络的环境中安装依赖后重新验证一次。
- `dashboard/app.py` 依赖 `streamlit` 与 `plotly`，同样因离线环境未能
  运行 UI 层，但核心业务逻辑（健康度评分、评分归因）已单独抽出测试
  并通过。
- `feature_engineering.py` 中 `aggregate_from_orders()` 提供的是"从
  真实订单明细聚合特征"的实现模板，其中 `sku_change_rate`、
  `new_customer_ratio`、`avg_settlement_days`、`platform_penalty_count`
  等字段依赖商品维度、历史客户名单、结算时间戳、平台处罚记录等
  暂未建模的数据源，留空由实际接入的平台数据补充。

## 许可与用途

本仓库为高校学科竞赛参赛作品的技术演示代码，不用于任何商业用途。
