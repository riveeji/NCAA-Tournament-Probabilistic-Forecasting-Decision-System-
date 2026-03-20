# NCAA Tournament Probabilistic Forecasting & Decision System

面向 Kaggle NCAA March Machine Learning Mania 的多源数据融合、概率建模、自动化决策与提交流水线。

这个项目不是单一模型脚本，而是一套完整的锦标赛预测系统。它覆盖了从外部数据抓取、结构化特征构建、模型训练、概率融合、赛前运行时调整，到最终提交文件校验与发布的完整闭环。

## 1. 项目概览

系统的核心目标是：

- 面向 NCAA 男篮与女篮锦标赛，生成全路径 `132133` 个潜在对阵的胜率预测。
- 将官方历史数据、实时赔率、预测市场、外部 rating / matchup projection、伤病与人工补录信号统一融合到同一条预测链路。
- 在 Selection Sunday 以及临近截止时间窗口下，稳定刷新数据、生成候选提交稿、执行 sanity check，并输出最终推荐稿。

从工程定位上看，这个仓库更像一个**概率预测与决策系统**，而不是只跑一个 notebook 的竞赛项目。

## 2. 这套系统解决了什么问题

传统的 Kaggle 方案通常只做一件事：训练一个模型，然后导出一次提交文件。

这套系统把问题拆成了 5 层：

1. **数据层**
   统一接入官方赛程、赔率、预测市场、第三方 rating、matchup projection、伤病与人工补录信息。
2. **特征层**
   将 team-level 与 matchup-level 信号标准化，构造 men / women 共享但可差异化的结构化特征。
3. **模型层**
   使用多模型、多路由、多视角特征进行基础概率建模。
4. **融合与决策层**
   通过 stacking / meta fusion、runtime rules、goldshot override 等机制，面向真实比赛场景做受控调整。
5. **发布层**
   自动生成候选 submission、推荐最终稿、执行 sanity check、输出 hash 与评估报告。

## 3. 系统核心能力

- 同时支持 **NCAA 男篮 + 女篮** 双赛道预测
- 支持 **historical CV / replay / holdout / scenario simulation**
- 支持 **market odds + prediction markets + external matchup models** 融合
- 支持 **Selection Sunday 自动化刷新**
- 支持 **bounded override / goldshot 决策层**
- 支持 **提交前概率分布、重复 ID、hash 校验**

## 4. 技术栈

### 4.1 编程语言与数据处理

- Python
- Pandas
- NumPy
- JSON / CSV / XLSX 数据管道

### 4.2 机器学习与建模框架

- scikit-learn
  - Logistic Regression
  - HistGradientBoostingClassifier / Regressor
  - ExtraTreesClassifier
  - Isotonic Regression
  - StandardScaler / Pipeline
- XGBoost
- LightGBM（可选）
- CatBoost
- TabPFN（可选）
- Optuna（部分实验调参）

### 4.3 文本与降维

- TF-IDF
- TruncatedSVD
- sentence-transformers / transformers（可选文本增强）

### 4.4 数据抓取与工程化

- requests
- BeautifulSoup
- lxml / html5lib
- Playwright
- rapidfuzz
- Git / GitHub

## 5. 使用的方法与建模思路

### 5.1 基础特征

系统使用的核心特征包括但不限于：

- Elo / 动态 Elo
- 进攻效率 / 防守效率 / 净效率
- 近期状态（Recent / Recent30）
- 赛程强度（SOS）
- 种子差、主场 / 中立场、host 相关特征
- 市场隐含概率、spread、market confidence、book count
- 外部 rating / matchup projection
- 文本嵌入与文本衍生信号（可选）

### 5.2 多路由建模

系统不是单一模型，而是按信号视图分路：

- `stats` 路由
- `market` 路由
- `text` 路由
- `tabpfn` 路由（可选）

每条路由可以训练不同模型，再统一汇总为 OOF 和最终预测输出。

### 5.3 融合策略

在基础模型输出之上，系统进一步做融合：

- simple blend
- linear meta
- HistGB meta
- loss selector
- legacy anchor（历史兼容层）

目标不是追求某一个模型极致最优，而是提升最终概率的稳定性与校准质量。

### 5.4 赛前运行时决策

在最终提交阶段，系统还会引入：

- men / women runtime rules
- market-model consensus
- goldshot override
- injury veto

这些逻辑只对少数高杠杆真实比赛做**受控修正**，避免大面积手工调整概率。

### 5.5 评估方法

核心评估指标为：

- **Brier Score**

同时配套：

- cross-validation
- holdout evaluation
- historical replay
- upset-band / scenario simulation
- final candidate recommendation

## 6. 外部数据源

当前系统支持并实际使用过的数据源包括：

### 6.1 官方 / 基础数据

- Kaggle 官方比赛数据
- NCAA tournament seeds / slots / results

### 6.2 赔率与市场

- The Odds API
- TeamRankings（部分 men spread backup）
- Action Network（补充 / 校验）
- Kalshi
- Polymarket

### 6.3 第三方 rating / matchup projection

- Silver Bulletin
- Bart Torvik
- Warren Nolan
- Her Hoop Stats
- CollegeHoopsHub
- Public external ratings

### 6.4 伤病 / 可用性

- RotoWire
- 人工补录 / availability watchlist

说明：

- 不同源在不同时间点可用性不同。
- 当前流水线已尽量做成**非阻塞容错**，不要求每个外部源每次都成功。

## 7. 仓库结构

```text
.
├─ hc/                         # HC 核心系统：训练、预测、融合、规则、数据源抽象
│  ├─ train.py
│  ├─ predict.py
│  ├─ features_structured.py
│  ├─ data_sources.py
│  ├─ fusion.py
│  ├─ models_routes.py
│  ├─ rules.py
│  └─ ...
├─ tools/                      # 数据刷新、转换、pipeline、校验与后处理脚本
│  ├─ run_selection_sunday_pipeline.py
│  ├─ fetch_theoddsapi_odds.py
│  ├─ fetch_kalshi_prediction_markets.py
│  ├─ fetch_polymarket_prediction_markets.py
│  ├─ fetch_barttorvik_matchup_projections.py
│  ├─ fetch_warrennolan_predict_winners.py
│  ├─ fetch_herhoopstats_matchup_projections.py
│  ├─ fetch_rotowire_injuries.py
│  ├─ build_goldshot_override_candidates.py
│  ├─ apply_goldshot_overrides.py
│  ├─ build_final_submission_recommendation.py
│  ├─ check_submission_sanity.py
│  └─ ...
├─ zizzii_features.py          # 旧主链 / baseline 特征逻辑
├─ zizzii_train.py             # 旧主链训练入口
├─ external-data/             # 外部数据缓存（默认不提交到 GitHub）
├─ ncaa-data/                 # 官方 Kaggle / NCAA 数据（默认不提交到 GitHub）
├─ results/                   # 结果与报告（默认不提交到 GitHub）
├─ README.md
├─ README.zh-CN.md
├─ requirements-kaggle.txt
├─ requirements-hc.txt
└─ ...
```

## 8. 为什么 GitHub 版本默认不包含数据

为了避免仓库体积过大、外部数据授权不清晰，以及结果文件污染源码仓库，以下目录默认不上传：

- `external-data/`
- `ncaa-data/`
- `results/`
- 生成型 submission 文件

因此 GitHub 仓库主要用于：

- 保留源码
- 保留文档
- 保留数据格式和运行流程

而不是作为完整数据 dump。

如果要本地复现，需要自行准备：

- Kaggle 官方数据
- 外部抓取数据
- 相关 API Key / 登录态

## 9. 环境准备

### 9.1 创建环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-kaggle.txt
python -m pip install -r requirements-hc.txt
python -m playwright install chromium
```

### 9.2 可选环境变量

```powershell
$env:THE_ODDS_API_KEY="your_the_odds_api_key"
$env:HERHOOPSTATS_EMAIL="your_email"
$env:HERHOOPSTATS_PASSWORD="your_password"
```

注意：

- 不要把任何密钥、账号密码或 session 直接写进代码或提交到 GitHub。

## 10. 训练与预测

### 10.1 HC 训练

```powershell
.\.venv\Scripts\python.exe hc\train.py --mode cv --years 8 --genders M W --profile aggressive
```

常用参数：

- `--mode cv`
- `--years`
- `--genders`
- `--market-policy`
- `--profile`
- `--text on|off|auto`
- `--tabpfn on|off|auto`
- `--text-dim`
- `--force-rebuild`
- `--quick`

### 10.2 生成单次提交

```powershell
.\.venv\Scripts\python.exe hc\predict.py --season 2026 --profile aggressive --runtime-rules silver --output submission_stage2_single_final_hc.csv
```

常用参数：

- `--season`
- `--profile`
- `--market-policy`
- `--runtime-rules silver|off`
- `--strict-replay`
- `--output`

### 10.3 旧主链训练

```powershell
.\.venv\Scripts\python.exe zizzii_train.py --mode selection_sunday_final --season 2026
```

## 11. Selection Sunday 自动化流水线

完整刷新与出稿入口：

```powershell
.\.venv\Scripts\python.exe tools\run_selection_sunday_pipeline.py
```

常用选项：

- `--skip-check`
- `--skip-ratings`
- `--skip-odds`
- `--skip-train`
- `--skip-hc`
- `--skip-final-check`
- `--skip-candidate-reports`
- `--gender`
- `--api-key`

这一条流水线会负责：

- 官方数据检查
- public ratings 刷新
- odds / prediction market 刷新
- HC submission 生成
- candidate reports
- goldshot / recommendation
- final sanity check

## 12. 常用输出

常见输出包括：

- 根目录最终提交稿
  - `submission_stage2_single_final_hc.csv`
- 各类候选稿
  - `results/submission_stage2_single_final_hc_goldshot.csv`
  - `results/submission_stage2_single_final_hc_current.csv`
- 推荐报告
  - `results/final_submission_recommendation_*.json`
- sanity / hash 校验
  - `results/final_submission_check_*.json`
- CV / replay / scenario reports
  - `results/*.json`
  - `results/*.csv`

## 13. 适合写进简历的技术点

如果把这个项目写进简历，最值得强调的是：

- 多源异构信号融合
- 概率预测与 Brier Score 评估
- ensemble learning / stacking / meta fusion
- market-implied probability / no-vig odds conversion
- runtime decision layer / bounded override
- automated release pipeline
- historical replay / scenario simulation

## 14. 公开上传时的注意事项

建议只上传：

- 源码
- 依赖文件
- README / 设计文档
- 不包含敏感信息的配置模板

不要上传：

- API key
- 登录凭证
- 大型外部数据缓存
- 本地生成的结果文件
- 私有 submission 历史稿

## 15. 当前状态说明

这个仓库当前已经具备：

- 完整代码结构
- 训练 / 预测 / 刷新 / 提交流水线
- 中文主 README
- GitHub 上传前的数据忽略规则

如果要真正推送到 GitHub，还需要：

- 一个目标仓库地址
- 当前机器上可用的 GitHub 认证方式（PAT / SSH key / 已登录的 Git 凭证）

## 16. License

当前仓库未单独附加开源许可证。

如果要公开发布到 GitHub，建议在上传前根据你的用途补充：

- MIT
- Apache-2.0
- 或仅保留私有仓库
