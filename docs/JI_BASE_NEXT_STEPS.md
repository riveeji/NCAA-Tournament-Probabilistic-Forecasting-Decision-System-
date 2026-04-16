# JI_base 下一步优先级清单

## 当前状态

- 当前长期主线：`JI_base`
- 当前 `JI_base` 最佳变体：`JI_lr_control@none`
- 当前最佳 `total_cv_brier_calibrated`：`0.164156`
- 当前最佳 `total_cv_brier_raw`：`0.164165`
- 当前官方 LB 最优提交仍然是：`gold_recover_market = 0.1289`
- 当前 `JI_base` 还没有超过现有强基线
- 结论：`JI_base` 继续按 replay-first 推进，不做默认 submission 替换

## 总原则

- `JI_base` 只做长期主线，不引入 current-year overlay、market、injury、futures
- 所有改动优先服务这 6 个指标：
  - `men_cv_brier_raw`
  - `men_cv_brier_calibrated`
  - `women_cv_brier_raw`
  - `women_cv_brier_calibrated`
  - `total_cv_brier_raw`
  - `total_cv_brier_calibrated`
- 默认决策顺序保持不变：
  1. `total_cv_brier_calibrated`
  2. `men_cv_brier_calibrated`
  3. `women_cv_brier_calibrated`
  4. `latest_season_equal_gender_brier`
  5. `recent_window_equal_gender_brier`
  6. `long_run_mean_brier`

## 优先级 1：提升 women 质量层

### 目标

改进 women composite quality，使 `women_cv_brier_calibrated` 明显下降。

### 原因

- women 侧仍是 `JI_base` 最明显的结构性短板
- 公开高排名方案也反复指出 women composite ranking 是高价值缺口

### 要做的事

- 重构 women consensus quality builder，彻底分离 women 与 men 的 quality 构造逻辑
- 对 women 的 MOV / dominance / quality-wins 聚合加入保守非线性抑制，例如 `log1p`、平方根、clip、winsorize
- 保持 women schema 与 men 同构，但 women quality 的权重和缩放独立定义

### 验收标准

- `women_cv_brier_calibrated` 下降
- `total_cv_brier_calibrated` 同步下降
- `latest-season` 与 `recent-window` 不明显回撤

### 不要做

- 不要把 current-year injury、market、manual override 混进 women quality
- 不要对 women 全部 Elo/Quality 做全局强扭曲

## 优先级 2：主干特征做小步受控迭代

### 目标

在不改变 `JI_base` 主体架构的前提下，降低 `total_cv_brier_calibrated`。

### 原因

- 当前主干已经成型，下一步应该做可解释的小范围迭代
- spread-first spine 更依赖高价值小改动，而不是继续扩模型家族

### 要做的事

- 试验少量高解释性交互，例如 `Seed × Quality`、`Seed × women_consensus_quality`
- 对 `EloProb`、`strength_blend` 的配方做有限替换或微调
- 逐项评估效率差分列，删除边际弱且增加噪声的列

### 验收标准

- `total_cv_brier_calibrated` 下降
- men / women 至少一侧明确改善，另一侧不明显恶化
- 特征变更保持可解释，不引入宽表失控

### 不要做

- 不要重新引入自由大搜索
- 不要增加新的 current-year 数据源

## 优先级 3：拆解 alpha_profile 的真实贡献

### 目标

确认 `core_alpha_v1` 中哪些增强是长期有效的，哪些只是噪声。

### 原因

- 当前 alpha 是整包默认开启，但还没有系统性 ablation
- 必须先知道每个 alpha 的单独价值，再决定默认组合

### 要做的事

- 分别跑：
  - `harry_Rating only`
  - `QualityWins / OpponentQualityTournamentRank only`
  - `women AvgBlkDiff only`
- 比较 `alpha off`、`single-alpha`、`core_alpha_v1` 的 6 项 CV Brier 指标
- 保留明确有增益的 alpha，移除无效或高噪声项

### 验收标准

- 找到至少一个能稳定降低 `total_cv_brier_calibrated` 的 alpha 子集
- alpha 配置从“整包默认”变成“证据驱动的最小有效组合”

### 不要做

- 不要把 men injury 或 market 信息伪装成 alpha 放进 `JI_base`
- 不要只看单个 season 的好结果就固定默认

## 优先级 4：优化 Margin 到 Probability 的映射

### 目标

提升 raw probability 质量，减少后续 isotonic 的负担。

### 原因

- 当前 `MarginProbabilityMapper` 是正态 / probit 风格的固定实现，仍有优化空间
- 如果 raw probability 更稳，calibration 层的收益会更清晰

### 要做的事

- 比较 gender-specific residual scale 与当前统一映射的差异
- 尝试 season-window weighted residual scale
- 验证 mapper 改动对 raw / calibrated 六项 Brier 的影响

### 验收标准

- 至少降低 `total_cv_brier_raw`
- 不破坏概率单调性和 `[0,1]` 边界
- 不让 calibrated 指标整体恶化

### 不要做

- 不要直接上复杂分布或黑盒概率映射
- 不要让 isotonic 同时承担映射与校准两种职责

## 优先级 5：重新评估校准层是否值得保留

### 目标

判断 `isotonic_gender` 在 `JI_base` 上是否应继续作为默认候选。

### 原因

- 当前 replay 结果显示 isotonic 在 `JI_base` 上整体是负收益
- 需要分清是 mapper 已经足够好，还是 nested isotonic 样本不足导致不稳

### 要做的事

- 提高 isotonic 的最小样本门槛，减少过拟合
- 对比 `none` 与 `isotonic_gender` 的分性别收益
- 如果 isotonic 始终负收益，就把 `none` 固定为默认校准模式

### 验收标准

- 校准策略选择有明确证据支撑
- 避免为了“流程完整”而保留负收益校准

### 不要做

- 不要在 `JI_base` replay 中引入 sharpen
- 不要用全局 OOF 拟合校准器

## 优先级 6：提升 replay 迭代效率

### 目标

降低多变体 replay 的运行成本，支持更高频率的严谨实验。

### 原因

- 当前多变体 replay 虽然能跑通，但耗时仍然高
- 如果迭代成本太高，women quality 和 alpha ablation 会被拖慢

### 要做的事

- 继续下沉 profile / dataset cache
- 减少 replay 中重复的 inner-OOF 训练
- 如有需要，只引入 DuckDB 做 cache 和 artifact join，不改主逻辑

### 验收标准

- 多变体 replay 明显提速
- 缓存不改变结果，只改变运行成本

### 不要做

- 不要引入 PostgreSQL
- 不要为了性能改动评估口径

## 优先级 7：第一条架构多样性控制线 `JI_node_control`

### 目标

在 `JI_base` 主干稳定后，增加一条误差结构不同的控制分支。

### 原因

- 长期上限不只来自单一主模型，也来自结构化的架构多样性
- `NODE` 的工程复杂度低于 `TabR`，更适合作为第一条实验分支

### 要做的事

- 复用 `team_profile_v2` 与相同评估口径，实现 `JI_node_control`
- 只作为 control / experimental branch，不接管主线
- 只有 replay 明显增益时，才进入轻量 blend 候选

### 验收标准

- 提供与 XGB 不同的误差分布
- 若进入 blend，能稳定带来小幅增益

### 不要做

- 不要在 `JI_base` v1 未稳定前就把 `NODE` 升成主线
- 不要同时引入 `NODE` 和 `TabR`

## 优先级 8：未来单独实现 `JI_base_overlay`

### 目标

在不污染长期主线的前提下，保留 current-year alpha 的扩展空间。

### 原因

- market、injury、text/LLM 抽取信号对单年冲榜有价值，但不属于长期 replay 主线
- 必须物理分离 `JI_base` 和 `JI_base_overlay`

### 要做的事

- 等 `JI_base` 主干稳定后，再设计 `JI_base_overlay`
- overlay 只接 current-year market、injury、LLM/text-derived structured signals
- 与 `JI_base` 的 replay、summary、submission 结果分开存储

### 验收标准

- `JI_base` 保持长期主线纯度
- `JI_base_overlay` 可以独立评估单年增益

### 不要做

- 不要把 overlay 提前并入 `JI_base`
- 不要用单年冲榜技巧反向污染长期模型选择

## 当前最该做的前 3 项

1. 提升 women 质量层
2. 主干特征做小步受控迭代
3. 拆解 alpha_profile 的真实贡献
