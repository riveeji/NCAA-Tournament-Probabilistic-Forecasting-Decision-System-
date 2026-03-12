# Kaggle NCAA March Mania 2026 中文技术说明

这个仓库已经收缩成单提交主线，只保留对最终结果有直接作用的训练、后处理、体检和比赛当天刷新流程。

当前 best-known 本地分数：

- men: `0.19215`
- women: `0.13740`
- `equal_gender`: `0.16478`

结果文件：

- [combined_cv_summary_20260312T040220Z.json](</j:/ide-workspace/kaggle-ncaa prediction/results/combined_cv_summary_20260312T040220Z.json>)

当前推荐上传文件：

- [submission_stage2_single_final.csv](</j:/ide-workspace/kaggle-ncaa prediction/submission_stage2_single_final.csv>)

## 当前主线

只保留这一条：

1. 主训练流  
   - [zizzii_features.py](</j:/ide-workspace/kaggle-ncaa prediction/zizzii_features.py>)
   - [zizzii_train.py](</j:/ide-workspace/kaggle-ncaa prediction/zizzii_train.py>)
2. 单文件后处理  
   - [tools/postprocess_single_final.py](</j:/ide-workspace/kaggle-ncaa prediction/tools/postprocess_single_final.py>)
3. 提交体检  
   - [tools/check_submission_sanity.py](</j:/ide-workspace/kaggle-ncaa prediction/tools/check_submission_sanity.py>)

比赛当天自动刷新入口：

- [tools/run_selection_sunday_pipeline.py](</j:/ide-workspace/kaggle-ncaa prediction/tools/run_selection_sunday_pipeline.py>)

## 仓库里还保留的关键工具

- [tools/check_competition_data.py](</j:/ide-workspace/kaggle-ncaa prediction/tools/check_competition_data.py>)
- [tools/fetch_public_external_ratings.py](</j:/ide-workspace/kaggle-ncaa prediction/tools/fetch_public_external_ratings.py>)
- [tools/fetch_collegehoopshub_ratings.py](</j:/ide-workspace/kaggle-ncaa prediction/tools/fetch_collegehoopshub_ratings.py>)
- [tools/fetch_theoddsapi_odds.py](</j:/ide-workspace/kaggle-ncaa prediction/tools/fetch_theoddsapi_odds.py>)
- [tools/fetch_teamrankings_odds.py](</j:/ide-workspace/kaggle-ncaa prediction/tools/fetch_teamrankings_odds.py>)
- [tools/postprocess_women_submission_b.py](</j:/ide-workspace/kaggle-ncaa prediction/tools/postprocess_women_submission_b.py>)

这些文件仍然保留，是因为它们还在主流程里被直接调用。

## 标准运行顺序

### 1. 检查官方数据

```powershell
python tools\check_competition_data.py
```

### 2. 训练主模型

```powershell
python zizzii_train.py
```

### 3. 生成最终单提交文件

```powershell
python tools\postprocess_single_final.py
```

### 4. 体检

```powershell
python tools\check_submission_sanity.py --submission submission_stage2_single_final.csv
```

### 5. 上传

上传：

- [submission_stage2_single_final.csv](</j:/ide-workspace/kaggle-ncaa prediction/submission_stage2_single_final.csv>)

不要直接上传：

- [submission_stage2.csv](</j:/ide-workspace/kaggle-ncaa prediction/submission_stage2.csv>)

因为它只是原始模型输出，不是当前推荐的最终版。

## Selection Sunday 当天

等 `2026` 官方 seeds 和 live odds 出来后，执行：

```powershell
$env:THE_ODDS_API_KEY="你的key"
python tools\run_selection_sunday_pipeline.py
python tools\postprocess_single_final.py
python tools\check_submission_sanity.py --submission submission_stage2_single_final.csv
```

## 外部数据说明

当前训练会直接读取 [external-data](</j:/ide-workspace/kaggle-ncaa prediction/external-data>) 下保留下来的有效 CSV，包括：

- men 历史赔率
- men / women 当前 matchup odds
- team ratings
- men manual signals

格式说明：

- [external-data/README.md](</j:/ide-workspace/kaggle-ncaa prediction/external-data/README.md>)

## 当前策略结论

- `2026-03-15` 前，不再继续修改核心算法。
- 后续只更新数据，不再扩张系统复杂度。
- 当前唯一推荐上传文件就是 [submission_stage2_single_final.csv](</j:/ide-workspace/kaggle-ncaa prediction/submission_stage2_single_final.csv>)。
