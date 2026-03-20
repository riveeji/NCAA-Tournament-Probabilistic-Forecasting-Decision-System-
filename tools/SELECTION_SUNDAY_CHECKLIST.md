# Selection Sunday Checklist

今年 `Stage 2` 只允许最终选择一次提交，所以这份清单的目标不是“尽快出一个文件”，而是“把最终那一个文件做成可追溯、可复核、可复现的 release”。

## 原则

- 先锁官方数据，再拉市场数据，再生成 submission。
- 不在最后一小时做结构性改动，只允许数据刷新和规则表更新。
- `HC` 作为主候选，baseline 保留作对照和兜底。
- 最后一步必须做 **哈希校验** 和 **概率分布检查**。

## 推荐执行顺序

1. 锁官方比赛文件
   - 检查 `ncaa-data/SampleSubmissionStage2.csv`
   - 检查 men / women `Seeds`、`Slots`、women host 相关输入是否已更新
   - 运行：
     - `python tools/check_competition_data.py`

2. 刷新公开 ratings
   - 运行：
     - `python tools/fetch_public_external_ratings.py`
     - `python tools/fetch_collegehoopshub_ratings.py`

3. 刷新 live odds / spread
   - 先确认 `THE_ODDS_API_KEY`
   - 运行：
     - `python tools/fetch_theoddsapi_odds.py --all`
     - `python tools/fetch_teamrankings_odds.py`

4. 检查 live market 覆盖和明显脏数据
   - 检查 `external-data/MMatchupOdds_2026.csv`
   - 检查 `external-data/WMatchupOdds_2026.csv`
   - 重点看：
     - men / women 是否都有数据
     - `MarketProb` 是否存在明显缺失
     - `LastSpread` 方向和极值是否异常

5. 生成 baseline submission
   - 运行：
     - `python zizzii_train.py`
   - 输出：
     - `submission_stage2_single_final.csv`

6. 生成 HC 主候选 submission
   - 运行：
     - `python hc/predict.py --season 2026 --output submission_stage2_single_final_hc.csv`

7. 做人工 sanity review
   - 对比 `submission_stage2_single_final.csv` 和 `submission_stage2_single_final_hc.csv`
   - 重点抽查：
     - men 重度 favorite
     - women 前两轮 host
     - 大 spread 场次
     - 概率极端接近 `0` / `1` 的场次

8. 选定唯一最终提交文件
   - 推荐默认优先：
     - `submission_stage2_single_final_hc.csv`
   - 如果 live market 覆盖异常差，再考虑切回 baseline

9. 对最终选定文件做 release 校验
   - 运行：
     - `python tools/check_submission_sanity.py --submission submission_stage2_single_final_hc.csv --summary-output results/final_submission_check.json`
   - 必查项目：
     - 行数是否等于官方 sample
     - `ID` 是否全量匹配
     - `Pred` 是否无 NaN / 无重复
     - `sha256` 是否已记录
     - `pred min / max / mean / std`
     - `pred_quantiles`
     - `pred_histogram`

10. 记录最终 release 信息
    - 在提交前保存：
      - 最终文件名
      - `sha256`
      - 对应 `results/final_submission_check*.json`
      - 提交时间
    - 这样如果 Kaggle 页面或本地文件被误覆盖，仍能确认“交出去的是不是同一个文件”

## 一键流程

如果你要按主候选 `HC` 直接跑完整流程，可以用：

```powershell
python tools/run_selection_sunday_pipeline.py --final-submission submission_stage2_single_final_hc.csv
```

如果最终决定提交 baseline：

```powershell
python tools/run_selection_sunday_pipeline.py --skip-hc --final-submission submission_stage2_single_final.csv
```

## 最后一步为什么必须做

因为 `Stage 2` 只允许一次最终选择提交，最后那份文件必须满足两件事：

- **哈希可复核**
  - 确认本地最终文件和你准备上传的文件是同一个二进制内容
- **概率分布可解释**
  - 避免出现全局塌缩、异常尖锐、NaN、重复 ID、或大量 `0.5` 之类的 release 事故
