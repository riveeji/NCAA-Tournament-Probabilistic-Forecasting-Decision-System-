# Selection Sunday 中文清单

适用日期：

- `2026-03-15`
- 到最终截止前 `2026-03-19 16:00 UTC`

目标：

- 刷新官方 `2026` seeds
- 刷新 live odds 和外部 ratings
- 重新生成最终上传文件

最终上传目标：

- [submission_stage2_single_final.csv](</j:/ide-workspace/kaggle-ncaa prediction/submission_stage2_single_final.csv>)

## 1. 先检查官方数据

```powershell
.\.venv\Scripts\Activate.ps1
python tools\check_competition_data.py
```

确认：

- men `2026` seeds 已到位
- women `2026` seeds 已到位
- 没有官方 CSV 缺失

## 2. 设置赔率 API key

```powershell
$env:THE_ODDS_API_KEY="你的key"
```

## 3. 一键刷新并训练

```powershell
python tools\run_selection_sunday_pipeline.py
```

这个脚本会做：

- 官方数据检查
- public ratings 刷新
- CollegeHoopsHub ratings 刷新
- The Odds API 当前赔率刷新
- TeamRankings men spread 补充赔率刷新
- 主训练流重跑

## 4. 生成最终单提交文件

```powershell
python tools\postprocess_single_final.py
```

## 5. 提交前体检

```powershell
python tools\check_submission_sanity.py --submission submission_stage2_single_final.csv
```

必须确认：

- 行数正确
- `NaN/Inf = 0`
- 概率范围正常
- `ID` 与官方 sample 一致

## 6. 上传文件

只上传：

- [submission_stage2_single_final.csv](</j:/ide-workspace/kaggle-ncaa prediction/submission_stage2_single_final.csv>)

不要上传：

- [submission_stage2.csv](</j:/ide-workspace/kaggle-ncaa prediction/submission_stage2.csv>)

## 7. 最短命令版

```powershell
.\.venv\Scripts\Activate.ps1
$env:THE_ODDS_API_KEY="你的key"
python tools\run_selection_sunday_pipeline.py
python tools\postprocess_single_final.py
python tools\check_submission_sanity.py --submission submission_stage2_single_final.csv
```
