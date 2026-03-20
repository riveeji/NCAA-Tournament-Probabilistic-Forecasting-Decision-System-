import pandas as pd
import numpy as np
from pathlib import Path

# 1. 读取官方真实锦标赛赛果
data_dir = Path("ncaa-data")
df = pd.read_csv(data_dir / "MNCAATourneyCompactResults.csv")

# 只取 2019 年以后的数据，确保能凑够 250 行 (触发你的 MARKET_RESIDUAL_MIN_ROWS)
df = df[df['Season'] >= 2019].copy()

np.random.seed(42)
records = []
for _, row in df.iterrows():
    # 模拟 Vegas 盘口：给实际赢球的队伍分配 0.55 到 0.95 之间的隐含概率
    # 这让 MarketProb 成为一个强特征，你的残差模型就有东西可学了
    market_prob_w = np.clip(np.random.normal(0.75, 0.1), 0.55, 0.95)
    
    # Kaggle 标准：T1 永远是 ID 较小的队伍
    if row['WTeamID'] < row['LTeamID']:
        t1, t2, prob = row['WTeamID'], row['LTeamID'], market_prob_w
    else:
        t1, t2, prob = row['LTeamID'], row['WTeamID'], 1.0 - market_prob_w
        
    records.append({
        'Season': row['Season'], 
        'T1': t1, 
        'T2': t2, 
        'MarketProb': round(prob, 4)
    })

out_df = pd.DataFrame(records)

# 2. 保存到你的 external-data 目录
out_path = Path("external-data") / "MMatchupOdds_Mock.csv"
out_df.to_csv(out_path, index=False)

print(f"成功生成 Mock 历史赔率: {len(out_df)} 行")
print(f"已保存至: {out_path}")