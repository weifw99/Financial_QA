# generate_daily_signals.py
# 运行前: pip install pandas
# 使用: python generate_daily_signals.py
import os
import glob
import pandas as pd
from datetime import datetime

DATA_DIR = "results"
OUT_DIR = "signals"
CSV_PATTERN = "*_TET_full.csv"
os.makedirs(OUT_DIR, exist_ok=True)

BUY_RULE = lambda row: (row['trend'] > 0.3) and (row['joint_trend'] > 0.3) and (row['emotion'] < 0.3) and (row['timing'] > 0.5)
SELL_RULE = lambda row: (row['timing'] < -0.5) or (row['trend'] < 0) or (row['emotion'] > 0.8)

files = glob.glob(os.path.join(DATA_DIR, CSV_PATTERN))
summary_rows = []

for fp in files:
    code = os.path.basename(fp).split("_TET_full")[0]
    df = pd.read_csv(fp, parse_dates=['date']).sort_values('date')
    if df.empty: continue
    # ensure required columns exist
    required = ['date','trend','joint_trend','emotion','anchored','timing','close']
    if not all(c in df.columns for c in required):
        print(f"skip {code} missing cols")
        continue

    # apply rules row-wise - creates actions for all dates
    def decide(row):
        try:
            if BUY_RULE(row): return 'buy'
            if SELL_RULE(row): return 'sell'
            return 'hold'
        except Exception:
            return 'hold'

    df['action'] = df.apply(decide, axis=1)
    out_fp = os.path.join(OUT_DIR, f"{code}_signals.csv")
    df[['date','action','trend','joint_trend','emotion','anchored','timing','close']].to_csv(out_fp, index=False)
    # latest date summary
    last = df.iloc[-1]
    summary_rows.append({
        'code': code,
        'date': last['date'].strftime('%Y-%m-%d'),
        'action': last['action'],
        'trend': last['trend'],
        'joint_trend': last['joint_trend'],
        'emotion': last['emotion'],
        'anchored': last['anchored'],
        'timing': last['timing'],
        'close': last['close']
    })

# save daily summary
today = datetime.now().strftime("%Y%m%d")
summary_df = pd.DataFrame(summary_rows).sort_values(['action','timing'], ascending=[False, False])
summary_fp = os.path.join(OUT_DIR, f"daily_signals_{today}.csv")
summary_df.to_csv(summary_fp, index=False)
print("Saved signals to", OUT_DIR, "summary:", summary_fp)