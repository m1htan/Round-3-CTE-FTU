import os
import pandas as pd
from datetime import datetime

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def append_df_csv(df: pd.DataFrame, cache_dir: str, prefix: str):
    if df is None or df.empty:
        return
    ensure_dir(cache_dir)
    # ghi theo ngày để file không quá lớn
    date_str = datetime.now().strftime("%Y%m%d")
    out = os.path.join(cache_dir, f"{prefix}_{date_str}.csv")
    # mode append, header chỉ khi file chưa có
    header = not os.path.exists(out)
    df.to_csv(out, mode="a", header=header, index=False)
