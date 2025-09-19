# main/step_2/step_2_from_merged.py

import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_DIR  = PROJECT_ROOT / "data" / "round2" / "step_3"   # nơi có step_3_merged.csv
OUT_DIR = PROJECT_ROOT / "data" / "step2_signals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- helpers ----------
def _to_ms(ts_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(ts_series, errors="coerce", utc=True)
    return ((dt.astype("int64") // 10**6)).astype("Int64")

def cross_up(x: pd.Series, y: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    return (x > y) & (x.shift(1) <= y.shift(1))

def cross_down(x: pd.Series, y: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    return (x < y) & (x.shift(1) >= y.shift(1))

def ensure_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out

def _pick_series(df, *candidates, default=np.nan):
    for c in candidates:
        if c in df.columns:
            val = df[c]
            if isinstance(val, pd.DataFrame):
                val = val.iloc[:, 0]  # nếu trùng tên -> lấy cột đầu
            return pd.to_numeric(val, errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")

# -------- rule layers ----------
def apply_handmade_rules(df: pd.DataFrame) -> pd.DataFrame:
    need = ["ticker","timestamp","close","rsi14","macd","macd_signal","sma20"]
    for c in need:
        if c not in df.columns:
            df[c] = np.nan  # CHỈ thêm cho cột kỹ thuật (ticker đã có thật ở bước 1)

    def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestamp")
        # đảm bảo cột ticker tồn tại trong group
        g["ticker"] = g["ticker"].iloc[0]

        rsi  = pd.to_numeric(g["rsi14"], errors="coerce")
        macd = pd.to_numeric(g["macd"], errors="coerce")
        sig  = pd.to_numeric(g["macd_signal"], errors="coerce")
        close= pd.to_numeric(g["close"], errors="coerce")
        sma20= pd.to_numeric(g["sma20"], errors="coerce")

        buy  = (rsi < 30) & ((macd > sig) & (macd.shift(1) <= sig.shift(1)))
        sell = (rsi > 70) | ((close < sma20) & (close.shift(1) >= sma20.shift(1)))

        g["rule_signal"] = pd.Series(pd.NA, index=g.index, dtype="object")
        g.loc[buy,  "rule_signal"] = "BUY"
        g.loc[sell, "rule_signal"] = "SELL"
        return g

    # LƯU Ý: không dùng include_groups=..., và reset_index để chắc chắn giữ cột
    out = df.groupby("ticker", group_keys=False).apply(_per_ticker)
    return out.reset_index(drop=True)

def apply_fi_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # prefer filled -> ttm -> raw; also accept the lower-case names after renaming
    pe  = _pick_series(out, "PE_filled", "PE_TTM", "PE", "pe")
    pb  = _pick_series(out, "PB_filled", "PB", "pb")
    roe = _pick_series(out, "ROE", "roe")
    dte = _pick_series(out, "debt_to_equity", "Debt_to_Equity", "DebtToEquity", "dte")

    # TI đã tính sẵn trong file (đổi tên cho khớp nếu cần)
    rsi  = _pick_series(out, "rsi_14", "RSI_14", "rsi14")
    macd = _pick_series(out, "macd_12_26", "MACD_12_26", "macd")
    macd_sig = _pick_series(out, "macd_signal_12_26_9", "MACD_signal_12_26_9", "macd_signal")
    close = _pick_series(out, "close")
    sma20 = _pick_series(out, "sma_20", "SMA_20", "sma20")

    # TI buy/sell cơ bản
    ti_buy  = (rsi < 35) & ((macd > macd_sig) & (macd.shift(1) <= macd_sig.shift(1)))
    ti_sell = (rsi > 70) | ((close < sma20) & (close.shift(1) >= sma20.shift(1)))

    # FI điều kiện
    pe_ok  = (pe > 0) & (pe <= 30)
    pb_ok  = (pb > 0) & (pb <= 5)
    roe_ok = (roe >= 0.10)
    # dte NaN coi như “không vi phạm” nếu bạn muốn nới lỏng:
    dte_ok = (dte <= 2.0) | dte.isna()

    fi_ok = pe_ok & pb_ok & roe_ok & dte_ok

    # momentum nhẹ (nếu có rank sẵn)
    if "rank_composite" in out.columns:
        mom_ok = pd.to_numeric(out["rank_composite"], errors="coerce") <= 50  # e.g., top 50 each day
    else:
        mom_ok = pd.Series([True] * len(out), index=out.index)

    fi_buy  = ti_buy & fi_ok & mom_ok
    fi_sell = ti_sell | (~fi_ok)

    out["fi_rule"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out.loc[fi_buy,  "fi_rule"] = "BUY"
    out.loc[fi_sell, "fi_rule"] = "SELL"
    return out

# -------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv",
                    help="Đường dẫn step_3_merged.csv. Nếu bỏ trống sẽ tự chọn file mới nhất trong data/round2/step_3/")
    ap.add_argument("--emit-latest-only", action="store_true",
                    help="Chỉ xuất tín hiệu ở nến mới nhất mỗi mã.")
    args = ap.parse_args()

    # 1) đọc input
    if args.input_csv:
        input_csv = Path(args.input_csv)
    else:
        cands = sorted(IN_DIR.glob("step_3_merged*.csv"))
        if not cands:
            raise FileNotFoundError(f"Không tìm thấy step_3_merged*.csv trong {IN_DIR}")
        input_csv = cands[-1]
    print(f"[step2] reading: {input_csv}")

    df0 = pd.read_csv(input_csv)

    # --- dedupe column names (giữ cột đầu tiên) ---
    if df0.columns.duplicated().any():
        print("[warn] duplicated columns found, dropping duplicates (keep first)")
        df0 = df0.loc[:, ~df0.columns.duplicated(keep="first")]

    # --- try to get ticker from multiple candidates ---
    ticker_candidates = ["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYM"]
    found = None
    for c in ticker_candidates:
        if c in df0.columns:
            found = c
            break

    if not found:
        raise ValueError("Input không có cột ticker/symbol. Hãy sửa Step 3 để merge kèm ticker.")

    # Chuẩn: tạo cột 'ticker' upper-case, không được toàn NaN/rỗng
    df0["ticker"] = df0[found].astype(str).str.upper().str.strip()
    if df0["ticker"].eq("").all():
        raise ValueError("Cột ticker rỗng. Kiểm tra step_3_merged.csv.")

    # --- chuẩn hoá timestamp -> epoch ms ---
    if "timestamp" in df0.columns:
        dt = pd.to_datetime(df0["timestamp"], errors="coerce", utc=True)
    elif "date" in df0.columns:
        dt = pd.to_datetime(df0["date"], errors="coerce", utc=True)
    else:
        raise ValueError("Thiếu cột timestamp/date trong input.")

    df0["timestamp"] = (dt.astype("int64") // 10 ** 6).astype("Int64")
    df = df0  # dùng df cho các bước sau

    # 2) chuẩn hoá tên cột & timestamp
    # map TI
    rename_map = {
        "rsi_14": "rsi14",
        "macd_12_26": "macd",
        "macd_signal_12_26_9": "macd_signal",
        "sma_20": "sma20",
        "sma_50": "sma50",
        "ema_12": "ema12",
        "ema_26": "ema26",
    }
    # map FI
    fi_rename = {
        "PE_TTM": "pe",          # ưu tiên PE_TTM/PE_filled -> pe
        "PE_filled": "pe",
        "PB": "pb",
        "PB_filled": "pb",
        "ROE": "roe",
        "EPS_TTM": "eps_ttm",
        "valuation_pref": "valuation_pref",
        "valuation_pref_metric": "valuation_metric"
    }

    df = df0.rename(columns={**rename_map, **fi_rename}).copy()

    # timestamp trong merged là dạng yyyy-mm-dd → đổi sang epoch ms
    if "timestamp" in df.columns:
        df["timestamp"] = _to_ms(df["timestamp"])
    elif "date" in df.columns:
        df["timestamp"] = _to_ms(df["date"])
    else:
        raise ValueError("Thiếu cột thời gian (timestamp/date) trong step_3_merged.csv")

    # ticker upper
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()
    else:
        raise ValueError("Thiếu cột ticker trong step_3_merged.csv")

    # 3) áp rules
    ruled1 = apply_handmade_rules(df)
    ruled2 = apply_fi_rules(ruled1)

    # 4) gộp tín hiệu: ưu tiên SELL > BUY; ưu tiên fi_rule > rule_signal
    def _combine(row):
        for key in ("fi_rule", "rule_signal", "signal"):
            v = row.get(key)
            if isinstance(v, str) and v:
                return v
        return pd.NA

    ruled2["final_signal"] = ruled2.apply(_combine, axis=1)

    # 5) lưu
    day_str = datetime.utcnow().strftime("%Y%m%d")
    out_all = OUT_DIR / f"signals_from_merged_full_{day_str}.csv"
    ruled2.to_csv(out_all, index=False)
    print(f"[step2] saved all signals: {out_all}")

    for c in ("ticker", "timestamp"):
        if c not in ruled2.columns:
            raise ValueError(f"Expected column '{c}' missing after rules. Check input & renaming.")

    assert "ticker" in ruled2.columns, "Thiếu cột ticker sau apply rules"
    assert "timestamp" in ruled2.columns, "Thiếu cột timestamp sau apply rules"

    latest = (ruled2.sort_values(["ticker","timestamp"]).groupby("ticker").tail(1))
    alerts = latest[latest["final_signal"].notna()].copy()
    out_alerts = OUT_DIR / f"alerts_from_merged_{day_str}.csv"
    alerts.to_csv(out_alerts, index=False)
    print(f"[step2] saved alerts(latest): {out_alerts}")

    if args.emit_latest_only:
        if alerts.empty:
            print("[step2] No alerts.")
        else:
            print("\n=== Alerts (latest per ticker) ===")
            cols_show = ["ticker","timestamp","close","rsi14","macd","macd_signal","sma20",
                         "pe","pb","roe","final_signal"]
            cols_show = [c for c in cols_show if c in alerts.columns]
            for _, r in alerts[cols_show].iterrows():
                row = {k: (None if pd.isna(r[k]) else r[k]) for k in cols_show}
                print(row)

if __name__ == "__main__":
    main()