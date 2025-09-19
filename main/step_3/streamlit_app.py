import time
from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_DIR  = PROJECT_ROOT / "data" / "step2_signals"
LOG_DIR = PROJECT_ROOT / "logs"

st.set_page_config(page_title="EOD Alerts", layout="wide")

@st.cache_data(ttl=30)
def load_latest():
    cands = sorted(IN_DIR.glob("signals_from_merged_full_*.csv"))
    if not cands:
        return None
    df = pd.read_csv(cands[-1])

    # chuẩn cột signal
    sig_col = "final_signal" if "final_signal" in df.columns else ("rule_signal" if "rule_signal" in df.columns else "signal")
    if sig_col not in df.columns:
        df["final_signal"] = None
        sig_col = "final_signal"
    df["signal_col"] = df[sig_col]

    # NEW: nếu trống/null/chuỗi rỗng -> gán "BUY"
    s = df["signal_col"]
    df["signal_col"] = s.where(~(s.isna() | (s.astype(str).str.strip() == "")), "BUY")

    return df

df = load_latest()
st.title("EOD Signals Dashboard")

if df is None or df.empty:
    st.warning("Chưa có dữ liệu Step 2.")
    st.stop()

# Bộ lọc
signals = ["BUY", "SELL"]
selected = st.multiselect("Lọc tín hiệu", options=signals, default=signals)

latest = (df.sort_values(["ticker","timestamp"])
            .groupby("ticker", as_index=False).tail(1))

if selected:
    latest = latest[latest["signal_col"].isin(selected)]

# Bảng tổng quan
cols = [c for c in ["ticker","timestamp","close","rsi14","macd","macd_signal","sma20","ema20","ema50","of_ratio","pe","pb","roe","debt_to_equity","signal_col"] if c in latest.columns]
st.dataframe(latest[cols].sort_values("ticker").reset_index(drop=True), use_container_width=True)

# Chi tiết theo ticker
tickers = latest["ticker"].unique().tolist()
pick = st.selectbox("Xem lịch sử 1 mã", tickers)
hist = df[df["ticker"] == pick].sort_values("timestamp")
st.line_chart(hist.set_index("timestamp")[["close"]])

st.caption("Nguồn: Step 2 (signals_eod_full_*.csv). Làm mới mỗi 30 giây.")
