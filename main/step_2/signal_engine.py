import pandas as pd
import numpy as np
from typing import Dict, Any

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def attach_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: cột bắt buộc ['ticker','timestamp','open','high','low','close','volume','bu','sd']
    """
    if df.empty:
        return df

    df = df.sort_values(["ticker", "timestamp"]).copy()
    df["ema20"] = df.groupby("ticker")["close"].transform(lambda s: _ema(s, 20))
    df["ema50"] = df.groupby("ticker")["close"].transform(lambda s: _ema(s, 50))
    df["rsi14"] = df.groupby("ticker")["close"].transform(lambda s: _rsi(s, 14))

    # Order-flow ratio (mua chủ động / bán chủ động) để lấy “nhịp”
    df["of_ratio"] = (df["bu"].replace(0, np.nan)) / (df["sd"].replace(0, np.nan))

    return df

def generate_signals(df: pd.DataFrame,
                     params: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Trả về df với cột 'signal' ∈ {'BUY','SELL',None}
    Rule ví dụ (placeholder):
      - BUY: close cắt lên ema20 AND of_ratio >= 1.2 AND rsi14 > 50
      - SELL: close cắt xuống ema20 OR rsi14 < 45
    """
    if params is None:
        params = {
            "of_min": 1.2,
            "rsi_buy": 50,
            "rsi_sell": 45
        }
    if df.empty:
        df["signal"] = None
        return df

    df = df.sort_values(["ticker", "timestamp"]).copy()
    # Cross detection
    df["prev_close"] = df.groupby("ticker")["close"].shift(1)
    df["prev_ema20"] = df.groupby("ticker")["ema20"].shift(1)

    cross_up = (df["prev_close"] <= df["prev_ema20"]) & (df["close"] > df["ema20"])
    cross_down = (df["prev_close"] >= df["prev_ema20"]) & (df["close"] < df["ema20"])

    buy = cross_up & (df["of_ratio"] >= params["of_min"]) & (df["rsi14"] > params["rsi_buy"])
    sell = cross_down | (df["rsi14"] < params["rsi_sell"])

    df["signal"] = np.where(buy, "BUY", np.where(sell, "SELL", None))
    return df[[
        "ticker","timestamp","open","high","low","close","volume","bu","sd",
        "ema20","ema50","rsi14","of_ratio","signal"
    ]].copy()
