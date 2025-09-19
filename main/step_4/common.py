import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "step4"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def setup_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def latest_file(glob_pattern: str, base: Path) -> Path | None:
    cands = sorted(base.glob(glob_pattern))
    return cands[-1] if cands else None

def normalize_signals_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hoá cột cơ bản: ticker, timestamp(ms), close, signal.
    Ưu tiên final_signal > rule_signal > signal.
    """
    out = df.copy()

    # ticker
    if "ticker" not in out.columns:
        # Một số pipeline ghi "symbol" thay vì "ticker"
        if "symbol" in out.columns:
            out["ticker"] = out["symbol"]
        else:
            raise ValueError("Thiếu cột 'ticker' hoặc 'symbol' trong file input.")
    out["ticker"] = out["ticker"].astype(str).str.upper()

    # timestamp
    if "timestamp" in out.columns:
        ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        # nếu đã là int ms thì parse sẽ NaT; fallback:
        if ts.isna().mean() > 0.5:
            # thử coi timestamp là ms số
            out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").astype("Int64")
        else:
            out["timestamp"] = ((ts.view("int64") // 10**6)).astype("Int64")
    elif "date" in out.columns:
        ts = pd.to_datetime(out["date"], utc=True, errors="coerce")
        out["timestamp"] = ((ts.view("int64") // 10**6)).astype("Int64")
    else:
        raise ValueError("Thiếu cột thời gian 'timestamp' hoặc 'date'.")

    # close
    if "close" not in out.columns:
        raise ValueError("Thiếu cột 'close' trong file input.")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    # chọn cột signal
    sig_col = None
    for c in ("final_signal", "rule_signal", "signal", "fi_rule"):
        if c in out.columns:
            sig_col = c
            break
    if sig_col is None:
        out["final_signal"] = pd.NA
        sig_col = "final_signal"

    out["entry_signal"] = out[sig_col].astype("string")
    return out

def ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["ticker", "timestamp"] if c in df.columns]
    return df.sort_values(cols).reset_index(drop=True)

def pct_rank_xs_in_universe(s: pd.Series, by_date: pd.Series, mask: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=s.index, dtype=float)
    s_u = s[mask]
    by_u = by_date[mask]
    out.loc[mask] = s_u.groupby(by_u).rank(pct=True)
    return out

def as_date_str(ms: int | float | pd.Series) -> pd.Series:
    ts = pd.to_datetime(ms, unit="ms", utc=True, errors="coerce")
    return ts.dt.tz_localize(None).dt.date.astype("string")

def today_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d")
