# main/step_1/validate_step1_allinone.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import logging
import pandas as pd

# =============== TÌM PROJECT ROOT ===============
def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "config/.env").exists() and (cur / "main").exists() and (cur / "data").exists():
            return cur
        cur = cur.parent
    return start.resolve().parents[2]

PROJ_ROOT   = find_project_root(Path(__file__))
DATA_DIR    = PROJ_ROOT / "data"
UNIVERSE_DIR= DATA_DIR / "universe"
LOG_DIR     = PROJ_ROOT / "logs" / "realtime" / "validation"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =============== LOGGER ===============
def get_logger():
    lg = logging.getLogger("dq")
    lg.propagate = False
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
        sh = logging.StreamHandler(); sh.setFormatter(fmt); lg.addHandler(sh)
        fh = logging.FileHandler(LOG_DIR / f"{datetime.now():%Y-%m-%d}_dq.log", encoding="utf-8")
        fh.setFormatter(fmt); lg.addHandler(fh)
    return lg

log = get_logger()

# =============== HELPERS ===============
def read_universe_txt(fn: Path) -> list[str]:
    """Đọc file 1 dòng 'AAA','AAM',... hoặc AAA, AAM, ...; bỏ nháy, khoảng trắng, comment #"""
    if not fn.exists():
        return []
    text = fn.read_text(encoding="utf-8")
    text = (text.replace("’","'").replace("‘","'").replace("“",'"').replace("”",'"'))
    tokens = []
    for tok in text.replace("\n", ",").split(","):
        t = tok.strip()
        if not t or t.startswith("#"): continue
        if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
            t = t[1:-1].strip()
        if t: tokens.append(t.upper())
    return sorted(set(tokens))

def load_universe() -> dict[str, list[str]]:
    return {
        "HOSE":  read_universe_txt(UNIVERSE_DIR / "VNINDEX.txt"),
        "HNX":   read_universe_txt(UNIVERSE_DIR / "HNXIndex.txt"),
        "UPCOM": read_universe_txt(UNIVERSE_DIR / "UpcomIndex.txt"),
    }

def minutes_ok_15m(ts: pd.Series) -> pd.Series:
    return ts.dt.minute.isin({0,15,30,45})

def add_issue(issues, level, check, detail, ticker=None, timestamp=None, count=None):
    issues.append({
        "level": level, "check": check, "detail": detail,
        "ticker": ticker, "timestamp": timestamp, "count": count
    })

# =============== CORE VALIDATOR ===============
def validate_csv(csv_path: Path, by: str = "15m", max_days_15m: int = 92, require_exchange_col: bool = True) -> int:
    """
    Trả về số ERROR (dùng làm exit-code). Xuất báo cáo:
      - logs/realtime/validation/YYYY-MM-DD_dq_report.csv
      - ..._per_ticker_stats.csv
      - ..._missing_tickers.txt (nếu có)
    """
    issues = []
    if not csv_path.exists():
        add_issue(issues, "ERROR", "file_exists", f"Không tìm thấy file: {csv_path}")
        dump_report(issues)
        print_report_summary(issues)
        return 1

    log.info(f"== Validate: {csv_path.name} ==")
    df = pd.read_csv(csv_path)
    n0 = len(df)
    log.info(f"Loaded shape={df.shape}")

    # 1) Cột bắt buộc
    must_cols = {"ticker","timestamp","open","high","low","close","volume"}
    missing = [c for c in must_cols if c not in df.columns]
    if missing:
        add_issue(issues, "ERROR", "missing_columns", f"Thiếu cột bắt buộc: {missing}")

    # 2) Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        bad_ts = df["timestamp"].isna().sum()
        if bad_ts > 0:
            add_issue(issues, "ERROR", "timestamp_parse", f"{bad_ts} dòng timestamp không parse được")

    # 3) Ép numeric & Null rate
    num_cols = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    for c in num_cols:
        na = df[c].isna().mean()
        if na > 0.05:
            add_issue(issues, "WARN", "null_ratio", f"Cột {c} null ratio={na:.2%} (>5%)")
    for c in num_cols:  # ép numeric
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in num_cols:
        na = df[c].isna().sum()
        if na > 0:
            add_issue(issues, "WARN", "nan_numeric", f"Cột {c} có {na} giá trị không hợp lệ (NaN sau ép)")

    # 4) Trùng (ticker,timestamp)
    if {"ticker","timestamp"}.issubset(df.columns):
        dups = df.duplicated(subset=["ticker","timestamp"]).sum()
        if dups > 0:
            add_issue(issues, "ERROR", "duplicates", f"{dups} bản ghi trùng (ticker,timestamp)")

    # 5) Ràng buộc OHLC/volume
    if set(num_cols) >= {"open","high","low","close"}:
        bad_range = df.query("(open<0) or (high<0) or (low<0) or (close<0)").shape[0]
        if bad_range > 0:
            add_issue(issues, "ERROR", "negative_prices", f"{bad_range} dòng có giá âm")
        bad_hl = df.query("high < low").shape[0]
        if bad_hl > 0:
            add_issue(issues, "ERROR", "high_low_order", f"{bad_hl} dòng high<low")
        warn_h = df[(df["high"] < df[["open","close"]].max(axis=1))].shape[0]
        warn_l = df[(df["low"]  > df[["open","close"]].min(axis=1))].shape[0]
        if warn_h > 0:
            add_issue(issues, "WARN", "high_vs_oc", f"{warn_h} dòng: high < max(open,close)")
        if warn_l > 0:
            add_issue(issues, "WARN", "low_vs_oc", f"{warn_l} dòng: low > min(open,close)")
    if "volume" in df.columns:
        neg_vol = (df["volume"] < 0).sum()
        if neg_vol > 0:
            add_issue(issues, "ERROR", "negative_volume", f"{neg_vol} dòng volume âm")

    # 6) Khung thời gian 15m
    if by == "15m" and "timestamp" in df.columns:
        ok_min = minutes_ok_15m(df["timestamp"])
        bad_min = (~ok_min).sum()
        if bad_min > 0:
            add_issue(issues, "ERROR", "minute_bucket", f"{bad_min} dòng không rơi vào phút 0/15/30/45")
        earliest_allowed = (datetime.now().date() - timedelta(days=max(1, max_days_15m-1)))
        too_old = (df["timestamp"].dt.date < earliest_allowed).sum()
        if too_old > 0:
            add_issue(issues, "ERROR", "timeframe_limit", f"{too_old} dòng 15m cũ hơn {max_days_15m} ngày (earliest {earliest_allowed})")

    # 7) Exchange column & values
    if require_exchange_col:
        if "Exchange" not in df.columns:
            add_issue(issues, "WARN", "exchange_col", "Thiếu cột Exchange (HOSE/HNX/UPCOM)")
        else:
            bad_ex = (~df["Exchange"].isin(["HOSE","HNX","UPCOM"])).sum()
            if bad_ex > 0:
                add_issue(issues, "ERROR", "exchange_values", f"{bad_ex} dòng Exchange ngoài tập (HOSE/HNX/UPCOM)")

    # 8) Coverage so với universe
    uni = load_universe()
    all_uni = set(uni["HOSE"] + uni["HNX"] + uni["UPCOM"])
    if "ticker" in df.columns and all_uni:
        tickers_in_file = set(map(str.upper, df["ticker"].dropna().astype(str).tolist()))
        missing = sorted(all_uni - tickers_in_file)
        cov = len(tickers_in_file & all_uni) / max(1, len(all_uni))
        level = "WARN" if cov < 0.95 else "INFO"
        add_issue(issues, level, "universe_coverage",
                  f"Coverage={cov:.1%}, missing={len(missing)} tickers (ghi chi tiết ra file)", count=len(missing))
        if missing:
            miss_path = LOG_DIR / f"{datetime.now():%Y-%m-%d}_missing_tickers.txt"
            miss_path.write_text("\n".join(missing), encoding="utf-8")
            log.info(f"Missing tickers list -> {miss_path}")

    # 9) Thống kê theo mã
    if {"ticker","timestamp"}.issubset(df.columns):
        per_tk = df.groupby("ticker")["timestamp"].agg(["min","max","count"]).reset_index()
        per_tk_path = LOG_DIR / f"{datetime.now():%Y-%m-%d}_per_ticker_stats.csv"
        per_tk.to_csv(per_tk_path, index=False)
        log.info(f"Per-ticker stats -> {per_tk_path}")

    dump_report(issues)
    print_report_summary(issues)

    err_cnt = sum(1 for it in issues if it["level"] == "ERROR")
    log.info(f"Errors={err_cnt}, Warnings={sum(1 for it in issues if it['level']=='WARN')}, Rows={n0}")
    return err_cnt

# =============== REPORTERS ===============
def dump_report(issues: list[dict]):
    if not issues:
        issues = [{"level":"INFO","check":"all_good","detail":"No issues found","ticker":None,"timestamp":None,"count":None}]
    rep = pd.DataFrame(issues)
    out = LOG_DIR / f"{datetime.now():%Y-%m-%d}_dq_report.csv"
    rep.to_csv(out, index=False)
    log.info(f"Report -> {out}")

def print_report_summary(issues: list[dict]):
    if not issues:
        print("✓ No issues found.")
        return
    df = pd.DataFrame(issues)
    summary = df.groupby(["level","check"]).size().reset_index(name="count").sort_values(["level","count"], ascending=[True,False])
    print("\n=== DQ Summary ===")
    print(summary.to_string(index=False))

# =============== MAIN (1 FILE CHẠY THẲNG) ===============
def build_config():
    # Mặc định: chỉ bấm Run là dùng được
    cfg = {
        "csv": str(DATA_DIR / "step_1" / "raw_stocks_15m.csv"),
        "by": "15m",
        "require_exchange_col": True,
        "max_days_15m": int(os.getenv("FIIN_15M_MAX_DAYS", "92")),
    }
    # Cho phép override qua ENV (tuỳ chọn)
    cfg["csv"] = os.getenv("DQ_CSV", cfg["csv"])
    cfg["by"]  = os.getenv("DQ_BY",  cfg["by"])
    _env_req_ex = os.getenv("DQ_REQUIRE_EXCHANGE")
    if _env_req_ex is not None:
        cfg["require_exchange_col"] = _env_req_ex.strip().lower() in {"1","true","yes","y"}
    _env_days = os.getenv("DQ_MAX_DAYS_15M")
    if _env_days and _env_days.isdigit():
        cfg["max_days_15m"] = int(_env_days)
    return cfg

def parse_cli(cfg: dict) -> dict:
    # CLI là TUỲ CHỌN — nếu không truyền tham số thì vẫn chạy bằng defaults ở trên
    ap = argparse.ArgumentParser("Validate Step-1 crawled data (all-in-one)")
    ap.add_argument("--csv", help="Đường dẫn file CSV cần kiểm tra")
    ap.add_argument("--by", choices=["15m","1d"], help="Khung dữ liệu (ảnh hưởng check phút & timeframe)")
    ap.add_argument("--max_days_15m", type=int, help="Giới hạn ngày gần nhất cho 15m")
    ap.add_argument("--no_require_exchange", action="store_true", help="Không bắt buộc có cột Exchange")
    args, _ = ap.parse_known_args()

    if args.csv: cfg["csv"] = args.csv
    if args.by: cfg["by"] = args.by
    if args.max_days_15m: cfg["max_days_15m"] = args.max_days_15m
    if args.no_require_exchange: cfg["require_exchange_col"] = False
    return cfg

def main():
    cfg = build_config()
    cfg = parse_cli(cfg)  # vẫn chạy ok nếu bạn không truyền gì

    csv_path = Path(cfg["csv"])
    print(f"[DQ] Project root : {PROJ_ROOT}")
    print(f"[DQ] CSV to check : {csv_path}")
    print(f"[DQ] Frame (by)   : {cfg['by']}")
    print(f"[DQ] Require EXCH : {cfg['require_exchange_col']}")
    print(f"[DQ] 15m max days : {cfg['max_days_15m']}")

    err = validate_csv(
        csv_path,
        by=cfg["by"],
        max_days_15m=cfg["max_days_15m"],
        require_exchange_col=cfg["require_exchange_col"],
    )
    sys.exit(1 if err > 0 else 0)

if __name__ == "__main__":
    main()
