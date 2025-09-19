import os, time, argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from FiinQuantX import FiinSession
from utils_universe import load_universe

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "step1_eod"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = ['open','high','low','close','volume','bu','sd','fb','fs','fn']

def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def split_batches(universe, max_size=92):
    for i in range(0, len(universe), max_size):
        yield universe[i:i+max_size]

def discover_valid_tickers(client, all_tickers, by="1d", fields=None, batch=400, sleep_sec=0.2):
    fields = fields or FIELDS
    valid = set()
    for grp in chunked(all_tickers, batch):
        try:
            ev = client.Fetch_Trading_Data(
                realtime=False, tickers=list(grp), fields=fields,
                adjusted=True, by=by, period=1, lasted=True
            )
            df = ev.get_data()
            if isinstance(df, pd.DataFrame) and "ticker" in df.columns:
                valid.update(df["ticker"].astype(str).str.upper().unique().tolist())
        except Exception:
            for sub in chunked(grp, max(50, batch//4)):
                try:
                    ev = client.Fetch_Trading_Data(
                        realtime=False, tickers=list(sub), fields=fields,
                        adjusted=True, by=by, period=1, lasted=True
                    )
                    df = ev.get_data()
                    if isinstance(df, pd.DataFrame) and "ticker" in df.columns:
                        valid.update(df["ticker"].astype(str).str.upper().unique().tolist())
                except Exception:
                    pass
        time.sleep(sleep_sec)
    return sorted(valid)

def normalize_eod_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker","timestamp","open","high","low","close","volume","bu","sd","fb","fs","fn"])
    df = df.copy()

    # tìm cột thời gian phổ biến
    dt_col = None
    for c in ["t", "time", "date", "tradingDate", "trading_date", "timestamp"]:
        if c in df.columns:
            dt_col = c
            break

    if dt_col is not None:
        dt = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
    else:
        dt = pd.Series(pd.NaT, index=df.index)

    # epoch ms dạng Int64
    ts_ms = (dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    df["timestamp"] = ts_ms.astype("Int64")

    # ticker & numeric
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()
    for c in ["open","high","low","close","volume","bu","sd","fb","fs","fn"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = pd.NA

    # drop dòng thiếu timestamp/ticker, dedup
    df = df.dropna(subset=["ticker","timestamp"])
    df = (df.sort_values(["ticker","timestamp"])
            .drop_duplicates(["ticker","timestamp"], keep="last"))

    return df[["ticker","timestamp","open","high","low","close","volume","bu","sd","fb","fs","fn"]]

def fetch_eod_history(client, tickers, fields=None, by="1d", from_date=None, to_date=None, period=750, lasted=False):
    if fields is None:
        fields = FIELDS
    ev = client.Fetch_Trading_Data(
        realtime=False,
        tickers=tickers,
        fields=fields,
        adjusted=True,
        by=by,
        **({"from_date": from_date, "to_date": to_date} if from_date and to_date else {"period": period}),
        lasted=lasted
    )
    return ev.get_data()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers-limit", type=int, default=92, help="Số mã tối đa/lô (quota)")
    parser.add_argument("--period", type=int, default=800, help="Số nến EOD (≈3 năm)")
    parser.add_argument("--universe-path", default=".", help="Đường dẫn file universe (utils_universe sẽ đọc)")
    args = parser.parse_args()

    load_dotenv(dotenv_path=PROJECT_ROOT / "config" / ".env")
    USERNAME = os.getenv("FIINQUANT_USERNAME")
    PASSWORD = os.getenv("FIINQUANT_PASSWORD")
    if not USERNAME or not PASSWORD:
        raise RuntimeError("Thiếu FIINQUANT_USERNAME/FIINQUANT_PASSWORD trong config/.env")

    # 1) universe & quota
    universe_all = load_universe(args.universe_path)
    print(f"[init] Universe size (raw) = {len(universe_all)}")
    universe = universe_all[:args.tickers_limit]
    print(f"[init] Universe size (limited) = {len(universe)}")

    # 2) login
    client = FiinSession(username=USERNAME, password=PASSWORD).login()

    # 3) lọc các mã hợp lệ cho by=1d
    print("[init] Discovering valid tickers for by=1d")
    cache = PROJECT_ROOT / "data" / "universe" / "valid_1d.txt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        valid = [x.strip().upper() for x in cache.read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        valid = discover_valid_tickers(client, universe, by="1d", fields=FIELDS, batch=400, sleep_sec=0.2)
        cache.write_text("\n".join(valid), encoding="utf-8")
    print(f"[init] Valid universe size = {len(valid)}")

    # 4) fetch EOD theo batch <= 92
    frames = []
    for i, batch in enumerate(split_batches(valid, 92), start=1):
        print(f"[batch-{i}] fetching {len(batch)} tickers (EOD)")
        try:
            df_raw = fetch_eod_history(client, batch, fields=FIELDS, by="1d", period=args.period, lasted=False)
            df_norm = normalize_eod_df(df_raw)
            if not df_norm.empty:
                frames.append(df_norm)
            time.sleep(0.2)
        except Exception as e:
            print(f"[batch-{i}] error: {e}")

    if not frames:
        print("[eod] Không lấy được dữ liệu.")
        return

    # 5) ghi CSV 1 file/ngày
    df = pd.concat(frames, ignore_index=True)
    day_str = datetime.utcnow().strftime("%Y%m%d")
    csv_path = OUT_DIR / f"bars_eod_{day_str}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[eod] saved: {csv_path} | rows={len(df)} | tickers≈{df['ticker'].nunique()}")

if __name__ == "__main__":
    main()
