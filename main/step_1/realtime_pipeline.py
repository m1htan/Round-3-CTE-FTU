import math
import os
import time
import json
import signal
import queue
import argparse
import threading
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from FiinQuantX import FiinSession, BarDataUpdate

from utils_universe import load_universe
from signal_engine import attach_indicators, generate_signals

# -------- Settings ----------
DEFAULT_TIMEFRAME = {
    "intraday_15m": "15m",
    "eod": "1d"
}
CHUNK_SIZE = 60      # Số mã/stream (tùy thực tế; tăng/giảm nếu cần)
BATCH_FLUSH_SEC = 5   # Flush queue -> disk mỗi N giây

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]         # main/step_1/realtime_pipeline.py -> lên 2 cấp = gốc dự án
OUT_DIR = PROJECT_ROOT / "data" / "realtime"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# graceful stop flag
_stop_all = False

def _install_signal_handlers():
    def _handler(signum, frame):
        global _stop_all
        _stop_all = True
        print(f"\n[!] Received signal {signum}. Stopping gracefully...")
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, _handler)

def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

class RealtimeWorker:
    """
    Một worker cho 1 nhóm tickers (<= CHUNK_SIZE).
    - Mở Fetch_Trading_Data(realtime=True) với by=...
    - Nhận callback BarDataUpdate -> put vào queue dùng chung
    """
    def __init__(self, client, tickers, by, fields, wait_full_tf=False, from_date=None, period=None, shared_queue=None, name="worker"):
        self.client = client
        self.tickers = tickers
        self.by = by
        self.fields = fields
        self.wait_full_tf = wait_full_tf
        self.from_date = from_date
        self.period = period
        self.shared_queue = shared_queue or queue.Queue()
        self.name = name
        self._event = None
        self._thread = None

    def _on_update(self, data: BarDataUpdate):
        try:
            df = data.to_dataFrame()
            expected = ["ticker", "timestamp", "open", "high", "low", "close", "volume", "bu", "sd", "fb", "fs", "fn"]
            for col in expected:
                if col not in df.columns:
                    df[col] = pd.NA
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
            df["ingest_ts"] = int(time.time() * 1000)

            payload = df[["ticker", "timestamp", "open", "high", "low", "close", "volume", "bu", "sd", "fb", "fs", "fn",
                          "ingest_ts"]]
            try:
                self.shared_queue.put(payload, timeout=0.05)
            except queue.Full:
                # rơi batch này để giữ realtime
                pass
        except Exception as e:
            print(f"[{self.name}] Callback error: {e}")

    def start(self):
        print(f"[{self.name}] Starting stream for {len(self.tickers)} tickers, by={self.by}")
        self._event = self.client.Fetch_Trading_Data(
            realtime=True,
            tickers=self.tickers,
            fields=self.fields,
            adjusted=True,
            callback=self._on_update,
            by=self.by,
            **({"from_date": self.from_date} if self.from_date else {}),
            **({"period": self.period} if self.period else {}),
            wait_for_full_timeFrame=self.wait_full_tf
        )
        # get_data() sẽ block và pump dữ liệu; chạy ở thread riêng
        self._thread = threading.Thread(target=self._event.get_data, name=self.name, daemon=True)
        self._thread.start()

    def stop(self):
        if self._event:
            try:
                self._event.stop()
            except Exception as e:
                print(f"[{self.name}] stop() error: {e}")

def discover_valid_tickers(client, all_tickers, by="15m", fields=None,
                           batch=90, sleep_sec=0.5, quota_limit=90):
    """
    Dò các mã hợp lệ cho timeframe `by` bằng cách gọi lịch sử 1 nến.
    Mọi request luôn <= quota_limit (mặc định 92).
    """
    fields = fields or ['open','high','low','close','volume','bu','sd','fb','fs','fn']
    valid = set()

    safe_batch = max(1, min(batch, quota_limit))  # cap theo quota
    for grp in chunked(all_tickers, safe_batch):
        try:
            ev = client.Fetch_Trading_Data(
                realtime=False,
                tickers=list(grp),
                fields=fields,
                adjusted=True,
                by=by,
                period=1,
                lasted=True
            )
            df = ev.get_data()
            if isinstance(df, pd.DataFrame) and "ticker" in df.columns:
                valid.update(df["ticker"].astype(str).str.upper().unique().tolist())
        except Exception:
            # fallback: chia nhỏ hơn nhưng vẫn ≤ quota
            fallback_batch = min(50, quota_limit)
            for subgrp in chunked(grp, fallback_batch):
                try:
                    ev = client.Fetch_Trading_Data(
                        realtime=False,
                        tickers=list(subgrp),
                        fields=fields,
                        adjusted=True,
                        by=by,
                        period=1,
                        lasted=True
                    )
                    df = ev.get_data()
                    if isinstance(df, pd.DataFrame) and "ticker" in df.columns:
                        valid.update(df["ticker"].astype(str).str.upper().unique().tolist())
                except Exception:
                    pass
        time.sleep(sleep_sec)

    return sorted(valid)

def split_batches(universe, max_size=90):
    for i in range(0, len(universe), max_size):
        yield universe[i:i+max_size]

def run_pipeline(mode: str, lasted: bool, max_workers: int = 2,
                 tickers=None, max_duration_sec=None, tickers_limit=92):
    global _stop_all
    _stop_all = False  # reset cho batch mới

    load_dotenv(dotenv_path=PROJECT_ROOT / "config" / ".env")

    USERNAME = os.getenv("FIINQUANT_USERNAME")
    PASSWORD = os.getenv("FIINQUANT_PASSWORD")
    if not USERNAME or not PASSWORD:
        raise RuntimeError("Thiếu FiinQuantX credential trong config/.env (FIINQUANT_USERNAME, F IQ UANT_PASSWORD)")

    # 1) Universe cho batch hiện tại
    if tickers is not None:
        universe = list(tickers)
    else:
        universe = load_universe(".")[:tickers_limit]
    print(f"[init] Universe size = {len(universe)}")

    chunk_size = max(1, math.ceil(len(universe) / max_workers))
    print(f"[init] Using max_workers={max_workers}, chunk_size={chunk_size}")

    def try_start_workers(tickers, by, fields, max_workers):
        while max_workers >= 1:
            chunk_size = max(1, math.ceil(len(tickers) / max_workers))
            print(f"[init] Using max_workers={max_workers}, chunk_size={chunk_size}")

            shared_q = queue.Queue(maxsize=10000)
            workers = []
            ok = True
            for idx, group in enumerate(chunked(tickers, chunk_size), start=1):
                w = RealtimeWorker(
                    client=client,
                    tickers=group,
                    by=by,
                    fields=fields,
                    wait_full_tf=False if mode == "intraday_15m" else True,
                    period=150 if mode == "intraday_15m" else 200,
                    shared_queue=shared_q,
                    name=f"worker-{idx}"
                )
                try:
                    w.start()
                    workers.append(w)
                    time.sleep(0.8)  # throttle
                except RuntimeError as e:
                    print(f"[init] Failed to start {w.name}: {e}")
                    ok = False
                    break

            if ok:
                return workers, shared_q
            # stop những worker đã start trước khi giảm worker count
            for w in workers:
                w.stop()
                if w._thread:
                    w._thread.join(timeout=2)
            max_workers -= 1
            print(f"[init] Retry with fewer workers: {max_workers}")

        raise RuntimeError("Không thể khởi tạo kết nối realtime (thread limit).")

    # 2) Chọn timeframe
    if mode not in DEFAULT_TIMEFRAME:
        raise ValueError("mode phải là 'intraday_15m' hoặc 'eod'")
    by = DEFAULT_TIMEFRAME[mode]

    # 3) Login
    client = FiinSession(username=USERNAME, password=PASSWORD).login()

    # 4) Fields cần lấy
    fields = ['open', 'high', 'low', 'close', 'volume', 'bu', 'sd', 'fb', 'fs', 'fn']

    by = DEFAULT_TIMEFRAME[mode]
    print("[init] Discovering valid tickers for by =", by)
    valid_tickers_cache = PROJECT_ROOT / "data" / "universe" / f"valid_{by}.txt"
    valid_tickers_cache.parent.mkdir(parents=True, exist_ok=True)
    if valid_tickers_cache.exists():
        with valid_tickers_cache.open("r", encoding="utf-8") as f:
            all_valid = [x.strip().upper() for x in f if x.strip()]
        # chỉ lấy những mã trong batch hiện tại
        valid_universe = [t for t in universe if t in all_valid]
    else:
        # CHỈ dò trên universe của batch
        valid_universe = discover_valid_tickers(
            client, universe, by=by, fields=fields,
            batch=min(len(universe), tickers_limit),
            sleep_sec=0.3,
            quota_limit=tickers_limit
        )
        valid_tickers_cache.write_text("\n".join(valid_universe), encoding="utf-8")

    print(f"[init] Valid universe size = {len(valid_universe)}")

    # 5) Khởi tạo workers theo batch tickers
    workers, shared_q = try_start_workers(valid_universe, by, fields, max_workers)

    # 6) Thread flusher: gom queue -> file, gắn chỉ báo & tín hiệu
    last_flush = 0
    buf = []

    _install_signal_handlers()
    t0 = time.time()
    print("[init] Streaming... Press Ctrl+C to stop.")

    while not _stop_all:
        try:
            try:
                item = shared_q.get(timeout=1.0)
                buf.append(item)
            except queue.Empty:
                pass

            now = time.time()
            if (now - last_flush) >= BATCH_FLUSH_SEC and buf:
                df = pd.concat(buf, ignore_index=True)
                buf.clear()

                # Lưu raw append (parquet rolling by date)
                day_str = datetime.utcnow().strftime("%Y%m%d")
                csv_path = OUT_DIR / f"bars_raw_{mode}_{day_str}.csv"
                df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
                raw_path = OUT_DIR / f"bars_raw_{mode}_{day_str}.parquet"

                if raw_path.exists():
                    # append bằng csv cũng được; parquet cần cầu kỳ hơn -> fallback csv an toàn:
                    csv_path = OUT_DIR / f"bars_raw_{mode}_{day_str}.csv"
                    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
                else:
                    df.to_parquet(raw_path)

                # Gắn chỉ báo + tín hiệu
                enriched = attach_indicators(df)
                signaled = generate_signals(enriched)

                # Chỉ log các tín hiệu phát sinh (BUY/SELL != None) ở tick gần nhất mỗi ticker
                latest = (signaled.sort_values(["ticker","timestamp"])
                                   .groupby("ticker").tail(1))
                alerts = latest[latest["signal"].notna()].copy()
                if not alerts.empty:
                    alerts_path = OUT_DIR / f"signals_{mode}_{day_str}.csv"
                    alerts.to_csv(alerts_path, mode="a", header=not alerts_path.exists(), index=False)

                    # In ra console một ít cho monitoring
                    print(f"\n=== {datetime.now().strftime('%H:%M:%S')} | {mode} signals ({len(alerts)}) ===")
                    for _, r in alerts.iterrows():
                        print(json.dumps({
                            "ticker": r["ticker"],
                            "ts": int(r["timestamp"]),
                            "close": float(r["close"]),
                            "ema20": float(r["ema20"]) if pd.notna(r["ema20"]) else None,
                            "rsi14": float(r["rsi14"]) if pd.notna(r["rsi14"]) else None,
                            "of_ratio": float(r["of_ratio"]) if pd.notna(r["of_ratio"]) else None,
                            "signal": r["signal"]
                        }, ensure_ascii=False))

                last_flush = now

        except Exception as e:
            print(f"[flush-loop] error: {e}")

    # 7) Stop everything
    print("[shutdown] Stopping workers...")
    for w in workers:
        w.stop()
    for w in workers:
        if w._thread:
            w._thread.join(timeout=2)
    print("[shutdown] Done.")

def main():
    parser = argparse.ArgumentParser(description="FiinQuantX realtime pipeline (intraday 15m / EOD)")
    parser.add_argument("--mode", choices=["intraday_15m","eod"], default="intraday_15m")
    parser.add_argument("--lasted", action="store_true")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--tickers-limit", type=int, default=92,
                        help="Số mã tối đa theo quota account (FiinQuantX).")
    parser.add_argument("--rotate-mins", type=int, default=0,
                        help="Số phút chạy mỗi batch trước khi chuyển batch kế (0 = chạy vô hạn).")

    args = parser.parse_args()

    universe_raw = load_universe(".")
    print(f"[init] Universe size (raw) = {len(universe_raw)}")

    rotate_sec = args.rotate_mins * 60 if args.rotate_mins > 0 else None

    for batch_id, batch in enumerate(split_batches(universe_raw, args.tickers_limit), start=1):
        print(f"[batch-{batch_id}] running {len(batch)} tickers")
        run_pipeline(
            mode=args.mode,
            lasted=args.lasted,
            max_workers=args.max_workers,
            tickers=batch,
            max_duration_sec=rotate_sec,
            tickers_limit=args.tickers_limit
        )

if __name__ == "__main__":
    main()
