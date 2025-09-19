import argparse, hashlib
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

from notifier import send_email, send_telegram

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_DIR  = PROJECT_ROOT / "data" / "step2_signals"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "alerts_sent_log.csv"  # chống gửi trùng

def latest_alerts_csv() -> Path:
    cands = sorted(IN_DIR.glob("alerts_eod_*.csv"))
    if not cands:
        raise FileNotFoundError("Không tìm thấy file alerts_eod_*.csv (Step 2).")
    return cands[-1]

def load_log() -> pd.DataFrame:
    if LOG_FILE.exists():
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=["sent_ts","ticker","signal","hash"])

def save_log(df_log: pd.DataFrame):
    df_log.to_csv(LOG_FILE, index=False)

def row_hash(row: pd.Series) -> str:
    # Hash dựa trên (ticker, signal, timestamp) để chống gửi lại cùng tín hiệu
    base = f"{row.get('ticker','')}|{row.get('final_signal') or row.get('rule_signal') or row.get('signal')}|{row.get('timestamp')}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def format_alert_text(row: pd.Series) -> str:
    # Nội dung hiển thị gọn và có số chính
    sig = row.get("final_signal") or row.get("rule_signal") or row.get("signal")
    ts  = row.get("timestamp")
    dt  = ""
    try:
        ts_int = int(ts)
        dt = datetime.fromtimestamp(ts_int/1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        dt = str(ts)

    fields = []
    for k in ["close","rsi14","macd","macd_signal","sma20","ema20","ema50","of_ratio","pe","pb","roe","debt_to_equity"]:
        if k in row and pd.notna(row[k]):
            try:
                fields.append(f"{k}={float(row[k]):.2f}")
            except Exception:
                fields.append(f"{k}={row[k]}")
    fields_str = ", ".join(fields[:8])  # tránh quá dài

    return f"[{sig}] {row.get('ticker')} @ {dt} | {fields_str}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="+", choices=["email","telegram"], default=["telegram"],
                    help="Kênh gửi cảnh báo.")
    ap.add_argument("--subject-prefix", default="[EOD Alerts]",
                    help="Tiêu đề email prefix.")
    ap.add_argument("--dedupe", action="store_true", help="Bật chống gửi trùng theo log.")
    ap.add_argument("--input-csv", help="Tự chỉ định file alerts_eod_*.csv (mặc định: mới nhất).")
    args = ap.parse_args()

    alerts_csv = Path(args.input_csv) if args.input_csv else latest_alerts_csv()
    print(f"[step3] reading: {alerts_csv}")
    df = pd.read_csv(alerts_csv)

    # Xác định cột tín hiệu
    sig_col = "final_signal" if "final_signal" in df.columns else ("rule_signal" if "rule_signal" in df.columns else "signal")
    if sig_col not in df.columns:
        print("[step3] Không có cột tín hiệu (final_signal/rule_signal/signal).")
        return
    df = df[df[sig_col].notna()].copy()
    if df.empty:
        print("[step3] No alerts to send.")
        return

    log_df = load_log()
    sent_rows = []

    for _, row in df.iterrows():
        h = row_hash(row) if args.dedupe else None
        if args.dedupe and not log_df[log_df["hash"] == h].empty:
            continue  # đã gửi rồi

        text = format_alert_text(row)
        subject = f"{args.subject_prefix} {row.get('ticker')} {row.get(sig_col)}"

        ok = True
        if "email" in args.channels:
            ok = send_email(subject, text) and ok
        if "telegram" in args.channels:
            ok = send_telegram(text) and ok

        if ok:
            sent_rows.append({
                "sent_ts": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": row.get("ticker"),
                "signal": row.get(sig_col),
                "hash": h or ""
            })

    if sent_rows:
        log_df = pd.concat([log_df, pd.DataFrame(sent_rows)], ignore_index=True)
        save_log(log_df)
        print(f"[step3] logged {len(sent_rows)} alerts -> {LOG_FILE}")
    else:
        print("[step3] Nothing sent.")

if __name__ == "__main__":
    main()
