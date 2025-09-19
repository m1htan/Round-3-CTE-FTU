import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common import setup_logger, PROJECT_ROOT, DATA_DIR, latest_file, normalize_signals_df, ensure_sorted, today_tag

LOGGER = setup_logger("step4.shock")

def apply_price_shock(df: pd.DataFrame, shock_pct: float, on_date: str | None) -> pd.DataFrame:
    """
    Giảm/tăng giá 'close' theo tỷ lệ shock_pct vào 1 ngày (YYYY-MM-DD) hoặc toàn bộ khung thời gian nếu None.
    Không tái tính indicator nâng cao; mục tiêu: đo NHẠY CẢM của danh sách tín hiệu đã sinh.
    """
    out = df.copy()
    if on_date:
        # so sánh theo ngày UTC
        day = pd.to_datetime(out["timestamp"], unit="ms", utc=True).dt.tz_localize(None).dt.date.astype("string")
        mask = (day == on_date)
    else:
        mask = np.full(len(out), True, dtype=bool)

    out.loc[mask, "close"] = out.loc[mask, "close"] * (1.0 + shock_pct)
    return out

def main():
    ap = argparse.ArgumentParser(description="Step 4 - Shock test (tin sốc / biến động mạnh)")
    ap.add_argument("--input",
                    help="CSV tín hiệu (signals_*full_*.csv hoặc step_3_merged.csv). "
                         "Nếu bỏ trống tự chọn mới nhất trong step2_signals.")
    ap.add_argument("--shock-pct", type=float, default=-0.1,
                    help="Tỷ lệ shock giá close. Ví dụ: -0.1 = -10%")
    ap.add_argument("--shock-date", type=str, default=None,
                    help="YYYY-MM-DD. Nếu bỏ trống: shock toàn bộ.")
    args = ap.parse_args()

    # 1) đọc
    if args.input:
        src = Path(args.input)
    else:
        src = latest_file("signals_*full_*.csv", PROJECT_ROOT / "data" / "step2_signals")
        if src is None:
            src = latest_file("step_3_merged*.csv", PROJECT_ROOT / "data" / "round2" / "step_3")
    if src is None:
        raise FileNotFoundError("Không tìm thấy file input tín hiệu.")
    LOGGER.info(f"Input: {src}")

    df0 = pd.read_csv(src)
    base = normalize_signals_df(df0)
    base = ensure_sorted(base)

    # 2) shock
    shocked = apply_price_shock(base, shock_pct=args.shock_pct, on_date=args.shock_date)

    # 3) so sánh ảnh hưởng trên danh sách "có tín hiệu mua" của ngày shock
    # (đơn giản: nếu shock âm -> bao nhiêu lệnh trở thành lỗ > X%)
    merge = base.merge(shocked, on=["ticker","timestamp"], suffixes=("_base","_shock"))
    merge["chg"] = (merge["close_shock"] / merge["close_base"]) - 1.0

    # Tổng hợp thống kê trong ngày shock (nếu có), hoặc toàn bộ
    if args.shock_date:
        dtag = args.shock_date
    else:
        dtag = "ALL"

    stat = {
        "shock_pct": args.shock_pct,
        "shock_scope": dtag,
        "n_rows": int(len(merge)),
        "pct_moves_over_-5%": float((merge["chg"] <= -0.05).mean()) if len(merge) else np.nan,
        "pct_moves_over_-8%": float((merge["chg"] <= -0.08).mean()) if len(merge) else np.nan,
        "pct_moves_over_-10%": float((merge["chg"] <= -0.10).mean()) if len(merge) else np.nan,
        "median_move": float(merge["chg"].median()) if len(merge) else np.nan,
    }

    # 4) lưu
    tag = today_tag()
    out_dir = DATA_DIR / f"shock_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "shock_diff.csv"
    out_json = out_dir / "shock_summary.json"
    merge.to_csv(out_csv, index=False)

    from .common import save_json
    save_json(stat, out_json)

    LOGGER.info(f"Saved: {out_csv}")
    LOGGER.info(f"Summary: {out_json}")
    LOGGER.info(f"Done.")

if __name__ == "__main__":
    main()
