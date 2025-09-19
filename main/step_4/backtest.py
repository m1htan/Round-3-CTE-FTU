import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common import setup_logger, PROJECT_ROOT, DATA_DIR, latest_file, normalize_signals_df, ensure_sorted, save_json, today_tag
from risk_guard import (build_trades_from_signals, apply_fixed_sl_tp,
                         attach_returns, equity_curve, max_drawdown)

LOGGER = setup_logger("step4.backtest")

def main():
    ap = argparse.ArgumentParser(description="Step 4 - Backtest EOD signals")
    ap.add_argument("--input",
                    help="CSV tín hiệu (vd: data/step2_signals/signals_from_merged_full_*.csv "
                         "hoặc data/round2/step_3/step_3_merged.csv). "
                         "Nếu bỏ trống: tự lấy mới nhất từ step2_signals/*signals*.csv.")
    ap.add_argument("--hold-days", type=int, default=10)
    ap.add_argument("--stoploss", type=float, default=0.10, help="10%% = 0.10 (None để tắt)", nargs="?")
    ap.add_argument("--takeprofit", type=float, default=0.15, help="15%% = 0.15 (None để tắt)", nargs="?")
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--cost", type=float, default=0.0005)
    args = ap.parse_args()

    # 1) đọc input
    if args.input:
        src = Path(args.input)
    else:
        # ưu tiên file step2_signals
        src = latest_file("signals_*full_*.csv", PROJECT_ROOT / "data" / "step2_signals")
        if src is None:
            # fallback: merged
            src = latest_file("step_3_merged*.csv", PROJECT_ROOT / "data" / "round2" / "step_3")
    if src is None:
        raise FileNotFoundError("Không tìm thấy file input tín hiệu.")
    LOGGER.info(f"Input: {src}")

    df0 = pd.read_csv(src)
    df = normalize_signals_df(df0)
    df = ensure_sorted(df)

    # 2) build trades đơn giản (BUY giữ K ngày)
    trades = build_trades_from_signals(df, hold_days=args.hold_days)

    # 3) áp SL/TP và tính PnL
    trades = apply_fixed_sl_tp(trades, px=df, stoploss=args.stoploss, takeprofit=args.takeprofit)
    trades = attach_returns(trades)

    # 4) portfolio equity & metrics
    curve = equity_curve(trades, max_positions=args.max_positions, capital=args.capital, cost_per_trade=args.cost)
    m = {
        "n_trades": int(len(trades)),
        "avg_ret": float(trades["ret"].mean()) if len(trades) else np.nan,
        "win_rate": float((trades["ret"] > 0).mean()) if len(trades) else np.nan,
        "max_drawdown": float(max_drawdown(curve["equity"])) if len(curve) else np.nan,
        "final_equity": float(curve["equity"].iloc[-1]) if len(curve) else float(args.capital),
    }

    # 5) lưu kết quả
    tag = today_tag()
    out_dir = DATA_DIR / f"run_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    trades_out = out_dir / "trades.csv"
    curve_out = out_dir / "equity_curve.csv"
    metrics_out = out_dir / "backtest_metrics.json"

    trades.to_csv(trades_out, index=False)
    curve.to_csv(curve_out, index=False)
    save_json(m, metrics_out)

    LOGGER.info(f"Saved trades : {trades_out}")
    LOGGER.info(f"Saved curve  : {curve_out}")
    LOGGER.info(f"Saved metrics: {metrics_out}")
    LOGGER.info(f"Done. Ntrades={m['n_trades']}, win={m['win_rate']:.2% if m['win_rate']==m['win_rate'] else 0}, "
                f"avg_ret={m['avg_ret']:.4f if m['avg_ret']==m['avg_ret'] else 0}, "
                f"MDD={m['max_drawdown']:.2% if m['max_drawdown']==m['max_drawdown'] else 0}")

if __name__ == "__main__":
    main()
