import argparse
from pathlib import Path
import json
import pandas as pd

from common import setup_logger, DATA_DIR, latest_file, today_tag

LOGGER = setup_logger("step4.report")

def md_kv(title: str, d: dict) -> str:
    lines = [f"## {title}"]
    for k, v in d.items():
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines) + "\n\n"

def main():
    ap = argparse.ArgumentParser(description="Step 4 - Tổng hợp báo cáo")
    ap.add_argument("--run-dir",
                    help="Thư mục run (ví dụ: data/step4/run_YYYYMMDD). Nếu bỏ trống: tự pick mới nhất.")
    ap.add_argument("--shock-dir",
                    help="Thư mục shock (ví dụ: data/step4/shock_YYYYMMDD). Tuỳ chọn.")
    args = ap.parse_args()

    # pick run dir
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        # tìm thư mục con mới nhất bắt đầu bằng 'run_'
        runs = sorted([p for p in DATA_DIR.glob("run_*") if p.is_dir()])
        if not runs:
            raise FileNotFoundError("Không tìm thấy run_* trong data/step4/")
        run_dir = runs[-1]
    LOGGER.info(f"Using run dir: {run_dir}")

    trades_csv = run_dir / "trades.csv"
    curve_csv  = run_dir / "equity_curve.csv"
    metrics_js = run_dir / "backtest_metrics.json"

    trades = pd.read_csv(trades_csv) if trades_csv.exists() else pd.DataFrame()
    curve  = pd.read_csv(curve_csv) if curve_csv.exists() else pd.DataFrame()
    metrics = json.loads(metrics_js.read_text()) if metrics_js.exists() else {}

    # shock (optional)
    shock_summary = {}
    if args.shock_dir:
        sdir = Path(args.shock_dir)
        sjson = sdir / "shock_summary.json"
        if sjson.exists():
            shock_summary = json.loads(sjson.read_text())

    # tạo report.md
    out_md = run_dir / f"report_{today_tag()}.md"
    md = ["# Step 4 Report\n"]
    md.append(md_kv("Backtest metrics", metrics))
    if not trades.empty:
        md.append(f"### Trades (head)\n\n```\n{trades.head(10).to_string(index=False)}\n```\n")
    if not curve.empty:
        md.append(f"### Equity (tail)\n\n```\n{curve.tail(10).to_string(index=False)}\n```\n")
    if shock_summary:
        md.append(md_kv("Shock test", shock_summary))

    out_md.write_text("\n".join(md), encoding="utf-8")
    LOGGER.info(f"Saved report: {out_md}")

if __name__ == "__main__":
    main()
