import pandas as pd
import numpy as np

def build_trades_from_signals(df: pd.DataFrame,
                              hold_days: int = 10,
                              entry_col: str = "entry_signal",
                              entry_value: str = "BUY") -> pd.DataFrame:
    """
    Tạo trades “đơn giản”: vào lệnh nếu entry_signal==BUY, giữ K ngày, thoát cuối kỳ
    (sẽ được thay/ghi đè bởi stoploss/takeprofit nếu cấu hình).
    Yêu cầu: cột ['ticker','timestamp','close', entry_col]
    """
    d = df[["ticker","timestamp","close", entry_col]].copy()
    d = d.sort_values(["ticker","timestamp"])
    entries = d[d[entry_col] == entry_value].copy()

    # index ngày (ms) theo từng ticker để nhảy K ngày
    next_ts = (
        d.groupby("ticker")["timestamp"]
          .shift(-hold_days)
          .rename("exit_ts_default")
    )
    d = pd.concat([d, next_ts], axis=1)

    entries = entries.merge(
        d[["ticker","timestamp","exit_ts_default"]],
        on=["ticker","timestamp"], how="left"
    )
    entries.rename(columns={"timestamp":"entry_ts", "close":"entry_price"}, inplace=True)

    # Lấy giá exit theo exit_ts_default
    exits = d[["ticker","timestamp","close"]].rename(columns={"timestamp":"exit_ts_default",
                                                              "close":"exit_price_default"})
    trades = entries.merge(exits, on=["ticker","exit_ts_default"], how="left")
    return trades

def apply_fixed_sl_tp(trades: pd.DataFrame,
                      px: pd.DataFrame,
                      stoploss: float | None = None,
                      takeprofit: float | None = None) -> pd.DataFrame:
    """
    Áp SL/TP đơn giản dựa trên dãy close theo từng ticker.
    - stoploss/takeprofit là tỷ lệ (%), ví dụ 0.1 = 10%.
    - Nếu không chạm SL/TP thì giữ nguyên exit mặc định.
    """
    if stoploss is None and takeprofit is None:
        trades["exit_ts"] = trades["exit_ts_default"]
        trades["exit_price"] = trades["exit_price_default"]
        trades["exit_reason"] = "time"
        return trades

    px = px[["ticker","timestamp","close"]].sort_values(["ticker","timestamp"]).copy()

    out = trades.copy()
    out["exit_ts"] = out["exit_ts_default"]
    out["exit_price"] = out["exit_price_default"]
    out["exit_reason"] = "time"

    # quét theo từng trade
    for i, r in out.iterrows():
        tkr = r["ticker"]
        t_entry = r["entry_ts"]
        t_exit_default = r["exit_ts_default"]
        p0 = r["entry_price"]

        path = px[(px["ticker"] == tkr) &
                  (px["timestamp"] >= t_entry) &
                  (px["timestamp"] <= t_exit_default)].copy()

        # bỏ ngày vào lệnh (ta muốn xét từ phiên sau)
        path = path[path["timestamp"] > t_entry]
        if path.empty or pd.isna(p0):
            continue

        hit_ts = None
        hit_px = None
        reason = None

        for _, rr in path.iterrows():
            p = rr["close"]
            if pd.isna(p):
                continue
            chg = (p / p0) - 1.0
            if (stoploss is not None) and (chg <= -abs(stoploss)):
                hit_ts = rr["timestamp"]; hit_px = p; reason = "SL"; break
            if (takeprofit is not None) and (chg >=  abs(takeprofit)):
                hit_ts = rr["timestamp"]; hit_px = p; reason = "TP"; break

        if hit_ts is not None:
            out.at[i, "exit_ts"] = hit_ts
            out.at[i, "exit_price"] = hit_px
            out.at[i, "exit_reason"] = reason
    return out

def attach_returns(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    t["ret"] = (t["exit_price"] / t["entry_price"]) - 1.0
    return t

def equity_curve(trades: pd.DataFrame,
                 max_positions: int = 5,
                 capital: float = 1_000_000.0,
                 cost_per_trade: float = 0.0005) -> pd.DataFrame:
    """
    Mô phỏng equity với phân bổ đều vốn cho tối đa N vị thế đồng thời.
    Đơn giản: khởi tạo theo ngày bằng cách phân bổ trọng số 1/n cho các lệnh mở trong ngày đó.
    """
    t = trades.dropna(subset=["entry_ts","exit_ts","entry_price","exit_price"]).copy()
    t = t.sort_values(["entry_ts","exit_ts"]).reset_index(drop=True)

    # hạn mức vị thế đồng thời
    # (đơn giản: nếu >max_positions trong cùng ngày, chỉ lấy top theo %ret kỳ vọng = (exit_default/entry-1))
    t["exp_ret"] = (t["exit_price_default"] / t["entry_price"]) - 1.0
    eq_trades = []
    by_day = t.groupby("entry_ts", as_index=True)
    for entry_ts, g in by_day:
        gg = g.sort_values("exp_ret", ascending=False).head(max_positions).copy()
        gg["weight"] = 1.0 / len(gg)
        eq_trades.append(gg)
    t2 = pd.concat(eq_trades, ignore_index=True) if eq_trades else t.assign(weight=0.0)

    # equity
    t2["pnl"] = capital * t2["weight"] * ((t2["exit_price"] / t2["entry_price"]) - 1.0) - (2 * cost_per_trade * capital * t2["weight"])
    daily = t2.groupby("exit_ts")["pnl"].sum().sort_index()
    curve = daily.cumsum().rename("equity_pnl").to_frame()
    curve["equity"] = capital + curve["equity_pnl"]
    return curve.reset_index(names="timestamp")

def max_drawdown(series: pd.Series) -> float:
    roll_max = series.cummax()
    dd = (series / roll_max) - 1.0
    return float(dd.min())
