import pandas as pd
from typing import Optional
from utils.logger import get_logger
from utils.io_buffer import append_df_csv

log = get_logger("callbacks")

# === Hook tích hợp chiến lược từ Round 2 ===
# TODO: bạn gắn hàm đánh tín hiệu tại đây, ví dụ:
# from main.step_3.filtering import generate_signals   # (nếu có)
# hoặc từ ML.py / backtest.py nếu đã có interface chung.
def evaluate_signals(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Nhận DataFrame realtime (ticks hoặc bars).
    Trả về DataFrame các tín hiệu (mua/bán/cảnh báo) nếu có.
    Hiện để placeholder; sẽ kết nối code Round 2 khi bạn cung cấp interface.
    """
    # Ví dụ placeholder: không phát tín hiệu, chỉ log kích thước.
    if df is None or df.empty:
        return None
    log.info(f"Received batch df shape={df.shape}")
    return None

def on_ticks_callback(data) -> None:
    """
    Callback cho Trading_Data_Stream (ticks). 'data' là RealTimeData object.
    """
    try:
        df = data.to_dataFrame()
        # Chuẩn hóa cột thời gian nếu cần:
        # df['TradingDate'] = pd.to_datetime(df['TradingDate'])
        # Lưu cache (tùy chọn)
        append_df_csv(df, cache_dir="data/stream_cache/ticks", prefix="ticks")
        # Phát tín hiệu
        sig = evaluate_signals(df)
        if sig is not None and not sig.empty:
            append_df_csv(sig, cache_dir="data/stream_cache/signals", prefix="signals_ticks")
            log.info(f"Signals(TICKS): {len(sig)} rows")
    except Exception as e:
        log.exception(f"[on_ticks_callback] error: {e}")

def on_bars_callback(data) -> None:
    """
    Callback cho Fetch_Trading_Data(realtime=True). 'data' là BarDataUpdate object.
    """
    try:
        df = data.to_dataFrame()
        # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')  # nếu timestamp là epoch (tuỳ FQX)
        append_df_csv(df, cache_dir="data/stream_cache/bars", prefix="bars")
        sig = evaluate_signals(df)
        if sig is not None and not sig.empty:
            append_df_csv(sig, cache_dir="data/stream_cache/signals", prefix="signals_bars")
            log.info(f"Signals(BARS): {len(sig)} rows")
    except Exception as e:
        log.exception(f"[on_bars_callback] error: {e}")
