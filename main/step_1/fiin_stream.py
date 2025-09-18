import time
from typing import List, Optional, Callable
from utils.logger import get_logger

log = get_logger("fiin_stream")

def start_ticks_stream(client, tickers: List[str], callback: Callable, hold_seconds: int = 3600):
    """
    Trading_Data_Stream cho tick-level realtime.
    """
    Events = client.Trading_Data_Stream(tickers=tickers, callback=callback)
    Events.start()
    log.info(f"[TICKS] Started stream for {len(tickers)} tickers")
    try:
        t0 = time.time()
        while not getattr(Events, "_stop", False):
            time.sleep(1)
            if hold_seconds and time.time() - t0 > hold_seconds:
                log.info("[TICKS] Hold time reached, stopping.")
                break
    except KeyboardInterrupt:
        log.info("[TICKS] KeyboardInterrupt received, stopping...")
    finally:
        Events.stop()
        log.info("[TICKS] Stopped stream.")

def start_bars_stream(client, tickers: List[str], callback: Callable, timeframe: str = "15m",
                      period: Optional[int] = 200, wait_for_full: bool = False, hold_seconds: int = 3600*6):
    """
    Fetch_Trading_Data(realtime=True) để tổng hợp nến theo timeframe (vd 15m).
    """
    event = client.Fetch_Trading_Data(
        realtime=True,
        tickers=tickers,
        fields=['open', 'high', 'low', 'close', 'volume', 'bu', 'sd', 'fb', 'fs', 'fn'],
        adjusted=True,
        by=timeframe,
        callback=callback,
        period=period,
        wait_for_full_timeFrame=wait_for_full
    )
    event.get_data()
    log.info(f"[BARS] Started stream timeframe={timeframe} for {len(tickers)} tickers (period={period})")
    try:
        t0 = time.time()
        while not getattr(event, "_stop", False):
            time.sleep(1)
            if hold_seconds and time.time() - t0 > hold_seconds:
                log.info("[BARS] Hold time reached, stopping.")
                break
    except KeyboardInterrupt:
        log.info("[BARS] KeyboardInterrupt received, stopping...")
    finally:
        event.stop()
        log.info("[BARS] Stopped stream.")

def validate_batch_by_history(client, tickers: List[str], timeframe: str = "15m") -> List[str]:
    """
    Gọi nhẹ lịch sử 1-2 nến để loại mã không hợp lệ (ETF/index/delisted...),
    tránh làm fail cả JoinGroup.
    """
    ok = []
    try:
        event = client.Fetch_Trading_Data(
            realtime=False,
            tickers=tickers,
            fields=['close'],
            adjusted=True,
            by=timeframe,
            period=2,
            lasted=True
        )
        df = event.get_data()
        if df is None or len(df) == 0:
            return ok
        # Cột "ticker" có thể khác biệt tùy lib (ticker / Ticker)
        col = 'ticker' if 'ticker' in df.columns else ('Ticker' if 'Ticker' in df.columns else None)
        if col:
            ok = sorted(set(df[col].astype(str).str.upper().tolist()))
        else:
            ok = []
    except Exception as e:
        log.warning(f"[VALIDATE] history probe failed for batch of size {len(tickers)}: {e}")
        # fallback: để nguyên nhưng không loại
        ok = tickers
    if len(ok) < len(tickers):
        bad = sorted(set(tickers) - set(ok))
        log.info(f"[VALIDATE] filtered out {len(bad)} invalid symbols: {bad[:10]}{'...' if len(bad)>10 else ''}")
    return ok
