import argparse
from settings import (
    USERNAME, PASSWORD, STREAM_MODE, TIMEFRAME, BATCH_SIZE
)
from utils.universe import load_universe, chunk
from utils.logger import get_logger
from fiin_client import build_client
from fiin_stream import start_ticks_stream, start_bars_stream, validate_batch_by_history
from callbacks import on_ticks_callback, on_bars_callback
from settings import UNIVERSE_FILES

log = get_logger("entrypoint")

def main():
    parser = argparse.ArgumentParser(description="Step 1 - Realtime Data Streaming via FiinQuantX")
    parser.add_argument("--mode", choices=["ticks", "bars"], default=STREAM_MODE, help="ticks (Trade stream) or bars (15m/1d...)")
    parser.add_argument("--tf", "--timeframe", dest="timeframe", default=TIMEFRAME, help="Timeframe for bars mode (e.g., 15m,1d)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Chunk size for streaming")
    parser.add_argument("--hold-seconds", type=int, default=0, help="Stop after N seconds (0 = run until stopped)")
    args = parser.parse_args()

    log.info("========== STEP 1: REALTIME STREAM INIT ==========")
    log.info(f"Mode={args.mode} | Timeframe={args.timeframe} | BatchSize={args.batch_size} | Hold={args.hold_seconds}s")
    log.info(f"Universe files: {UNIVERSE_FILES}")

    # 1) Load tickers
    try:
        tickers = load_universe(UNIVERSE_FILES, equities_only=True)
        log.info(f"Loaded {len(tickers)} tickers from HOSE/HNX/UPCOM")
    except Exception as e:
        log.exception(f"Failed to load universe: {e}")
        raise

    log.info(f"Loaded {len(tickers)} tickers from HOSE/HNX/UPCOM")

    # 2) Build client
    client = build_client(USERNAME, PASSWORD)
    if client is None:
        log.error("Cannot login to FiinQuantX. Abort.")
        return

    for i, batch in enumerate(chunk(tickers, args.batch_size), start=1):
        # PROBE 1–2 nến để gạn mã không hợp lệ (ví dụ ATP)
        valid_batch = validate_batch_by_history(client, batch,
                                                timeframe=args.timeframe) if args.mode == "bars" else batch
        if not valid_batch:
            log.warning(f"Batch {i} has 0 valid symbols after probe; skip.")
            continue
        log.info(f"Start batch {i} with {len(valid_batch)} tickers")
        try:
            if args.mode == "ticks":
                start_ticks_stream(client, valid_batch, on_ticks_callback, hold_seconds=args.hold_seconds)
            else:
                start_bars_stream(client, valid_batch, on_bars_callback, timeframe=args.timeframe,
                                  hold_seconds=args.hold_seconds)
        except Exception as e:
            log.exception(f"Batch {i} crashed: {e}")

    log.info("========== STEP 1: REALTIME STREAM END ==========")

if __name__ == "__main__":
    main()
