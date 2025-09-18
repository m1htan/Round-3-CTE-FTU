import os
from dotenv import load_dotenv

# Đọc .env từ config/.env
load_dotenv(dotenv_path='../../config/.env')

USERNAME = os.getenv("FIINQUANT_USERNAME")
PASSWORD = os.getenv("FIINQUANT_PASSWORD")

# Mặc định: intraday 15m. Có thể override bằng CLI (--mode bars --tf 15m) hoặc .env
STREAM_MODE = os.getenv("STREAM_MODE", "bars")     # "ticks" | "bars"
TIMEFRAME   = os.getenv("TIMEFRAME", "15m")        # 1m,5m,15m,30m,1h,2h,4h,1d
LASTED      = os.getenv("LASTED", "False").lower() == "true"  # với Fetch_Trading_Data realtime False (không dùng ở đây)

# Tùy chọn lưu cache (debug/kiểm tra): "none" | "csv"
CACHE_MODE  = os.getenv("CACHE_MODE", "csv")
CACHE_DIR   = os.getenv("CACHE_DIR", "data/stream_cache")

# Batch size phòng khi server hạn chế số ticker/connection
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", "150"))

# Chọn sàn: HOSE/HNX/UPCOM. Ta đọc cả 3 theo yêu cầu Step 1.
UNIVERSE_FILES = {
    "HOSE":  os.getenv("UNIVERSE_VNINDEX", "../../data/universe/VNINDEX.txt"),
    "HNX":   os.getenv("UNIVERSE_HNX", "../../data/universe/HNXIndex.txt"),
    "UPCOM": os.getenv("UNIVERSE_UPCOM", "../../data/universe/UpcomIndex.txt"),
}
