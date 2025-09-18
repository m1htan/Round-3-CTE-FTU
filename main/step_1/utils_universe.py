from pathlib import Path

def load_universe(root_dir: str = "."):
    """
    Đọc danh sách mã từ:
      data/universe/HNXIndex.txt
      data/universe/UpcomIndex.txt
      data/universe/VNINDEX.txt

    Mỗi file: một dòng duy nhất, nhiều mã phân tách bằng dấu phẩy,
    ví dụ: 'AAV', 'ADC', 'ALT', ...
    Trả về: sorted list (unique).
    """
    base = Path(root_dir).resolve().parents[1] / "data" / "universe"
    files = ["HNXIndex.txt", "UpcomIndex.txt", "VNINDEX.txt"]

    tickers = set()
    for fn in files:
        p = base / fn
        if not p.exists():
            raise FileNotFoundError(f"Universe file not found: {p}")
        content = p.read_text(encoding="utf-8").strip()
        # tách theo dấu phẩy
        for raw_code in content.split(","):
            code = raw_code.strip().strip("'\"").upper()
            if code:
                tickers.add(code)
    return sorted(tickers)
