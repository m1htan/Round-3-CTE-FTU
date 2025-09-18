from typing import List, Dict
import os
import re
from pathlib import Path

# ====== RULES: giữ lại cổ phiếu thường (equities) ======
ETF_PREFIXES = ("FU", "FUE", "FUC", "E1", "E1V")   # ETF/quỹ
INDEX_KEYWORDS = ("INDEX", "VN30", "HNX30", "UPCOM")  # chỉ số
DERIV_KEYWORDS = ("VN30F", "F1M", "F2M", "F3M")    # phái sinh / futures
CW_REGEX = re.compile(r"[A-Z]{1,3}\d{2,}")         # pattern CW
MUTUAL_FUND_SUFFIXES = ("FUND", "ETF", "REIT")     # quỹ / reit
BLACKLIST = {"VNINDEX", "HNXINDEX", "UPCOMINDEX"}  # chỉ số tổng


PROJ_ROOT = Path(__file__).resolve().parents[2]
BLACKLIST_FILE = PROJ_ROOT / "data" / "universe" / "blacklist.txt"

def _load_blacklist() -> set:
    if BLACKLIST_FILE.exists():
        with open(BLACKLIST_FILE, "r", encoding="utf-8") as f:
            return {line.strip().upper() for line in f if line.strip()}
    return set()

def _is_equity_symbol(t: str) -> bool:
    t = t.strip().upper()
    if not t:
        return False
    if any(k in t for k in INDEX_KEYWORDS):
        return False
    if any(k in t for k in DERIV_KEYWORDS):
        return False
    if t.startswith(ETF_PREFIXES) or t.endswith(MUTUAL_FUND_SUFFIXES):
        return False
    if CW_REGEX.search(t):
        return False
    if t in BLACKLIST:
        return False
    return True

def _parse_single_line_symbols(text: str) -> List[str]:
    """
    Nhận chuỗi kiểu: `'AAV', 'ADC', 'ALT'` hoặc `[ 'AAA','BBB' ]` hoặc `AAA, BBB`
    Trả về list mã (uppercase), đã loại dấu nháy/ngoặc/khoảng trắng.
    """
    # bỏ ngoặc vuông/tròn
    text = re.sub(r"[\[\]\(\)]", " ", text)
    # thay ; bằng , nếu có
    text = text.replace(";", ",")
    # tách theo dấu phẩy
    raw_parts = [p.strip() for p in text.split(",") if p.strip()]
    tokens = []
    for p in raw_parts:
        # bỏ toàn bộ dấu nháy đơn/nháy kép
        p = p.strip().strip("'").strip('"')
        # fallback: nếu vẫn dính ký tự lạ, lấy chuỗi chữ-số liên tục dài nhất
        m = re.findall(r"[A-Za-z0-9]+", p)
        if not m:
            continue
        # thường phần tử hợp lệ là token đầu
        tok = m[0].upper()
        tokens.append(tok)
    # nếu file không có dấu phẩy mà là chuỗi liền: fallback bắt tất cả cụm chữ-số
    if not tokens:
        tokens = [m.upper() for m in re.findall(r"[A-Za-z0-9]+", text)]
    # unique, giữ thứ tự
    seen, uniq = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

def read_tickers_from_txt(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Universe file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return _parse_single_line_symbols(content)

def load_universe(universe_files: Dict[str, str], equities_only: bool = True) -> List[str]:
    tickers, bl = [], _load_blacklist()
    for _board, path in universe_files.items():
        ls_ = read_tickers_from_txt(path)
        tickers.extend(ls_)
    # unique
    seen, uniq = set(), []
    for t in tickers:
        t = t.upper()
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    if equities_only:
        filtered = [t for t in uniq if _is_equity_symbol(t)]
    else:
        filtered = uniq
    # áp blacklist động
    filtered = [t for t in filtered if t not in bl]
    return filtered

def chunk(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]
