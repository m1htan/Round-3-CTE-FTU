import time
from typing import Optional
from FiinQuantX import FiinSession
from utils.logger import get_logger

log = get_logger("fiin_client")

def build_client(username: str, password: str, max_retries: int = 3) -> Optional[object]:
    """
    Tạo client FiinSession có retry cơ bản.
    """
    for attempt in range(1, max_retries+1):
        try:
            log.info(f"Login to FiinQuantX attempt {attempt}/{max_retries}")
            client = FiinSession(username=username, password=password).login()
            log.info("Login successful.")
            return client
        except Exception as e:
            log.exception(f"Login failed (attempt {attempt}): {e}")
            time.sleep(2 * attempt)
    return None
