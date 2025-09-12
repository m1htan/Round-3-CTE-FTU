import os
from dotenv import load_dotenv

# Tải các biến từ config.env
load_dotenv(dotenv_path='../config/.env')

# Truy cập các biến môi trường
FIINQUANT_USERNAME = os.getenv("FIINQUANT_USERNAME")
FIINQUANT_PASSWORD = os.getenv("FIINQUANT_PASSWORD")

# Kiểm tra xem các biến đã được tải chưa
if FIINQUANT_USERNAME:
    print(f"USERNAME: {FIINQUANT_USERNAME}")
else:
    print("Không tìm thấy FIINQUANT_USERNAME. Hãy kiểm tra file của bạn.")

if FIINQUANT_PASSWORD:
    print(f"PASSWORD: {FIINQUANT_PASSWORD}")
else:
    print("Không tìm thấy FIINQUANT_PASSWORD. Hãy kiểm tra file của bạn.")