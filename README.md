# [CTE-FTU] ROUND 3: DATA SCIENCE TALENT COMPETITION 2025

## Tổng quan  
* Dự án này triển khai một hệ thống đầu tư định lượng **end-to-end**, từ thu thập dữ liệu thị trường (EOD/Intraday), xử lý tín hiệu kỹ thuật & cơ bản, cho đến hiển thị và kiểm thử chiến lược.  


* Hệ thống được phát triển phục vụ **Data Science Talent Competition 2025 (Round 3)**, đồng thời có thể mở rộng để áp dụng trong môi trường nghiên cứu học thuật và thực tiễn đầu tư.  

Pipeline bao gồm **4 bước chính**:  
1. **Step 1 - Data Fetching:** Thu thập dữ liệu EOD (OHLCV + giao dịch khối lượng, bu/sd/fn/fs/fb).  
2. **Step 2 – Signal Generation:** Tính toán Technical Indicators (TI) & Fundamental Indicators (FI), sinh tín hiệu giao dịch theo rule-based.  
3. **Step 3 – Alerting System:** Hiển thị tín hiệu BUY/SELL qua **dashboard (Streamlit)**, đồng thời hỗ trợ xuất cảnh báo qua email/Telegram.  
4. **Step 4 – Backtest & Stress Testing:** Phản biện và kiểm thử hệ thống trong nhiều kịch bản thị trường, đánh giá rủi ro và khả năng cải tiến.  

---

## Cấu trúc thư mục  
```bash
Round-3-CTE-FTU
    ├── .gitattributes
    ├── .gitignore
    ├── README.md
    ├── config
    │   └── .env
    ├── data
    │   ├── round2
    │   │   ├── step_1
    │   │   │   ├── cleaned_stocks.csv
    │   │   │   └── raw_stocks.csv
    │   │   ├── step_2
    │   │   │   ├── HNX_fundamental_ratios_quarterly.csv
    │   │   │   ├── HNX_ohlcv_with_fundamentals.csv
    │   │   │   ├── HNX_technical_indicators.csv
    │   │   │   ├── null_report_before_after.csv
    │   │   │   ├── part_1_cleaned_ohlcv_with_fundamentals_and_technical.csv
    │   │   │   └── part_2_cleaned_ohlcv_with_fundamentals_and_technical.csv
    │   │   ├── step_3
    │   │   │   ├── HNX_picks_daily.csv
    │   │   │   ├── HNX_signals_daily.csv
    │   │   │   └── step_3_merged.csv
    │   │   └── step_4
    │   │       └── HNX_ml_scores_daily.ckpt_H10.csv
    │   ├── step1_eod
    │   │   ├── HNX_ohlcv_with_fundamentals.csv
    │   │   └── bars_eod_20250919.csv
    │   ├── step2_signals
    │   │   ├── alerts_from_merged_20250919.csv
    │   │   └── signals_from_merged_full_20250919.csv
    │   └── universe
    │       ├── HNXIndex.txt
    │       ├── UpcomIndex.txt
    │       ├── VNINDEX.txt
    │       └── valid_1d.txt
    ├── docs
    │   ├── project_structure.txt
    │   ├── steps_by_steps.txt
    │   └── Đề bài Vòng 3 - Bảng ĐH _ DSTC 2025.pdf
    ├── logs
    ├── main
    │   ├── round2
    │   │   ├── filtering.py
    │   │   ├── fundamental_indicators.py
    │   │   └── technical_indicators.py
    │   ├── step_1
    │   │   ├── step_1.py
    │   │   └── utils_universe.py
    │   ├── step_2
    │   │   ├── signal_engine.py
    │   │   └── step_2.py
    │   ├── step_3
    │   │   ├── dispatch_alerts.py
    │   │   ├── notifier.py
    │   │   └── streamlit_app.py
    │   └── step_4
    │       ├── backtest.py
    │       ├── common.py
    │       ├── report.py
    │       ├── risk_guard.py
    │       └── shock_test.py
    ├── output
    ├── requirements.txt
    └── test
        ├── test_connections.py
        ├── test_crawl_realtime.py
        ├── test_data_validation.py
        ├── test_login.py
        └── tree.py
```
---

## Cách chạy  

### 1. Cài đặt môi trường  
```bash
pip install -r requirements.txt
```

### 2. Thu thập dữ liệu (Step 1)  
```bash
python main/step_1/fetch_eod.py
```
Dữ liệu được lưu tại `data/step1_eod/`  

### 3. Sinh tín hiệu (Step 2)  
```bash
python main/step_2/step_2.py --emit-latest-only
```
Kết quả:  
- `signals_eod_full_YYYYMMDD.csv`  
- `alerts_eod_YYYYMMDD.csv`  

### 4.Dashboard cảnh báo (Step 3)  
```bash
streamlit run main/step_3/streamlit_app.py
```
Hiển thị bảng tín hiệu BUY/SELL mới nhất.  

### 5.Backtest & Stress Testing (Step 4)  
```bash
python main/step_4/backtest.py
python main/step_4/shock_test.py
```
Báo cáo tổng hợp được xuất trong `data/reports/`  

---

## Các chỉ số đánh giá  

- **Tín hiệu:** BUY / SELL từ TI + FI  
- **Backtest:**  
  - Lợi nhuận trung bình  
  - Win-rate (%)  
  - Max Drawdown  
  - Equity Curve  
- **Stress Test:**  
  - Tác động của cú sốc giá (±10%, ±20%)  
  - Độ nhạy tín hiệu trước biến động tin tức  
- **Risk Guard:**  
  - Stop-loss / Take-profit động  
  - Giới hạn số lượng mã trong danh mục  

---

## Công nghệ sử dụng  

- **Ngôn ngữ:** Python 3.11  
- **Thư viện chính:**  
  - `pandas`, `numpy` – xử lý dữ liệu  
  - `ta`, `scikit-learn` – tính chỉ báo kỹ thuật, mô hình ML  
  - `streamlit` – giao diện dashboard  
  - `matplotlib` – trực quan hóa  
  - `logging` – quản trị log toàn hệ thống

---

**Tác giả:** Nhóm CTP  
