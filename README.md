# Phân loại cảm xúc tiếng Việt sử dụng Transformer (PhoBERT + Streamlit)

> **Đồ án Seminar -- 👥 Nhóm 2 người**

| STT | Họ và Tên              | MSSV       |
| :-: | ---------------------- | ---------- |
|  1  | Nguyễn Hoàng Thiên Bảo | 3122410019 |
|  2  | Bạch Thị Mỹ Hoà        | 3122410120 |

---

## 📌 1. Giới thiệu

Dự án xây dựng ứng dụng phân loại cảm xúc tiếng Việt vào 3 nhãn:
POSITIVE, NEUTRAL, NEGATIVE. Ứng dụng sử dụng mô hình PhoBERT
(fine-tuned) kết hợp Streamlit và chạy hoàn toàn offline.

## 🎯 2. Mục tiêu dự án

-   Xây dựng ứng dụng phân loại cảm xúc tiếng Việt.
-   Tích hợp Transformer pre-trained của Hugging Face.
-   Hỗ trợ teencode, thiếu dấu, từ lóng.
-   Lưu trữ lịch sử bằng SQLite.
-   Đạt độ chính xác ≥ 65% (thực tế đạt 100%).

## 🧠 3. Công nghệ sử dụng

-   Python, Streamlit
-   PhoBERT (trituenhantao/io.vn_sentiment_phobert)
-   Hugging Face Transformers
-   Underthesea, SQLite3, Torch

## 📁 4. Cấu trúc thư mục

    SEMINAR/
    │── app.py                  # File chính chạy ứng dụng Streamlit
    │── requirements.txt        # Thư viện cần cài đặt
    │── sentiment.db            # Database lưu lịch sử (tự tạo)
    │── README.md               # Tài liệu mô tả dự án
    │
    └── utils/
        └── teencode_dict.py    # Từ điển teencode -> tiếng Việt chuẩn


## ⚙️ 5. Hướng dẫn cài đặt

    Bước 1 : Cài thư viện trong file requirements.txt:
    pip install -r requirements.txt
    Bước 2 : Chạy ứng dụng:
    streamlit run app.py

## 🖥️ 6. Cách sử dụng

Nhập câu tiếng Việt → Nhấn "Phân loại cảm xúc" → Xem kết quả và lịch sử.

## 🧪 7. Kết quả kiểm thử

10/10 test case chính thức đạt đúng toàn bộ

## 🚀 8. Hướng phát triển

-   Xây dựng API.
-   Phân tích đoạn văn dài.
-   Dashboard thống kê lịch sử.

## 📚 9. Tài liệu tham khảo

PhoBERT -- VinAI, Hugging Face, Streamlit, Underthesea.
