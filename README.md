# Hệ Hỗ Trợ Ra Quyết Định (DSS) - Đánh giá Sản phẩm Shopee

Đây là đồ án ứng dụng mô hình **TOPSIS** để xây dựng Hệ Hỗ Trợ Ra Quyết Định (Decision Support System - DSS). Ứng dụng giúp chủ cửa hàng phân tích dữ liệu bán hàng thực tế trên Shopee và đưa ra các khuyến nghị chiến lược (Đẩy mạnh marketing, Duy trì, Xả kho) dựa trên đa tiêu chí.

## 🛠️ Công nghệ sử dụng
* **Ngôn ngữ:** Python
* **Giao diện Web:** Streamlit
* **Xử lý dữ liệu & Toán học:** Pandas, Numpy
* **Trực quan hóa biểu đồ:** Plotly Express

## 🚀 Hướng dẫn Cài đặt & Chạy Ứng dụng

**Bước 1: Tải Code về máy**
Bấm vào nút xanh lá `Code` -> `Download ZIP` và giải nén ra một thư mục.

**Bước 2: Cài đặt thư viện**
Mở Terminal (hoặc CMD/PowerShell) tại thư mục vừa giải nén, chạy câu lệnh sau để cài đặt các công cụ cần thiết:
```bash
pip install streamlit pandas numpy plotly openpyxl
```

**Bước 3: Khởi động Ứng dụng**
Sau khi cài đặt xong, gõ lệnh sau để mở giao diện Web:
```bash
streamlit run topsis_app.py
```
Giao diện sẽ tự động mở lên tại địa chỉ `http://localhost:8501` trên trình duyệt mặc định của bạn.

## 📊 Hướng dẫn sử dụng
1. Ứng dụng sẽ tự động nạp dữ liệu từ file Excel `Quản lý sản phẩm shopee cuahangchutchiu.xlsx`.
2. Ở menu bên trái, người dùng có thể điều chỉnh **Trọng số (Weights)** để thực hiện Phân tích độ nhạy (What-If Analysis). 
3. Bảng xếp hạng và Biểu đồ bên phải sẽ ngay lập tức được tính toán lại theo thuật toán TOPSIS để cập nhật khuyến nghị mới nhất.
