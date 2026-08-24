# Hệ Hỗ Trợ Ra Quyết Định (DSS) - Nhập hàng & Đẩy hàng trên Shopee

Đồ án môn **Hệ hỗ trợ quyết định**. Ứng dụng dùng **AHP** để xác định trọng số tiêu chí
và **TOPSIS** để xếp hạng sản phẩm, rồi chuyển điểm số thành khuyến nghị hành động
(**NÊN NHẬP / NHẬP VỪA / KHÔNG NÊN NHẬP**) cho chủ cửa hàng *cuahangchutchiu*.

## 🧭 Luồng hoạt động

```
File Excel gốc
   └─> load_data()            đọc sheet "Danh mục hàng hóa", làm sạch, dựng ma trận quyết định
        └─> AHP               so sánh cặp 5 tiêu chí -> trọng số -> kiểm tra CR < 0.1
             └─> TOPSIS       chuẩn hóa -> nhân trọng số -> A+/A- -> khoảng cách -> C*
                  └─> Ngưỡng khuyến nghị (phân vị / tuyệt đối) -> nhóm + hành động
                       └─> Giao diện: bảng xếp hạng, biểu đồ, phân tích độ nhạy
                            └─> Chủ shop ra quyết định cuối cùng
```

## 📁 Cấu trúc mã nguồn

| File | Vai trò |
|---|---|
| `topsis_app.py` | Giao diện Streamlit + tầng đọc/làm sạch dữ liệu (`load_data`) |
| `dss_model.py` | **Module mô hình (TV3)**: AHP, TOPSIS, ngưỡng khuyến nghị, phân tích độ nhạy, các mục tiêu kinh doanh đặt sẵn |
| `test_dss_model.py` | 25 ca kiểm thử tự động chứng minh công thức cài đặt đúng |
| `Tai_lieu_thuyet_trinh_TOPSIS.docx` | **Toàn bộ lý thuyết, công thức và bộ câu hỏi phản biện** - tách khỏi app để giao diện gọn như một sản phẩm thật |
| `tao_tai_lieu.py` | Sinh lại file Word ở trên từ chính mã nguồn và dữ liệu: `python tao_tai_lieu.py` |
| `requirements.txt` | Danh sách thư viện và phiên bản đã kiểm thử |
| `Quản lý sản phẩm shopee cuahangchutchiu.xlsx` | Dữ liệu gốc (**chỉ đọc, không ghi đè**) |

Toàn bộ phần toán học nằm trong `dss_model.py`, tách khỏi giao diện để kiểm thử độc lập.

## 🚀 Cài đặt & chạy

**Bước 1 - Cài thư viện**

```bash
pip install -r requirements.txt
```

**Bước 2 - Chạy bộ kiểm thử** (nên chạy trước để chắc chắn mô hình đúng)

```bash
python test_dss_model.py
```
Kết quả mong đợi: `TẤT CẢ 25/25 CA KIỂM THỬ ĐỀU ĐẠT.`

**Bước 3 - Khởi động ứng dụng**

```bash
python -m streamlit run topsis_app.py
```
Giao diện mở tại `http://localhost:8501`.

> File Excel phải nằm **cùng thư mục** với `topsis_app.py`.

## 📊 Hướng dẫn sử dụng

Giao diện thiết kế cho **chủ shop**, không phải dân kỹ thuật: mở lên là đã có sẵn
kết quả, chưa cần chỉnh gì.

**Thanh bên chỉ có 2 lựa chọn chính:**

1. **Kỳ này bạn ưu tiên điều gì?** - chọn một mục tiêu kinh doanh bằng tiếng Việt
   (*cân bằng*, *kiếm lời nhiều nhất*, *chạy theo hàng bán chạy*,
   *tránh ôm thêm hàng đang tồn*, *ít vốn xoay vòng nhanh*). Mỗi mục tiêu đã được
   quy đổi sẵn thành một ma trận so sánh cặp AHP hoàn chỉnh, **cả 5 đều đạt CR < 0.1**.
   Biểu đồ mức quan trọng hiện ngay bên dưới, cập nhật tức thì.
2. **Độ dài danh sách nên nhập** - bao nhiêu phần trăm danh mục vào nhóm nên nhập,
   tùy vốn hiện có. Số sản phẩm thật hiện ngay dưới thanh trượt.

> Lưu ý về mục tiêu *"Tránh ôm thêm hàng đang tồn"*: ứng dụng luôn trả lời câu hỏi
> **nên nhập thêm gì**, nên đầu bảng luôn là hàng đáng mua. Chọn mục tiêu này sẽ đẩy
> hàng đang tồn nhiều **xuống nhóm "Nên dừng nhập"** - đó mới là danh sách cần xả kho.

Mọi thứ phức tạp hơn nằm trong **Tùy chọn nâng cao**: tự cho điểm 10 cặp tiêu chí,
đổi cách tính doanh thu, bỏ qua sản phẩm chưa bán, dùng mốc điểm cố định.

**Ba tab kết quả:**

| Tab | Trả lời câu hỏi |
|---|---|
| **Gợi ý nhập hàng** | 5 sản phẩm đáng nhập nhất, bảng đầy đủ có tìm kiếm và lọc, biểu đồ, tải CSV |
| **Lựa chọn an toàn** | Sản phẩm nào vẫn trụ vững trong nhóm đầu dù đổi ưu tiên |
| **Dữ liệu** | Nguồn dữ liệu, tiêu chí và mức quan trọng đang áp dụng, các điểm cần lưu ý |

## ⚙️ Ghi chú về mô hình

- **Bộ tiêu chí (5):** Lợi nhuận/SP, Số lượng bán, Doanh thu (*lợi ích* - càng cao càng tốt);
  Tồn kho, Giá nhập (*chi phí* - càng thấp càng tốt).
- **CR của ma trận mặc định = 0.0074 < 0.1** → bộ đánh giá nhất quán.
- **Ngưỡng khuyến nghị mặc định theo phân vị** (20% / 50% / 30%). Điểm C* của TOPSIS phụ
  thuộc vào chính tập phương án đang xét; với 217 sản phẩm, điểm dồn về dải hẹp nên ngưỡng
  cứng 0.7/0.4 sẽ đẩy gần như toàn bộ danh mục vào một nhóm. Ứng dụng vẫn giữ chế độ ngưỡng
  tuyệt đối để đối chiếu và sẽ **cảnh báo** khi có nhóm bị rỗng.
- **Tồn kho âm** (20 sản phẩm, thấp nhất −35) là lỗi ghi sổ, được đưa về 0 trước khi tính.
  Nếu giữ số âm, TOPSIS sẽ coi đó là mức tồn kho *lý tưởng* và đẩy nhầm sản phẩm lên hạng 1.
- **Hệ thống chỉ đưa khuyến nghị**, quyết định cuối cùng vẫn thuộc về chủ cửa hàng.
