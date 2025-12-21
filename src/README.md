# 🔥 Hệ thống phân tích lửa chi tiết từng bước

## 📋 Mô tả

Hệ thống này cung cấp phân tích chi tiết từng bước để xác định chính xác một ảnh có chứa lửa hay không. Thay vì chỉ đưa ra kết quả cuối cùng, hệ thống sẽ hiển thị rõ ràng từng bước phân tích và lý do tại sao đưa ra quyết định đó.

## 🎯 Tại sao cần hệ thống này?

Hệ thống phân loại lửa trước đây có thể đưa ra kết quả sai (ví dụ: ảnh có lửa rõ ràng nhưng lại phân loại là "NO FIRE"). Hệ thống mới này sẽ:

1. **Phân tích từng bước rõ ràng** - Hiển thị chính xác những gì hệ thống "nhìn thấy"
2. **Giải thích lý do** - Tại sao ảnh được phân loại là FIRE hay NO FIRE
3. **Kiểm tra điều kiện** - Xem ảnh có đáp ứng các tiêu chí nào
4. **Visualization** - Hiển thị trực quan các vùng lửa được phát hiện

## 🏗️ Kiến trúc hệ thống

### 7 bước phân tích chi tiết:

1. **📸 Bước 1: Load và preprocess ảnh**
   - Load ảnh và resize về kích thước chuẩn (224x224)
   - Chuyển đổi màu: RGB → HSV → Grayscale
   - Tính toán thống kê cơ bản (độ sáng, độ tương phản)

2. **🎨 Bước 2: Phân tích màu sắc**
   - Tạo mask cho từng màu lửa: đỏ, cam, vàng
   - Tính tỷ lệ từng màu trong ảnh
   - Tạo mask tổng hợp cho tất cả màu lửa

3. **🔥 Bước 3: Phân tích vùng lửa**
   - Tìm contours của các vùng lửa
   - Tính diện tích và số lượng vùng lửa
   - Phân tích độ sáng và độ bão hòa của từng vùng

4. **🌀 Bước 4: Phân tích texture**
   - Tính gradient của ảnh (Sobel)
   - Tính entropy của gradient để đo độ phức tạp
   - Phân tích hướng gradient

5. **📊 Bước 5: Phân tích histogram**
   - Tính histogram cho từng channel HSV
   - Phân tích tỷ lệ màu lửa trong histogram
   - Tính tỷ lệ độ bão hòa và độ sáng cao

6. **🎯 Bước 6: Tổng hợp kết quả**
   - Kiểm tra 6 điều kiện quan trọng
   - Tính điểm tổng hợp và độ tin cậy
   - Đưa ra quyết định cuối cùng

7. **📋 Bước 7: Tạo báo cáo**
   - Lưu kết quả chi tiết dưới dạng JSON
   - Tạo visualization trực quan

## 📁 Cấu trúc thư mục

```
src/
├── detailed_fire_analyzer.py      # Core analyzer
├── detailed_web_app.py            # Web application
├── test_detailed_analyzer.py      # Test script
├── templates/
│   └── detailed_index.html        # Web interface
├── results/                       # Kết quả phân tích
├── uploads/                       # Ảnh upload
└── README.md                      # Tài liệu này
```

## 🚀 Cách sử dụng

### 1. Chạy web application

```bash
cd src
python detailed_web_app.py
```

Truy cập: http://localhost:8083

### 2. Test với script

```bash
# Test với danh sách ảnh mặc định
python test_detailed_analyzer.py

# Test với ảnh cụ thể
python test_detailed_analyzer.py ../dataset/train/images/train_1.jpg
```

### 3. Sử dụng trực tiếp trong code

```python
from detailed_fire_analyzer import DetailedFireAnalyzer

# Khởi tạo analyzer
analyzer = DetailedFireAnalyzer()

# Phân tích ảnh
report = analyzer.analyze_image_step_by_step("path/to/image.jpg")

# Xem kết quả
print(f"Kết quả: {report['final_classification']['classification']}")
print(f"Độ tin cậy: {report['final_classification']['confidence']*100:.1f}%")
```

## 📊 Các điều kiện kiểm tra

Hệ thống kiểm tra 6 điều kiện chính:

1. **has_fire_colors**: Có màu lửa (đỏ/cam/vàng) > 2%
2. **has_fire_area**: Có vùng lửa đủ lớn > 2%
3. **has_brightness**: Độ sáng trung bình > 150
4. **has_saturation**: Độ bão hòa trung bình > 100
5. **has_texture**: Texture phức tạp (entropy > 4.0)
6. **has_fire_histogram**: Histogram có tỷ lệ màu lửa > 10%

**Quyết định cuối cùng:**
- FIRE: Ít nhất 3/6 điều kiện được đáp ứng (50%)
- NO FIRE: Dưới 3/6 điều kiện

## 🎨 Visualization

Hệ thống tạo ra visualization với 6 panel:

1. **Ảnh gốc** - Ảnh đã được resize
2. **Ảnh HSV** - Ảnh trong không gian màu HSV
3. **Mask tổng hợp** - Vùng màu lửa được phát hiện
4. **Mask từng màu** - Phân biệt đỏ/cam/vàng
5. **Contours vùng lửa** - Các vùng lửa được vẽ viền
6. **Thống kê màu sắc** - Biểu đồ tỷ lệ màu

## 📄 Báo cáo chi tiết

Mỗi lần phân tích sẽ tạo ra:

1. **File JSON** - Chứa tất cả kết quả chi tiết
2. **File PNG** - Visualization trực quan
3. **Log console** - Thông tin từng bước

## 🔧 Tùy chỉnh

### Thay đổi ngưỡng

```python
analyzer = DetailedFireAnalyzer()
analyzer.thresholds['fire_ratio_min'] = 0.03  # Tăng ngưỡng lên 3%
```

### Thay đổi màu lửa

```python
# Điều chỉnh range màu HSV
analyzer.fire_color_ranges['red_lower'] = np.array([0, 120, 120])
analyzer.fire_color_ranges['red_upper'] = np.array([8, 255, 255])
```

## 🐛 Xử lý lỗi

### Lỗi thường gặp:

1. **Import error**: Cài đặt thư viện cần thiết
   ```bash
   pip install opencv-python numpy matplotlib seaborn
   ```

2. **Memory error**: Giảm kích thước ảnh hoặc số lượng ảnh test

3. **File not found**: Kiểm tra đường dẫn ảnh

## 📈 Hiệu suất

- **Thời gian phân tích**: ~2-5 giây/ảnh
- **Độ chính xác**: Cao hơn hệ thống cũ do kiểm tra nhiều điều kiện
- **Khả năng giải thích**: 100% - mọi quyết định đều có lý do rõ ràng

## 🔮 Phát triển tương lai

1. **Machine Learning**: Kết hợp với deep learning để cải thiện độ chính xác
2. **Real-time**: Xử lý video stream
3. **Multi-class**: Phân loại nhiều loại lửa khác nhau
4. **API**: Tạo REST API cho tích hợp

## 📞 Hỗ trợ

Nếu gặp vấn đề, hãy:
1. Kiểm tra log console để xem lỗi chi tiết
2. Xem file báo cáo JSON để hiểu kết quả
3. Kiểm tra visualization để xác nhận phát hiện 