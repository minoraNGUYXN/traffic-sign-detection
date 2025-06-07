# 🚦 Hệ thống Nhận diện Biển báo Giao thông Thời gian Thực

Dự án này triển khai một hệ thống nhận diện biển báo giao thông thời gian thực sử dụng camera, áp dụng các kỹ thuật học sâu để phát hiện và phân loại biển báo trong môi trường thực tế.

## Cấu trúc Dự án

- `src/`: Chứa mã nguồn chính cho việc xử lý hình ảnh, huấn luyện mô hình và nhận diện biển báo.
- `models/`: Lưu trữ các mô hình học sâu đã được huấn luyện.
- `dataset/`: Chứa dữ liệu huấn luyện, bao gồm hình ảnh và nhãn biển báo.
- `.idea/`: Tệp cấu hình của môi trường phát triển (IDE).

## Yêu cầu Hệ thống

- Python 3.8 trở lên
- OpenCV
- TensorFlow
- PyTorch

## Dataset
-Mapillary Traffic Sign Dataset
-German Traffic Sign Recognition Benchmark (GTSRB)

## Mô hình
-YOLOv8, YOLOv11 cho phát hiện biển báo giao thông
-VGG16 cho phân loại biển báo giao thông
