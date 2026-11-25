# Memory-Masked-EVT-MIL

Nghiên cứu phát hiện bất thường trong video với giám sát yếu, ngân hàng bộ nhớ thưa hóa và EVT-MIL trên đặc trưng CLIP ViT-B/16.

## Giai đoạn 1
Chuẩn bị nguồn dữ liệu, môi trường GPU và Git. Chưa triển khai mô hình đề xuất.

## Dữ liệu
- [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/)
- [VadCLIP – nguồn đặc trưng tham khảo](https://github.com/nwpu-zxr/VadCLIP)

## Local
Cài Python 3.11, Git và NVIDIA driver. Tạo môi trường riêng. Cài PyTorch CUDA theo https://pytorch.org/get-started/locally/ trước khi chạy:

```bash
pip install -r requirements.txt
python scripts/check_gpu.py
```

## Colab
Mở notebooks/01_setup_environment.ipynb trong Google Colab, chọn GPU và chạy tất cả cell.

## Tiêu chí kiểm tra
CUDA khả dụng, CLIP tải thành công, đặc trưng có kích thước (1, 512) trên GPU. Lưu kết quả thực tế vào báo cáo. Các bài kiểm tra chưa được chạy trên GPU khi tạo repository mẫu.
