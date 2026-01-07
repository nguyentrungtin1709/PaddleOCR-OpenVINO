# PP-OCRv5 OpenVINO Pipeline

Pipeline OCR sử dụng PP-OCRv5 models chạy trên OpenVINO Runtime.

## Yêu cầu hệ thống

- Python 3.12
- Windows 10/11 hoặc Linux
- RAM >= 4GB

## Cài đặt

### 1. Tạo Virtual Environment

```powershell
# Di chuyển đến thư mục project
cd "c:\Users\POD SOFTWARE 03\Desktop\PaddleOCR"

# Tạo virtual environment
python -m venv .venv

# Kích hoạt virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Hoặc trên Command Prompt
# .\.venv\Scripts\activate.bat

# Hoặc trên Linux/Mac
# source .venv/bin/activate
```

### 2. Cài đặt Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Tải Models (nếu chưa có)

Tải các file model vào thư mục `archive/`:

- [PP-OCRv5_mobile_det_infer.tar](https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/PP-OCRv5_mobile_det_infer.tar)
- [PP-OCRv5_mobile_rec_infer.tar](https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/PP-OCRv5_mobile_rec_infer.tar)

### 4. Chuyển đổi Models sang ONNX

```powershell
# Windows PowerShell
.\scripts\convert_models.ps1

# Hoặc Linux/Mac
# chmod +x scripts/convert_models.sh
# ./scripts/convert_models.sh
```

## Sử dụng

### Chạy OCR trên một ảnh

```powershell
python scripts/ocr_pipeline.py --image samples/001.png --visualize
```

### Chạy OCR trên thư mục ảnh

```powershell
python scripts/ocr_pipeline.py --input-dir samples --output-dir output --visualize
```

### Các tham số

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--image` | Đường dẫn đến file ảnh | - |
| `--input-dir` | Thư mục chứa ảnh đầu vào | `samples` |
| `--output-dir` | Thư mục lưu kết quả | `output` |
| `--config` | Đường dẫn file cấu hình | `config/application.json` |
| `--visualize` | Lưu ảnh với bounding boxes | `false` |
| `--verbose` | Hiển thị log chi tiết | `false` |

## Cấu trúc thư mục

```
PaddleOCR/
├── archive/                    # Model archives (.tar)
├── config/
│   └── application.json        # Cấu hình chính
├── core/
│   └── text_extract.py         # OCR engine
├── models/                     # ONNX models (sau khi convert)
├── output/                     # Kết quả OCR
├── samples/                    # Ảnh mẫu
├── scripts/
│   ├── convert_models.ps1      # Script chuyển đổi (Windows)
│   ├── convert_models.sh       # Script chuyển đổi (Linux)
│   └── ocr_pipeline.py         # Pipeline chính
├── utils/
│   └── config_loader.py        # Config loader
├── .venv/                      # Virtual environment
├── requirements.txt            # Dependencies
└── README.md                   # File này
```

## Cấu hình

Chỉnh sửa file `config/application.json` để thay đổi các tham số:

- **detection**: Tham số phát hiện văn bản (thresh, box_thresh, unclip_ratio)
- **recognition**: Tham số nhận dạng (batch_size, score_thresh)
- **openvino**: Cấu hình runtime (device, num_threads, performance_hint)

## Khắc phục sự cố

### Lỗi "Import could not be resolved"
```powershell
# Đảm bảo đã kích hoạt virtual environment
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Lỗi "Model not found"
```powershell
# Chạy script chuyển đổi model
.\scripts\convert_models.ps1
```

### Lỗi "Character dictionary not found"
Script `convert_models.ps1` sẽ tự động tải file dictionary. Nếu không, tải thủ công:
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt" -OutFile "models/ppocr_keys_v1.txt"
```

## License

MIT License
