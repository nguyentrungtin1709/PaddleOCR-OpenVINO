# Kế hoạch triển khai PP-OCRv5 với OpenVINO

## 📁 Cấu trúc dự án

```
PaddleOCR/
├── archive/                          # Model gốc (đã có)
│   ├── PP-OCRv5_mobile_det_infer.tar
│   └── PP-OCRv5_mobile_rec_infer.tar
├── config/
│   └── application.json              # Cấu hình tập trung
├── core/
│   ├── __init__.py
│   └── text_extract.py               # Logic OCR cốt lõi
├── models/                           # Model sau chuyển đổi
│   ├── det/                          # Detection model (ONNX)
│   └── rec/                          # Recognition model (ONNX)
├── samples/                          # Hình ảnh đầu vào
├── output/                           # Kết quả OCR (JSON)
├── scripts/
│   ├── convert_models.sh             # Script chuyển đổi model
│   └── ocr_pipeline.py               # Điều phối xử lý batch
├── utils/
│   ├── __init__.py
│   └── config_loader.py              # Đọc cấu hình
├── fonts/                            # Font cho visualization
├── .venv/                            # Virtual environment
├── requirements.txt
└── PLAN_V0.md
```

---

## 🔧 Phần 1: Thiết lập môi trường

### 1.1. requirements.txt

```txt
# OpenVINO Runtime
openvino>=2024.0.0

# PaddlePaddle và PaddleOCR (cho chuyển đổi model)
paddlepaddle>=2.5.0
paddleocr>=2.7.0
paddle2onnx>=1.0.0

# Xử lý ảnh và utility
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
pyclipper>=1.3.0
shapely>=2.0.0

# CLI và logging
tqdm>=4.66.0
```

### 1.2. Thiết lập Virtual Environment

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🔄 Phần 2: Chuyển đổi Model

### 2.1. Script chuyển đổi: `scripts/convert_models.sh`

**Quy trình:**
1. Giải nén model từ `archive/`
2. Chuyển đổi từ PaddlePaddle sang ONNX bằng `paddle2onnx`
3. Lưu vào thư mục `models/`

**Lệnh chuyển đổi:**
```bash
# Detection model
paddle2onnx \
    --model_dir ./archive/PP-OCRv5_mobile_det_infer \
    --model_filename inference.json \
    --params_filename inference.pdiparams \
    --save_file ./models/det/inference.onnx \
    --opset_version 12 \
    --enable_onnx_checker True

# Recognition model
paddle2onnx \
    --model_dir ./archive/PP-OCRv5_mobile_rec_infer \
    --model_filename inference.json \
    --params_filename inference.pdiparams \
    --save_file ./models/rec/inference.onnx \
    --opset_version 12 \
    --enable_onnx_checker True
```

---

## Phan 3: Cau hinh tap trung

### 3.1. Mapping cau hinh tu PaddleOCR sang OpenVINO

#### Bang anh xa tham so

| PaddleOCR (cu)           | OpenVINO (moi)              | Gia tri mac dinh | Mo ta                                      |
|--------------------------|-----------------------------|-----------------|--------------------------------------------|
| `textDetLimitType`       | `det_limit_type`            | "max"           | Kieu gioi han kich thuoc: "min" hoac "max" |
| `textDetLimitSideLen`    | `det_limit_side_len`        | 640             | Kich thuoc canh toi da/toi thieu           |
| `textDetThresh`          | `det_thresh`                | 0.15            | Nguong pixel cho text detection            |
| `textDetBoxThresh`       | `det_box_thresh`            | 0.15            | Nguong cho bounding box                    |
| `textDetUnclipRatio`     | `det_unclip_ratio`          | 2.0             | He so mo rong vung text                    |
| `textRecScoreThresh`     | `rec_score_thresh`          | 0.3             | Nguong diem nhan dang text                 |
| `cpuThreads`             | `num_threads`               | 2               | So luong CPU threads                       |
| `enableMkldnn`           | N/A (OpenVINO tu dong)      | -               | OpenVINO da toi uu san cho Intel CPU       |

### 3.2. config/application.json

```json
{
    "models": {
        "_description": "Duong dan den cac model ONNX da chuyen doi",
        "detectionModelPath": "models/det/inference.onnx",
        "recognitionModelPath": "models/rec/inference.onnx",
        "characterDictPath": "fonts/ppocr_keys_v1.txt"
    },
    "detection": {
        "_description": "Cau hinh cho text detection (DB algorithm)",
        "limitType": "max",
        "_comment_limitType": "Kieu gioi han: 'max' = canh dai nhat <= limitSideLen, 'min' = canh ngan nhat >= limitSideLen",
        "limitSideLen": 640,
        "_comment_limitSideLen": "Kich thuoc canh anh (pixels). Gia tri lon hon -> chi tiet hon nhung cham hon",
        "thresh": 0.15,
        "_comment_thresh": "Nguong pixel detection. Thap hon -> phat hien nhieu hon nhung co the nhieu noise",
        "boxThresh": 0.15,
        "_comment_boxThresh": "Nguong box. Chi giu cac box co diem trung binh > nguong nay",
        "unclipRatio": 2.0,
        "_comment_unclipRatio": "He so mo rong box. Lon hon -> box rong hon"
    },
    "recognition": {
        "_description": "Cau hinh cho text recognition",
        "imageHeight": 48,
        "_comment_imageHeight": "Chieu cao anh dau vao cho recognition model",
        "maxWidth": 320,
        "_comment_maxWidth": "Chieu rong toi da sau khi resize",
        "batchSize": 6,
        "_comment_batchSize": "So luong text crop xu ly moi batch",
        "scoreThresh": 0.3,
        "_comment_scoreThresh": "Nguong diem. Chi giu ket qua co score > nguong nay"
    },
    "openvino": {
        "_description": "Cau hinh OpenVINO Runtime",
        "device": "CPU",
        "_comment_device": "Thiet bi inference: 'CPU', 'GPU', 'AUTO'",
        "numThreads": 2,
        "_comment_numThreads": "So CPU threads. 0 = su dung tat ca cores",
        "numStreams": 1,
        "_comment_numStreams": "So luong inference streams song song. 1 = latency thap nhat",
        "performanceHint": "LATENCY",
        "_comment_performanceHint": "'LATENCY' = toi uu cho single request, 'THROUGHPUT' = toi uu cho nhieu requests",
        "enableHyperThreading": false,
        "_comment_enableHyperThreading": "Su dung hyper-threading. false = chi dung physical cores",
        "enableCpuPinning": true,
        "_comment_enableCpuPinning": "Ghim threads vao CPU cores. Tat neu chay nhieu workloads",
        "cacheDir": "",
        "_comment_cacheDir": "Thu muc cache model. De trong = khong cache"
    },
    "output": {
        "_description": "Cau hinh output",
        "saveVisualization": true,
        "_comment_saveVisualization": "Luu anh visualization voi bounding boxes",
        "dropScoreThreshold": 0.5,
        "_comment_dropScoreThreshold": "Nguong de loai bo ket qua khi visualization"
    }
}
```

### 3.3. utils/config_loader.py

**Trach nhiem:**
- Doc file `application.json`
- Cung cap cac dataclass/NamedTuple de truyen gia tri cau hinh
- **KHONG** duoc import trong `core/` (tuan thu DIP)

**Interface:**
```python
from dataclasses import dataclass

@dataclass
class OpenVINOConfig:
    device: str
    num_threads: int
    num_streams: int
    performance_hint: str
    enable_hyper_threading: bool
    enable_cpu_pinning: bool
    cache_dir: str

@dataclass
class DetectionConfig:
    model_path: str
    limit_type: str          # "max" hoac "min"
    limit_side_len: int      # 640
    thresh: float            # 0.15 - nguong pixel
    box_thresh: float        # 0.15 - nguong box
    unclip_ratio: float      # 2.0 - he so mo rong

@dataclass
class RecognitionConfig:
    model_path: str
    char_dict_path: str
    image_height: int        # 48
    max_width: int           # 320
    batch_size: int          # 6
    score_thresh: float      # 0.3 - nguong diem

@dataclass
class OutputConfig:
    save_visualization: bool
    drop_score_threshold: float

class ConfigLoader:
    def __init__(self, config_path: str):
        ...
    
    def get_openvino_config(self) -> OpenVINOConfig: ...
    def get_detection_config(self) -> DetectionConfig: ...
    def get_recognition_config(self) -> RecognitionConfig: ...
    def get_output_config(self) -> OutputConfig: ...
```

---

## Phan 4: Core OCR Engine

### 4.1. core/text_extract.py

**Trach nhiem:**
- Doc model ONNX bang OpenVINO Runtime
- Tien xu ly anh (resize, normalize)
- Thuc hien inference (detection + recognition)
- Hau xu ly (decode text, tinh toan bounding box)

**Class Design:**
```python
class TextExtractor:
    """
    Core OCR engine using OpenVINO Runtime.
    
    Nguyen tac thiet ke:
    - Nhan gia tri cau hinh qua constructor (DIP)
    - Khong phu thuoc vao config_loader
    - Co the test doc lap
    """
    
    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        char_dict_path: str,
        # Detection params (tuong ung PaddleOCR)
        det_limit_type: str = "max",       # textDetLimitType
        det_limit_side_len: int = 640,     # textDetLimitSideLen
        det_thresh: float = 0.15,          # textDetThresh
        det_box_thresh: float = 0.15,      # textDetBoxThresh
        det_unclip_ratio: float = 2.0,     # textDetUnclipRatio
        # Recognition params
        rec_image_height: int = 48,
        rec_max_width: int = 320,
        rec_batch_size: int = 6,
        rec_score_thresh: float = 0.3,     # textRecScoreThresh
        # OpenVINO params
        device: str = "CPU",
        num_threads: int = 2,              # cpuThreads
        num_streams: int = 1,
        performance_hint: str = "LATENCY",
        enable_hyper_threading: bool = False,
        enable_cpu_pinning: bool = True,
        cache_dir: str = ""
    ):
        """Khoi tao voi cac gia tri cau hinh duoc truyen vao."""
        ...
    
    def extract(self, image: np.ndarray) -> list[dict]:
        """
        Thuc hien OCR tren mot anh.
        
        Args:
            image: BGR image (numpy array)
            
        Returns:
            List of dict voi keys: 'bbox', 'text', 'score'
        """
        ...
    
    # Private methods
    def _init_openvino(self) -> None:
        """Khoi tao OpenVINO Core va compile models."""
        ...
    
    def _preprocess_detection(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Tien xu ly anh cho detection.
        
        Returns:
            - input_tensor: normalized image tensor
            - shape_info: thong tin de khoi phuc toa do goc
        """
        ...
    
    def _postprocess_detection(
        self, 
        output: np.ndarray, 
        shape_info: dict
    ) -> list[np.ndarray]:
        """
        Hau xu ly detection output.
        
        Su dung:
        - det_thresh: nguong pixel
        - det_box_thresh: nguong box
        - det_unclip_ratio: mo rong box
        
        Returns:
            List of bounding box polygons
        """
        ...
    
    def _crop_text_regions(
        self, 
        image: np.ndarray, 
        boxes: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Cat cac vung text tu anh goc."""
        ...
    
    def _preprocess_recognition(
        self, 
        crops: list[np.ndarray]
    ) -> np.ndarray:
        """Tien xu ly text crops cho recognition."""
        ...
    
    def _postprocess_recognition(
        self, 
        output: np.ndarray
    ) -> list[tuple[str, float]]:
        """
        Decode recognition output thanh text va score.
        
        Su dung:
        - rec_score_thresh: loc ket qua
        """
        ...
    
    def _decode_text(self, indices: np.ndarray) -> str:
        """Decode indices thanh text su dung character dictionary."""
        ...
```

**Output Format:**
```python
[
    {
        "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        "text": "Recognized text",
        "score": 0.95
    },
    ...
]
```

---

## 🔀 Phần 5: Pipeline Điều phối

### 5.1. scripts/ocr_pipeline.py

**Trách nhiệm:**
- Đọc cấu hình từ `config_loader`
- Khởi tạo `TextExtractor` với các giá trị cấu hình
- Quét thư mục `samples/`
- Gọi `TextExtractor.extract()` cho từng ảnh
- Ghi kết quả JSON vào `output/`

**Flow:**
```
1. Load config từ application.json
2. Khởi tạo TextExtractor với giá trị từ config
3. Lấy danh sách ảnh từ samples/
4. Với mỗi ảnh:
   a. Đọc ảnh bằng OpenCV
   b. Gọi extractor.extract(image)
   c. Ghi kết quả ra output/{image_name}.json
5. In summary
```

**CLI Interface:**
```bash
python scripts/ocr_pipeline.py --input samples/ --output output/ --config config/application.json
```

**Output JSON Format:**
```json
{
    "image_path": "samples/test_001.png",
    "image_size": {"width": 1920, "height": 1080},
    "processing_time_ms": 125.5,
    "results": [
        {
            "bbox": [[10, 20], [100, 20], [100, 50], [10, 50]],
            "text": "Hello World",
            "score": 0.98
        }
    ]
}
```

---

## 📋 Phần 6: Checklist triển khai

### Phase 1: Setup (Ưu tiên cao)
- [ ] Tạo `requirements.txt`
- [ ] Thiết lập `.venv`
- [ ] Cài đặt dependencies
- [ ] Tạo cấu trúc thư mục

### Phase 2: Model Conversion
- [ ] Tạo `scripts/convert_models.sh`
- [ ] Giải nén models từ archive
- [ ] Chạy chuyển đổi sang ONNX
- [ ] Verify models hoạt động

### Phase 3: Configuration
- [ ] Tạo `config/application.json`
- [ ] Tạo `utils/config_loader.py`
- [ ] Test đọc cấu hình

### Phase 4: Core Engine
- [ ] Tạo `core/__init__.py`
- [ ] Tạo `core/text_extract.py`
- [ ] Implement preprocessing
- [ ] Implement detection inference
- [ ] Implement recognition inference
- [ ] Implement postprocessing
- [ ] Unit test với sample image

### Phase 5: Pipeline
- [ ] Tạo `scripts/ocr_pipeline.py`
- [ ] Implement batch processing
- [ ] Implement JSON output
- [ ] End-to-end test

### Phase 6: Documentation & Polish
- [ ] Thêm docstrings
- [ ] Thêm logging
- [ ] Thêm error handling
- [ ] Cập nhật README

---

## 🚀 Quick Start (Sau khi hoàn thành)

```bash
# 1. Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Convert models
bash scripts/convert_models.sh

# 3. Run OCR
python scripts/ocr_pipeline.py --input samples/ --output output/
```

---

## Quy tac coding

### Luu y quan trong

- **KHONG su dung emoji** trong bat ky file ma nguon nao (.py, .sh, .json, etc.)
- Chi su dung ky tu ASCII chuan trong code
- Emoji chi duoc phep trong file documentation (.md) neu can thiet
- Su dung tieng Anh cho ten bien, ham, class
- Comment co the dung tieng Viet (khong dau) hoac tieng Anh

---

## Ghi chu ky thuat

### OpenVINO Performance Tuning

```python
import openvino as ov

# Khoi tao OpenVINO Core
core = ov.Core()

# Cau hinh CPU properties
core.set_property("CPU", {
    "NUM_STREAMS": num_streams,                    # So luong streams
    "INFERENCE_NUM_THREADS": num_threads,          # So luong threads
    "ENABLE_HYPER_THREADING": enable_hyper_threading,
    "AFFINITY": "CORE" if enable_cpu_pinning else "NONE"
})

# Performance hint khi compile model
config = {
    "PERFORMANCE_HINT": performance_hint,  # "LATENCY" hoac "THROUGHPUT"
    "CACHE_DIR": cache_dir                 # Thu muc cache (neu co)
}
compiled_model = core.compile_model(model, device, config)
```

### Pre-processing Detection (DB Algorithm)

```python
def preprocess_detection(
    image: np.ndarray, 
    limit_type: str,
    limit_side_len: int
) -> tuple[np.ndarray, dict]:
    """
    Tien xu ly anh cho DB text detection.
    
    Args:
        image: BGR image
        limit_type: "max" hoac "min"
        limit_side_len: kich thuoc gioi han
    
    Returns:
        - input_tensor: (1, 3, H, W) normalized
        - shape_info: {"src_h", "src_w", "ratio_h", "ratio_w"}
    """
    src_h, src_w = image.shape[:2]
    
    # Tinh toan kich thuoc moi
    if limit_type == "max":
        ratio = min(limit_side_len / max(src_h, src_w), 1.0)
    else:  # "min"
        ratio = max(limit_side_len / min(src_h, src_w), 1.0)
    
    new_h = int(src_h * ratio)
    new_w = int(src_w * ratio)
    
    # Dam bao chia het cho 32 (yeu cau cua DB model)
    new_h = max(32, (new_h // 32) * 32)
    new_w = max(32, (new_w // 32) * 32)
    
    # Resize
    img = cv2.resize(image, (new_w, new_h))
    
    # Transpose HWC -> CHW
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    
    # Normalize (ImageNet mean/std)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = (img - mean) / std
    
    # Add batch dimension
    input_tensor = np.expand_dims(img, 0)
    
    shape_info = {
        "src_h": src_h,
        "src_w": src_w,
        "ratio_h": new_h / src_h,
        "ratio_w": new_w / src_w
    }
    
    return input_tensor, shape_info
```

### Post-processing Detection (DB Algorithm)

```python
def postprocess_detection(
    output: np.ndarray,
    shape_info: dict,
    thresh: float = 0.15,
    box_thresh: float = 0.15,
    unclip_ratio: float = 2.0
) -> list[np.ndarray]:
    """
    Hau xu ly DB detection output.
    
    Args:
        output: model output (1, 1, H, W)
        shape_info: thong tin resize
        thresh: nguong pixel
        box_thresh: nguong box
        unclip_ratio: he so mo rong
    
    Returns:
        List of bounding box polygons, moi polygon la (4, 2) array
    """
    pred = output[0, 0]  # (H, W)
    
    # Binary threshold
    segmentation = pred > thresh
    
    # Tim contours
    contours, _ = cv2.findContours(
        (segmentation * 255).astype(np.uint8),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    boxes = []
    for contour in contours:
        # Tinh min area rect
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        
        # Tinh score trung binh trong box
        # ... (logic tinh score)
        
        # Neu score > box_thresh, mo rong box va them vao ket qua
        # Su dung pyclipper de unclip
        # ...
        
    # Scale ve toa do goc
    # ...
    
    return boxes
```

### Recognition Batch Processing

```python
def process_recognition_batch(
    crops: list[np.ndarray],
    batch_size: int,
    image_height: int,
    max_width: int
) -> list[np.ndarray]:
    """
    Xu ly batch cho recognition.
    
    Cac crop co kich thuoc khac nhau can duoc:
    1. Sap xep theo width ratio
    2. Resize ve cung height
    3. Pad ve cung width trong moi batch
    """
    results = []
    
    # Tinh width ratio va sap xep
    width_ratios = [c.shape[1] / c.shape[0] for c in crops]
    indices = np.argsort(width_ratios)
    
    for i in range(0, len(crops), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_crops = [crops[idx] for idx in batch_indices]
        
        # Tinh max width trong batch
        max_ratio = max(c.shape[1] / c.shape[0] for c in batch_crops)
        target_width = min(int(image_height * max_ratio), max_width)
        
        # Resize va pad
        batch_tensors = []
        for crop in batch_crops:
            # Resize ve image_height
            ratio = image_height / crop.shape[0]
            new_w = int(crop.shape[1] * ratio)
            resized = cv2.resize(crop, (new_w, image_height))
            
            # Normalize
            tensor = resized.astype(np.float32) / 255.0
            tensor = (tensor - 0.5) / 0.5
            tensor = tensor.transpose(2, 0, 1)  # HWC -> CHW
            
            # Pad ve target_width
            padded = np.zeros((3, image_height, target_width), dtype=np.float32)
            padded[:, :, :new_w] = tensor
            
            batch_tensors.append(padded)
        
        results.append(np.stack(batch_tensors))
    
    return results
```
