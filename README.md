# PP-OCRv5 OpenVINO Pipeline

A project to convert PP-OCRv5 models to run on OpenVINO Runtime, execute OCR pipelines, and measure accuracy and speed.

## System Requirements

- Python 3.12
- Windows 10/11 or Linux (Ubuntu)

**Note:** Exporting models to ONNX format only works on Linux (Ubuntu) due to DLL errors on Windows.

## Installation

### 1. Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download Models (if not available)

Download model files to the `archive/` directory:

- [PP-OCRv5_mobile_det_infer.tar](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar)
- [PP-OCRv5_mobile_rec_infer.tar](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv5_mobile_rec_infer.tar)

### 4. Convert Models to ONNX

**Note:** Only runs on Linux (Ubuntu)

```bash
chmod +x scripts/convert_models.sh
./scripts/convert_models.sh
```

## Usage

### Run OpenVINO Pipeline

```bash
python scripts/ocr_pipeline.py --input-dir samples/ --output-dir output/openvino/ --visualize
```

### Run PaddleOCR Native Pipeline

```bash
python scripts/paddle_pipeline.py --input-dir samples/ --output-dir output/paddle/ --visualize
```

### Parameters

#### ocr_pipeline.py (OpenVINO)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--image` | Path to a single image file | - |
| `--input-dir` | Directory containing input images | `samples` |
| `--output-dir` | Directory to save results | `output` |
| `--config` | Path to configuration file | `config/application.json` |
| `--visualize` | Save images with bounding boxes | `false` |
| `--verbose` | Display detailed logs | `false` |

#### paddle_pipeline.py (PaddleOCR Native)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--image` | Path to a single image file | - |
| `--input-dir` | Directory containing input images | `samples` |
| `--output-dir` | Directory to save results | `output/paddle` |
| `--config` | Path to configuration file | `config/paddle.json` |
| `--visualize` | Save images with bounding boxes | `false` |
| `--verbose` | Display detailed logs | `false` |

### Output Structure

```
output/
├── openvino/           # OpenVINO pipeline results
│   ├── details.csv     # Timing details (detection_ms, recognition_ms, total_ms)
│   ├── *_result.json   # OCR results per image
│   └── *_vis.jpg       # Visualization images
└── paddle/             # PaddleOCR native results
    ├── details.csv     # Timing details (total_ms)
    ├── *_result.json   # OCR results per image
    └── *_vis.jpg       # Visualization images
```

### Benchmark Comparison

Both pipelines include a warm-up phase (2 dummy images) before processing to ensure fair comparison. Results can be analyzed using the `details.csv` files in each output directory.
