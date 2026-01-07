#!/bin/bash
#
# Model Conversion Script for PP-OCRv5
#
# This script extracts PP-OCRv5 models from tar archives and converts
# them to ONNX format for use with OpenVINO.
#
# Prerequisites:
#   - Python 3.8+
#   - paddle2onnx >= 1.0.0
#   - PaddlePaddle >= 2.5.0
#
# Usage:
#   ./scripts/convert_models.sh
#

set -e

# Project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Directories
ARCHIVE_DIR="$PROJECT_ROOT/archive"
MODELS_DIR="$PROJECT_ROOT/models"
TEMP_DIR="$PROJECT_ROOT/temp_models"

# Model archive files
DET_ARCHIVE="PP-OCRv5_mobile_det_infer.tar"
REC_ARCHIVE="PP-OCRv5_mobile_rec_infer.tar"

# Output model names
DET_ONNX="ch_PP-OCRv5_det_infer.onnx"
REC_ONNX="ch_PP-OCRv5_rec_infer.onnx"

# ONNX opset version (12 is recommended for OpenVINO 2024.0.0+)
OPSET_VERSION=12

echo "============================================"
echo "PP-OCRv5 Model Conversion Script"
echo "============================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Archive directory: $ARCHIVE_DIR"
echo "Output directory: $MODELS_DIR"
echo ""

# Check if archive directory exists
if [ ! -d "$ARCHIVE_DIR" ]; then
    echo "ERROR: Archive directory not found: $ARCHIVE_DIR"
    echo "Please create the 'archive' directory and place model tar files in it."
    exit 1
fi

# Check if detection model archive exists
if [ ! -f "$ARCHIVE_DIR/$DET_ARCHIVE" ]; then
    echo "ERROR: Detection model archive not found: $ARCHIVE_DIR/$DET_ARCHIVE"
    echo ""
    echo "Download from:"
    echo "  https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/PP-OCRv5_mobile_det_infer.tar"
    exit 1
fi

# Check if recognition model archive exists
if [ ! -f "$ARCHIVE_DIR/$REC_ARCHIVE" ]; then
    echo "ERROR: Recognition model archive not found: $ARCHIVE_DIR/$REC_ARCHIVE"
    echo ""
    echo "Download from:"
    echo "  https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/PP-OCRv5_mobile_rec_infer.tar"
    exit 1
fi

# Check if paddle2onnx is installed
if ! python -c "import paddle2onnx" 2>/dev/null; then
    echo "ERROR: paddle2onnx is not installed"
    echo "Install with: pip install paddle2onnx>=1.0.0"
    exit 1
fi

# Create directories
echo "Creating directories..."
mkdir -p "$MODELS_DIR"
mkdir -p "$TEMP_DIR"

# Extract detection model
echo ""
echo "Extracting detection model..."
tar -xf "$ARCHIVE_DIR/$DET_ARCHIVE" -C "$TEMP_DIR"

# Find the extracted directory
DET_DIR=$(find "$TEMP_DIR" -type d -name "*det*" | head -1)
if [ -z "$DET_DIR" ]; then
    DET_DIR="$TEMP_DIR/PP-OCRv5_mobile_det_infer"
fi
echo "Detection model directory: $DET_DIR"

# Convert detection model to ONNX
echo ""
echo "Converting detection model to ONNX..."
python -m paddle2onnx.convert \
    --model_dir "$DET_DIR" \
    --model_filename "inference.json" \
    --params_filename "inference.pdiparams" \
    --save_file "$MODELS_DIR/$DET_ONNX" \
    --opset_version $OPSET_VERSION \
    --enable_onnx_checker True

echo "Detection model saved to: $MODELS_DIR/$DET_ONNX"

# Extract recognition model
echo ""
echo "Extracting recognition model..."
tar -xf "$ARCHIVE_DIR/$REC_ARCHIVE" -C "$TEMP_DIR"

# Find the extracted directory
REC_DIR=$(find "$TEMP_DIR" -type d -name "*rec*" | head -1)
if [ -z "$REC_DIR" ]; then
    REC_DIR="$TEMP_DIR/PP-OCRv5_mobile_rec_infer"
fi
echo "Recognition model directory: $REC_DIR"

# Convert recognition model to ONNX
echo ""
echo "Converting recognition model to ONNX..."
python -m paddle2onnx.convert \
    --model_dir "$REC_DIR" \
    --model_filename "inference.json" \
    --params_filename "inference.pdiparams" \
    --save_file "$MODELS_DIR/$REC_ONNX" \
    --opset_version $OPSET_VERSION \
    --enable_onnx_checker True

echo "Recognition model saved to: $MODELS_DIR/$REC_ONNX"

# Clean up temporary directory
echo ""
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

# Download character dictionary if not exists
DICT_FILE="$MODELS_DIR/ppocr_keys_v1.txt"
if [ ! -f "$DICT_FILE" ]; then
    echo ""
    echo "Downloading character dictionary..."
    curl -L -o "$DICT_FILE" \
        "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt"
    echo "Character dictionary saved to: $DICT_FILE"
fi

echo ""
echo "============================================"
echo "Conversion completed successfully!"
echo "============================================"
echo ""
echo "Detection model:  $MODELS_DIR/$DET_ONNX"
echo "Recognition model: $MODELS_DIR/$REC_ONNX"
echo "Character dict:   $DICT_FILE"
echo ""
echo "You can now run the OCR pipeline:"
echo "  python scripts/ocr_pipeline.py --image samples/test.jpg"
