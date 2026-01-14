#!/bin/bash
#
# Model Conversion Script for PP-OCRv5
#
# This script extracts PP-OCRv5 models from tar archives and converts
# them to ONNX format for use with OpenVINO.
#
# Prerequisites:
#   - Python 3.11.9
#   - paddle2onnx 2.1.0
#   - PaddlePaddle 3.3.0
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
FONTS_DIR="$PROJECT_ROOT/fonts"
TEMP_DIR="$PROJECT_ROOT/temp_models"

# Model archive files
DET_ARCHIVE="PP-OCRv5_mobile_det_infer.tar"
REC_ARCHIVE="PP-OCRv5_mobile_rec_infer.tar"

# Output model names (chuẩn theo config)
DET_ONNX="PP-OCRv5_mobile_det_infer.onnx"
REC_ONNX="PP-OCRv5_mobile_rec_infer.onnx"

# ONNX opset version (11 is stable and recommended by official PaddleOCR docs)
OPSET_VERSION=11

echo "============================================"
echo "PP-OCRv5 Model Conversion Script"
echo "============================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Archive directory: $ARCHIVE_DIR"
echo "Output directory: $MODELS_DIR"
echo "ONNX Opset version: $OPSET_VERSION"
echo ""

# Function to detect model filename
get_model_filename() {
    local model_dir="$1"
    [ -f "$model_dir/inference.pdmodel" ] && echo "inference.pdmodel" && return 0
    [ -f "$model_dir/inference.json" ] && echo "inference.json" && return 0
    echo "ERROR: No model file found in $model_dir" >&2
    exit 1
}

# Function to convert Paddle model to ONNX
convert_paddle_to_onnx() {
    local model_dir="$1"
    local output_path="$2"
    local opset_version="$3"
    local model_filename=$(get_model_filename "$model_dir")
    
    echo "  Model: $model_filename -> $output_path"
    
    paddle2onnx \
        --model_dir "$model_dir" \
        --model_filename "$model_filename" \
        --params_filename "inference.pdiparams" \
        --save_file "$output_path" \
        --opset_version "$opset_version" \
        --enable_onnx_checker True
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Conversion failed!"
        return 1
    fi
    
    echo "  SUCCESS: $(du -h "$output_path" | cut -f1)"
    return 0
}

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

# Check if paddle2onnx is available
if ! command -v paddle2onnx &> /dev/null; then
    echo "ERROR: paddle2onnx not found. Install with: pip install paddle2onnx>=1.0.0"
    exit 1
fi
echo "paddle2onnx: OK"

# Create directories
echo "Creating directories..."
mkdir -p "$MODELS_DIR/det"
mkdir -p "$MODELS_DIR/rec"
mkdir -p "$FONTS_DIR"
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
echo "Detection model: $DET_DIR"

# Convert detection model to ONNX
echo "Converting detection model..."
if ! convert_paddle_to_onnx "$DET_DIR" "$MODELS_DIR/det/$DET_ONNX" "$OPSET_VERSION"; then
    echo "ERROR: Detection model conversion failed!"
    exit 1
fi

# Extract recognition model
echo ""
echo "Extracting recognition model..."
tar -xf "$ARCHIVE_DIR/$REC_ARCHIVE" -C "$TEMP_DIR"

# Find the extracted directory
REC_DIR=$(find "$TEMP_DIR" -type d -name "*rec*" | head -1)
if [ -z "$REC_DIR" ]; then
    REC_DIR="$TEMP_DIR/PP-OCRv5_mobile_rec_infer"
fi
echo "Recognition model: $REC_DIR"

# Convert recognition model to ONNX
echo "Converting recognition model..."
if ! convert_paddle_to_onnx "$REC_DIR" "$MODELS_DIR/rec/$REC_ONNX" "$OPSET_VERSION"; then
    echo "ERROR: Recognition model conversion failed!"
    exit 1
fi

# Clean up temporary directory
echo ""
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

# Download character dictionary if not exists
DICT_FILE="$FONTS_DIR/ppocr_keys_v1.txt"
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
echo "Detection model:  $MODELS_DIR/det/$DET_ONNX"
echo "Recognition model: $MODELS_DIR/rec/$REC_ONNX"
echo "Character dict:   $DICT_FILE"
echo ""
echo "You can now run the OCR pipeline:"
echo "  python scripts/ocr_pipeline.py --image samples/test.jpg"
