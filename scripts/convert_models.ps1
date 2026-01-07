<#
.SYNOPSIS
    Model Conversion Script for PP-OCRv5 (PowerShell version)

.DESCRIPTION
    This script extracts PP-OCRv5 models from tar archives and converts
    them to ONNX format for use with OpenVINO.

.PREREQUISITES
    - Python 3.8+
    - paddle2onnx >= 1.0.0
    - PaddlePaddle >= 2.5.0

.EXAMPLE
    .\scripts\convert_models.ps1
#>

$ErrorActionPreference = "Continue"

# Project root directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Directories
$ArchiveDir = Join-Path $ProjectRoot "archive"
$ModelsDir = Join-Path $ProjectRoot "models"
$TempDir = Join-Path $ProjectRoot "temp_models"

# Model archive files
$DetArchive = "PP-OCRv5_mobile_det_infer.tar"
$RecArchive = "PP-OCRv5_mobile_rec_infer.tar"

# Output model names
$DetOnnx = "ch_PP-OCRv5_det_infer.onnx"
$RecOnnx = "ch_PP-OCRv5_rec_infer.onnx"

# ONNX opset version (12 is recommended for OpenVINO 2024.0.0+)
$OpsetVersion = 12

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "PP-OCRv5 Model Conversion Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project root: $ProjectRoot"
Write-Host "Archive directory: $ArchiveDir"
Write-Host "Output directory: $ModelsDir"
Write-Host ""

# Check if archive directory exists
if (-not (Test-Path $ArchiveDir)) {
    Write-Host "ERROR: Archive directory not found: $ArchiveDir" -ForegroundColor Red
    Write-Host "Please create the 'archive' directory and place model tar files in it."
    exit 1
}

# Check if detection model archive exists
$DetArchivePath = Join-Path $ArchiveDir $DetArchive
if (-not (Test-Path $DetArchivePath)) {
    Write-Host "ERROR: Detection model archive not found: $DetArchivePath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Download from:"
    Write-Host "  https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/PP-OCRv5_mobile_det_infer.tar"
    exit 1
}

# Check if recognition model archive exists
$RecArchivePath = Join-Path $ArchiveDir $RecArchive
if (-not (Test-Path $RecArchivePath)) {
    Write-Host "ERROR: Recognition model archive not found: $RecArchivePath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Download from:"
    Write-Host "  https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/PP-OCRv5_mobile_rec_infer.tar"
    exit 1
}

# Check if paddle2onnx is installed
$paddle2onnxCheck = & python -c "import paddle2onnx; print('PADDLE2ONNX_OK')" 2>&1 | Out-String
if ($paddle2onnxCheck -notlike "*PADDLE2ONNX_OK*") {
    Write-Host "ERROR: paddle2onnx is not installed" -ForegroundColor Red
    Write-Host "Install with: pip install paddle2onnx>=1.0.0"
    exit 1
}
Write-Host "paddle2onnx check: OK" -ForegroundColor Green

# Create directories
Write-Host "Creating directories..."
New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

# Function to extract tar files
function Expand-TarFile {
    param (
        [string]$ArchivePath,
        [string]$DestinationPath
    )
    
    # Try using tar command (available on Windows 10+)
    try {
        tar -xf $ArchivePath -C $DestinationPath
    } catch {
        # Fallback to Python
        python -c "import tarfile; tarfile.open('$ArchivePath').extractall('$DestinationPath')"
    }
}

# Extract detection model
Write-Host ""
Write-Host "Extracting detection model..."
Expand-TarFile -ArchivePath $DetArchivePath -DestinationPath $TempDir

# Find the extracted directory
$DetDir = Get-ChildItem -Path $TempDir -Directory -Filter "*det*" | Select-Object -First 1
if ($null -eq $DetDir) {
    $DetDir = Join-Path $TempDir "PP-OCRv5_mobile_det_infer"
} else {
    $DetDir = $DetDir.FullName
}
Write-Host "Detection model directory: $DetDir"

# Convert detection model to ONNX
Write-Host ""
Write-Host "Converting detection model to ONNX..."
$DetOutputPath = Join-Path $ModelsDir $DetOnnx

python -m paddle2onnx.convert `
    --model_dir "$DetDir" `
    --model_filename "inference.json" `
    --params_filename "inference.pdiparams" `
    --save_file "$DetOutputPath" `
    --opset_version $OpsetVersion `
    --enable_onnx_checker True

Write-Host "Detection model saved to: $DetOutputPath" -ForegroundColor Green

# Extract recognition model
Write-Host ""
Write-Host "Extracting recognition model..."
Expand-TarFile -ArchivePath $RecArchivePath -DestinationPath $TempDir

# Find the extracted directory
$RecDir = Get-ChildItem -Path $TempDir -Directory -Filter "*rec*" | Select-Object -First 1
if ($null -eq $RecDir) {
    $RecDir = Join-Path $TempDir "PP-OCRv5_mobile_rec_infer"
} else {
    $RecDir = $RecDir.FullName
}
Write-Host "Recognition model directory: $RecDir"

# Convert recognition model to ONNX
Write-Host ""
Write-Host "Converting recognition model to ONNX..."
$RecOutputPath = Join-Path $ModelsDir $RecOnnx

python -m paddle2onnx.convert `
    --model_dir "$RecDir" `
    --model_filename "inference.json" `
    --params_filename "inference.pdiparams" `
    --save_file "$RecOutputPath" `
    --opset_version $OpsetVersion `
    --enable_onnx_checker True

Write-Host "Recognition model saved to: $RecOutputPath" -ForegroundColor Green

# Clean up temporary directory
Write-Host ""
Write-Host "Cleaning up temporary files..."
Remove-Item -Path $TempDir -Recurse -Force

# Download character dictionary if not exists
$DictFile = Join-Path $ModelsDir "ppocr_keys_v1.txt"
if (-not (Test-Path $DictFile)) {
    Write-Host ""
    Write-Host "Downloading character dictionary..."
    $DictUrl = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt"
    Invoke-WebRequest -Uri $DictUrl -OutFile $DictFile
    Write-Host "Character dictionary saved to: $DictFile" -ForegroundColor Green
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Conversion completed successfully!" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Detection model:  $DetOutputPath"
Write-Host "Recognition model: $RecOutputPath"
Write-Host "Character dict:   $DictFile"
Write-Host ""
Write-Host "You can now run the OCR pipeline:"
Write-Host "  python scripts/ocr_pipeline.py --image samples/test.jpg"
