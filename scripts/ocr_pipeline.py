"""
OCR Pipeline Script.

This script provides a command-line interface for running OCR
on images using the OpenVINO-based TextExtractor.

Usage:
    python scripts/ocr_pipeline.py --image path/to/image.jpg
    python scripts/ocr_pipeline.py --input-dir samples/ --output-dir output/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from core.text_extract import TextExtractor
from utils.config_loader import ConfigLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run OCR on images using OpenVINO"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a single image file"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="samples",
        help="Directory containing input images (default: samples)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for output results (default: output)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/application.json",
        help="Path to configuration file (default: config/application.json)"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization images with bounding boxes"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def load_image(image_path: Path) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        image_path: Path to image file.
        
    Returns:
        BGR image as numpy array.
        
    Raises:
        FileNotFoundError: If image does not exist.
        ValueError: If image cannot be read.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(str(image_path))
    
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    return image


def save_results(
    results: list,
    output_path: Path,
    image_name: str
) -> None:
    """
    Save OCR results to JSON file.
    
    Args:
        results: List of OCR results.
        output_path: Output directory path.
        image_name: Original image filename.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_file = output_path / f"{Path(image_name).stem}_result.json"
    
    output_data = {
        "source_image": image_name,
        "text_regions": results,
        "total_regions": len(results)
    }
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to: {json_file}")


def visualize_results(
    image: np.ndarray,
    results: list,
    output_path: Path,
    image_name: str
) -> None:
    """
    Draw bounding boxes and text on image and save.
    
    Args:
        image: Original BGR image.
        results: List of OCR results.
        output_path: Output directory path.
        image_name: Original image filename.
    """
    vis_image = image.copy()
    
    for result in results:
        bbox = np.array(result["bbox"], dtype=np.int32)
        text = result["text"]
        score = result["score"]
        
        # Draw bounding box
        cv2.polylines(vis_image, [bbox], True, (0, 255, 0), 2)
        
        # Draw text label
        label = f"{text} ({score:.2f})"
        x, y = int(bbox[0][0]), int(bbox[0][1]) - 5
        cv2.putText(
            vis_image, label, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
    
    output_path.mkdir(parents=True, exist_ok=True)
    vis_file = output_path / f"{Path(image_name).stem}_vis.jpg"
    cv2.imwrite(str(vis_file), vis_image)
    
    logger.info(f"Visualization saved to: {vis_file}")


def create_extractor(config_loader: ConfigLoader) -> TextExtractor:
    """
    Create TextExtractor instance from configuration.
    
    Args:
        config_loader: Configuration loader instance.
        
    Returns:
        Initialized TextExtractor.
    """
    det_config = config_loader.get_detection_config()
    rec_config = config_loader.get_recognition_config()
    ov_config = config_loader.get_openvino_config()
    
    # Resolve model paths - use detectionModelPath/recognitionModelPath/characterDictPath from JSON
    models_config = config_loader.config.get("models", {})
    det_model_path = str(
        config_loader.project_root / models_config.get("detectionModelPath", "")
    )
    rec_model_path = str(
        config_loader.project_root / models_config.get("recognitionModelPath", "")
    )
    char_dict_path = str(
        config_loader.project_root / models_config.get("characterDictPath", "")
    )
    
    extractor = TextExtractor(
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        char_dict_path=char_dict_path,
        # Detection parameters
        det_limit_type=det_config.limit_type,
        det_limit_side_len=det_config.limit_side_len,
        det_thresh=det_config.thresh,
        det_box_thresh=det_config.box_thresh,
        det_unclip_ratio=det_config.unclip_ratio,
        # Recognition parameters
        rec_image_height=rec_config.image_height,
        rec_max_width=rec_config.max_width,
        rec_batch_size=rec_config.batch_size,
        rec_score_thresh=rec_config.score_thresh,
        # OpenVINO parameters
        device=ov_config.device,
        num_threads=ov_config.num_threads,
        num_streams=ov_config.num_streams,
        performance_hint=ov_config.performance_hint,
        enable_hyper_threading=ov_config.enable_hyper_threading,
        enable_cpu_pinning=ov_config.enable_cpu_pinning,
        cache_dir=ov_config.cache_dir
    )
    
    return extractor


def process_image(
    image_path: Path,
    extractor: TextExtractor,
    output_path: Path,
    visualize: bool = False
) -> list:
    """
    Process a single image through the OCR pipeline.
    
    Args:
        image_path: Path to input image.
        extractor: TextExtractor instance.
        output_path: Output directory path.
        visualize: Whether to save visualization.
        
    Returns:
        List of OCR results.
    """
    logger.info(f"Processing: {image_path}")
    
    # Load image
    image = load_image(image_path)
    
    # Run OCR
    results = extractor.extract(image)
    
    # Save results
    save_results(results, output_path, image_path.name)
    
    # Save visualization if requested
    if visualize:
        visualize_results(image, results, output_path, image_path.name)
    
    # Print extracted text
    for i, result in enumerate(results, 1):
        logger.info(f"  [{i}] {result['text']} (score: {result['score']:.3f})")
    
    return results


def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args()
    
    # Set verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config_path = PROJECT_ROOT / args.config
    logger.info(f"Loading configuration from: {config_path}")
    
    try:
        config_loader = ConfigLoader(str(config_path))
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    
    # Create extractor
    try:
        extractor = create_extractor(config_loader)
    except Exception as e:
        logger.error(f"Failed to create extractor: {e}")
        return 1
    
    # Determine output directory
    output_config = config_loader.get_output_config()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    
    # Process images
    if args.image:
        # Process single image
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        
        try:
            process_image(image_path, extractor, output_dir, args.visualize)
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return 1
    else:
        # Process all images in input directory
        input_dir = Path(args.input_dir)
        if not input_dir.is_absolute():
            input_dir = PROJECT_ROOT / input_dir
        
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return 1
        
        # Supported image formats
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = [
            f for f in input_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        if len(image_files) == 0:
            logger.warning(f"No images found in: {input_dir}")
            return 0
        
        logger.info(f"Found {len(image_files)} images to process")
        
        success_count = 0
        for image_path in image_files:
            try:
                process_image(image_path, extractor, output_dir, args.visualize)
                success_count += 1
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
        
        logger.info(f"Processed {success_count}/{len(image_files)} images successfully")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
