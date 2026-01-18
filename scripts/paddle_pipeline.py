"""
PaddleOCR Pipeline Script.

This script provides a command-line interface for running OCR
on images using the native PaddleOCR library for benchmarking
against the OpenVINO implementation.

Usage:
    python scripts/paddle_pipeline.py --image path/to/image.jpg
    python scripts/paddle_pipeline.py --input-dir samples/ --output-dir output/paddle/
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
        description="Run OCR on images using PaddleOCR (for benchmarking)"
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
        default="output/paddle",
        help="Directory for output results (default: output/paddle)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/paddle.json",
        help="Path to configuration file (default: config/paddle.json)"
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


def load_config(config_path: Path) -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_ocr_engine(config: dict):
    """
    Create PaddleOCR engine with configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        PaddleOCR engine instance.
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        logger.error(
            "PaddleOCR not installed. Please install: "
            "pip install paddlepaddle paddleocr>=3.0.0"
        )
        raise e
    
    # Extract configuration
    models = config.get("models", {})
    detection = config.get("detection", {})
    recognition = config.get("recognition", {})
    runtime = config.get("runtime", {})
    pipeline = config.get("pipeline", {})
    
    # Build PaddleOCR parameters
    ocr_params = {
        # Language
        "lang": runtime.get("lang", "en"),
        
        # Model names
        "text_detection_model_name": models.get("textDetectionModelName", "PP-OCRv5_mobile_det"),
        "text_recognition_model_name": models.get("textRecognitionModelName", "PP-OCRv5_mobile_rec"),
        
        # Detection parameters
        "text_det_limit_type": detection.get("textDetLimitType", "max"),
        "text_det_limit_side_len": detection.get("textDetLimitSideLen", 640),
        "text_det_thresh": detection.get("textDetThresh", 0.15),
        "text_det_box_thresh": detection.get("textDetBoxThresh", 0.15),
        "text_det_unclip_ratio": detection.get("textDetUnclipRatio", 2.0),
        
        # Recognition parameters
        "text_rec_score_thresh": recognition.get("textRecScoreThresh", 0.3),
        
        # Runtime parameters
        "device": runtime.get("device", "cpu"),
        "cpu_threads": runtime.get("cpuThreads", 2),
        "enable_mkldnn": runtime.get("enableMkldnn", True),
        "mkldnn_cache_capacity": runtime.get("mkldnnCacheCapacity", 10),
        "precision": runtime.get("precision", "fp32"),
        
        # Pipeline modules (disable for fair comparison)
        "use_doc_orientation_classify": pipeline.get("useDocOrientationClassify", False),
        "use_doc_unwarping": pipeline.get("useDocUnwarping", False),
        "use_textline_orientation": pipeline.get("useTextlineOrientation", False),
    }
    
    logger.info("=" * 60)
    logger.info("PaddleOCR Configuration:")
    logger.info("=" * 60)
    for key, value in ocr_params.items():
        logger.info(f"  {key:35s} = {value}")
    logger.info("=" * 60)
    
    return PaddleOCR(**ocr_params)


def create_dummy_images(count: int = 2) -> list:
    """
    Create dummy images for warm-up phase.
    
    Images contain text patterns for the model to process,
    simulating real-world inference conditions.
    
    Args:
        count: Number of dummy images to create.
        
    Returns:
        List of numpy arrays (BGR images).
    """
    dummy_images = []
    
    for i in range(count):
        # Create white background 640x480
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Add text patterns for model to detect
        cv2.putText(
            img, f"WARMUP-{i+1}", (150, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 4
        )
        cv2.putText(
            img, "TEST-12345", (130, 280),
            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (50, 50, 50), 3
        )
        cv2.putText(
            img, "ABCDEFGH", (160, 370),
            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (80, 80, 80), 3
        )
        
        dummy_images.append(img)
    
    return dummy_images


def run_warmup(ocr_engine, count: int = 2) -> None:
    """
    Run warm-up phase with dummy images.
    
    This initializes model caches and JIT compilation,
    ensuring subsequent benchmark runs are representative.
    
    Args:
        ocr_engine: PaddleOCR engine instance.
        count: Number of warm-up iterations.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"WARM-UP PHASE: Running {count} dummy images...")
    logger.info("=" * 60)
    
    dummy_images = create_dummy_images(count)
    
    for i, img in enumerate(dummy_images, 1):
        t0 = time.perf_counter()
        _ = ocr_engine.predict(img)
        t1 = time.perf_counter()
        logger.info(f"  Warm-up {i}/{count} completed in {(t1-t0)*1000:.2f} ms")
    
    logger.info("Warm-up completed. Starting benchmark...")
    logger.info("=" * 60)
    logger.info("")


def parse_paddle_results(raw_result) -> list:
    """
    Parse PaddleOCR 3.x output format to standard format.
    
    Args:
        raw_result: Raw output from ocr.predict().
        
    Returns:
        List of dictionaries with keys: bbox, text, score.
    
    Note:
        PaddleOCR returns dt_polys (all detected boxes) and rec_polys (boxes after 
        score filtering). We use rec_polys to match with rec_texts/rec_scores.
    """
    results = []
    
    if not raw_result:
        return results
    
    for res in raw_result:
        if not isinstance(res, dict):
            continue
            
        rec_texts = res.get('rec_texts', [])
        rec_scores = res.get('rec_scores', [])
        # Use rec_polys instead of dt_polys for correct mapping
        # rec_polys contains only boxes that passed the score threshold
        rec_polys = res.get('rec_polys', res.get('dt_polys', []))
        
        for i, text in enumerate(rec_texts):
            score = rec_scores[i] if i < len(rec_scores) else 0.0
            bbox = rec_polys[i].tolist() if i < len(rec_polys) else []
            
            # Skip empty text
            display_text = str(text).strip() if text else "<None>"
            if not display_text:
                display_text = "<None>"
            
            results.append({
                "bbox": bbox,
                "text": display_text,
                "score": float(score)
            })
    
    return results


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
    
    logger.debug(f"Results saved to: {json_file}")


def visualize_results(
    image: np.ndarray,
    results: list,
    output_path: Path,
    image_name: str
) -> None:
    """
    Draw bounding boxes and text on image and save.
    
    Uses the same style as OpenVINO pipeline for fair comparison:
    - GREEN boxes for detected regions
    - DARK_ORANGE text labels at top-right of each box
    
    Args:
        image: Original BGR image.
        results: List of OCR results.
        output_path: Output directory path.
        image_name: Original image filename.
    """
    vis_image = image.copy()
    
    # Colors in BGR format (matching OpenVINO pipeline style)
    DARK_ORANGE = (0, 140, 255)   # Text label color
    COLOR_DETECTED = (0, 255, 0)  # Green for detected boxes
    
    for result in results:
        bbox = result.get("bbox", [])
        text = result.get("text", "")
        score = result.get("score", 0.0)
        
        if not bbox or len(bbox) < 4:
            continue
        
        bbox_np = np.array(bbox, dtype=np.int32)
        
        # Draw bounding box
        cv2.polylines(vis_image, [bbox_np], True, COLOR_DETECTED, 2)
        
        # Draw text label at top-right corner, outside the box
        label = f"{text} ({score:.2f})"
        
        # Find actual top-right corner (max x, min y)
        max_x = int(np.max(bbox_np[:, 0]))
        min_y = int(np.min(bbox_np[:, 1]))
        
        # Position: right of box + padding, aligned with top edge
        x = max_x + 5
        y = min_y + 12
        
        # Draw text with dark orange color and bold font
        cv2.putText(
            vis_image, label, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, DARK_ORANGE, 2
        )
    
    output_path.mkdir(parents=True, exist_ok=True)
    vis_file = output_path / f"{Path(image_name).stem}_vis.jpg"
    cv2.imwrite(str(vis_file), vis_image)
    
    logger.debug(f"Visualization saved to: {vis_file}")


def process_image(
    image_path: Path,
    ocr_engine,
    output_path: Path,
    visualize: bool = False
) -> tuple:
    """
    Process a single image through the OCR pipeline.
    
    Args:
        image_path: Path to input image.
        ocr_engine: PaddleOCR engine instance.
        output_path: Output directory path.
        visualize: Whether to save visualization.
        
    Returns:
        Tuple of (results, timing_info, image_size):
            - results: List of OCR results
            - timing_info: Dictionary with timing information
            - image_size: Tuple of (width, height)
    """
    # Load image
    image = load_image(image_path)
    image_size = (image.shape[1], image.shape[0])  # (width, height)
    
    # Run OCR with timing
    t_start = time.perf_counter()
    raw_result = ocr_engine.predict(image)
    t_end = time.perf_counter()
    
    timing_info = {
        "total_ms": (t_end - t_start) * 1000
    }
    
    # Parse results
    results = parse_paddle_results(raw_result)
    
    # Save results
    save_results(results, output_path, image_path.name)
    
    # Save visualization if requested
    if visualize:
        visualize_results(image, results, output_path, image_path.name)
    
    # Print extracted text
    for i, result in enumerate(results, 1):
        logger.info(f"  [{i}] {result['text']} (score: {result['score']:.3f})")
    
    return results, timing_info, image_size


def format_results_for_csv(results: list) -> str:
    """
    Format OCR results as a string for CSV output.
    
    Args:
        results: List of OCR result dictionaries.
        
    Returns:
        Formatted string like "text1 (0.9523), text2 (0.8912)"
    """
    if len(results) == 0:
        return ""
    
    return ", ".join(
        f"{r['text']} ({r['score']:.4f})"
        for r in results
    )


def print_summary(timing_list: list) -> None:
    """
    Print benchmark summary statistics.
    
    Args:
        timing_list: List of total_ms values.
    """
    if not timing_list:
        return
    
    times = np.array(timing_list)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total images processed: {len(times)}")
    logger.info(f"  Average time:           {np.mean(times):.2f} ms")
    logger.info(f"  Minimum time:           {np.min(times):.2f} ms")
    logger.info(f"  Maximum time:           {np.max(times):.2f} ms")
    logger.info(f"  Std deviation:          {np.std(times):.2f} ms")
    logger.info(f"  Throughput:             {1000.0 / np.mean(times):.2f} images/sec")
    logger.info("=" * 60)


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
        config = load_config(config_path)
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    
    # Create OCR engine
    try:
        ocr_engine = create_ocr_engine(config)
    except Exception as e:
        logger.error(f"Failed to create OCR engine: {e}")
        return 1
    
    # Run warm-up if enabled
    warmup_config = config.get("warmup", {})
    if warmup_config.get("enabled", True):
        warmup_count = warmup_config.get("count", 2)
        run_warmup(ocr_engine, warmup_count)
    
    # Determine output directory
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize CSV writer if debug enabled
    debug_config = config.get("debug", {})
    csv_file = None
    csv_writer = None
    timing_list = []
    
    if debug_config.get("enabled", True):
        csv_path = output_dir / "details.csv"
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow([
            "filename",
            "image_width",
            "image_height",
            "num_regions",
            "total_ms",
            "results"
        ])
        logger.info(f"Debug mode enabled. Writing timing details to: {csv_path}")
    
    try:
        # Process images
        if args.image:
            # Process single image
            image_path = Path(args.image)
            if not image_path.is_absolute():
                image_path = PROJECT_ROOT / image_path
            
            try:
                results, timing_info, image_size = process_image(
                    image_path, ocr_engine, output_dir, args.visualize
                )
                
                timing_list.append(timing_info['total_ms'])
                
                # Write to CSV if debug enabled
                if csv_writer:
                    csv_writer.writerow([
                        image_path.name,
                        image_size[0],  # width
                        image_size[1],  # height
                        len(results),
                        f"{timing_info['total_ms']:.2f}",
                        format_results_for_csv(results)
                    ])
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
            image_files = sorted([
                f for f in input_dir.iterdir()
                if f.suffix.lower() in image_extensions
            ])
            
            if len(image_files) == 0:
                logger.warning(f"No images found in: {input_dir}")
                return 0
            
            logger.info(f"Found {len(image_files)} images to process")
            logger.info("")
            
            success_count = 0
            for idx, image_path in enumerate(image_files, 1):
                try:
                    logger.info(f"[{idx}/{len(image_files)}] Processing: {image_path.name}")
                    results, timing_info, image_size = process_image(
                        image_path, ocr_engine, output_dir, args.visualize
                    )
                    
                    timing_list.append(timing_info['total_ms'])
                    
                    # Write to CSV if debug enabled
                    if csv_writer:
                        csv_writer.writerow([
                            image_path.name,
                            image_size[0],  # width
                            image_size[1],  # height
                            len(results),
                            f"{timing_info['total_ms']:.2f}",
                            format_results_for_csv(results)
                        ])
                    
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
            
            logger.info("")
            logger.info(f"Processed {success_count}/{len(image_files)} images successfully")
        
        # Print summary
        print_summary(timing_list)
    
    finally:
        # Close CSV file if opened
        if csv_file:
            csv_file.close()
            logger.info(f"Debug CSV file saved.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
