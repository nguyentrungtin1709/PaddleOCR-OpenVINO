"""
Core text extraction engine using OpenVINO Runtime.

This module provides the TextExtractor class that performs OCR using
PP-OCRv5 models converted to ONNX format and executed via OpenVINO.

Design principles:
- Receives configuration values via constructor (DIP)
- No dependencies on external configuration modules
- Can be tested independently
"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

try:
    import openvino as ov
except ImportError:
    raise ImportError(
        "OpenVINO is required. Install with: pip install openvino>=2024.0.0"
    )

logger = logging.getLogger(__name__)


class TextExtractor:
    """
    Core OCR engine using OpenVINO Runtime.
    
    This class handles text detection and recognition using PP-OCRv5 models.
    It receives all configuration as constructor parameters to maintain
    independence from external configuration systems.
    
    Attributes:
        det_model_path: Path to detection ONNX model.
        rec_model_path: Path to recognition ONNX model.
        char_dict_path: Path to character dictionary file.
    """
    
    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        char_dict_path: str,
        # Detection parameters
        det_limit_type: str = "max",
        det_limit_side_len: int = 640,
        det_thresh: float = 0.15,
        det_box_thresh: float = 0.15,
        det_unclip_ratio: float = 2.0,
        # Recognition parameters
        rec_image_height: int = 48,
        rec_max_width: int = 320,
        rec_batch_size: int = 6,
        rec_score_thresh: float = 0.3,
        # OpenVINO parameters
        device: str = "CPU",
        num_threads: int = 2,
        num_streams: int = 1,
        performance_hint: str = "LATENCY",
        enable_hyper_threading: bool = False,
        enable_cpu_pinning: bool = True,
        cache_dir: str = ""
    ):
        """
        Initialize the TextExtractor with configuration values.
        
        Args:
            det_model_path: Path to detection ONNX model.
            rec_model_path: Path to recognition ONNX model.
            char_dict_path: Path to character dictionary file.
            det_limit_type: Resize limit type ("max" or "min").
            det_limit_side_len: Target side length for resize.
            det_thresh: Pixel threshold for text detection.
            det_box_thresh: Box threshold for filtering detections.
            det_unclip_ratio: Expansion ratio for text boxes.
            rec_image_height: Height for recognition input images.
            rec_max_width: Maximum width for recognition input.
            rec_batch_size: Batch size for recognition inference.
            rec_score_thresh: Minimum score for recognition results.
            device: OpenVINO device (CPU, GPU, AUTO).
            num_threads: Number of CPU threads for inference.
            num_streams: Number of parallel inference streams.
            performance_hint: Performance mode (LATENCY or THROUGHPUT).
            enable_hyper_threading: Whether to use hyper-threading.
            enable_cpu_pinning: Whether to pin threads to CPU cores.
            cache_dir: Directory for model caching.
        """
        # Store paths
        self.det_model_path = det_model_path
        self.rec_model_path = rec_model_path
        self.char_dict_path = char_dict_path
        
        # Detection parameters
        self.det_limit_type = det_limit_type
        self.det_limit_side_len = det_limit_side_len
        self.det_thresh = det_thresh
        self.det_box_thresh = det_box_thresh
        self.det_unclip_ratio = det_unclip_ratio
        
        # Recognition parameters
        self.rec_image_height = rec_image_height
        self.rec_max_width = rec_max_width
        self.rec_batch_size = rec_batch_size
        self.rec_score_thresh = rec_score_thresh
        
        # OpenVINO parameters
        self.device = device
        self.num_threads = num_threads
        self.num_streams = num_streams
        self.performance_hint = performance_hint
        self.enable_hyper_threading = enable_hyper_threading
        self.enable_cpu_pinning = enable_cpu_pinning
        self.cache_dir = cache_dir
        
        # Load character dictionary
        self.character_dict = self._load_character_dict()
        
        # Initialize OpenVINO
        self._init_openvino()
        
        logger.info("TextExtractor initialized successfully")
    
    def _load_character_dict(self) -> list:
        """
        Load character dictionary from file.
        
        Returns:
            List of characters for text decoding.
            
        Raises:
            FileNotFoundError: If dictionary file does not exist.
        """
        dict_path = Path(self.char_dict_path)
        if not dict_path.exists():
            raise FileNotFoundError(
                f"Character dictionary not found: {dict_path}"
            )
        
        with open(dict_path, "r", encoding="utf-8") as f:
            characters = [line.strip() for line in f.readlines()]
        
        # Add blank token at the beginning for CTC decoding
        characters = ["blank"] + characters
        
        logger.info(f"Loaded {len(characters)} characters from dictionary")
        return characters
    
    def _init_openvino(self) -> None:
        """
        Initialize OpenVINO Core and compile models.
        
        This method sets up the OpenVINO runtime with the specified
        configuration and compiles both detection and recognition models.
        """
        self.core = ov.Core()
        
        # Configure CPU properties if using CPU device
        if "CPU" in self.device.upper():
            cpu_config = {}
            
            if self.num_streams > 0:
                cpu_config["NUM_STREAMS"] = str(self.num_streams)
            
            if self.num_threads > 0:
                cpu_config["INFERENCE_NUM_THREADS"] = str(self.num_threads)
            
            # Note: ENABLE_HYPER_THREADING and AFFINITY are deprecated in OpenVINO 2024+
            # Only set if needed for older versions
            
            if cpu_config:
                self.core.set_property("CPU", cpu_config)
                logger.info(f"CPU configuration: {cpu_config}")
        
        # Compile configuration
        compile_config = {}
        if self.performance_hint:
            compile_config["PERFORMANCE_HINT"] = self.performance_hint
        if self.cache_dir:
            compile_config["CACHE_DIR"] = self.cache_dir
        
        # Load and compile detection model
        logger.info(f"Loading detection model: {self.det_model_path}")
        det_model = self.core.read_model(self.det_model_path)
        self.det_compiled = self.core.compile_model(
            det_model, self.device, compile_config
        )
        self.det_input_layer = self.det_compiled.input(0)
        self.det_output_layer = self.det_compiled.output(0)
        
        # Load and compile recognition model
        logger.info(f"Loading recognition model: {self.rec_model_path}")
        rec_model = self.core.read_model(self.rec_model_path)
        self.rec_compiled = self.core.compile_model(
            rec_model, self.device, compile_config
        )
        self.rec_input_layer = self.rec_compiled.input(0)
        self.rec_output_layer = self.rec_compiled.output(0)
        
        logger.info("Models compiled successfully")
    
    def extract(self, image: np.ndarray) -> tuple:
        """
        Perform OCR on an image.
        
        This method runs the full OCR pipeline:
        1. Preprocess image for detection
        2. Run detection inference
        3. Post-process to get bounding boxes
        4. Crop text regions
        5. Run recognition on each region
        6. Return combined results with timing info
        
        Args:
            image: BGR image as numpy array (H, W, C).
            
        Returns:
            Tuple of (results, timing_info, crops):
                - results: List of dictionaries with keys:
                    - bbox: List of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    - text: Recognized text string
                    - score: Confidence score (0-1)
                - timing_info: Dictionary with keys:
                    - detection_ms: Detection time in milliseconds
                    - recognition_ms: Recognition time in milliseconds
                    - total_ms: Total processing time in milliseconds
                - crops: List of cropped text region images (np.ndarray)
        """
        # Initialize timing info and empty crops
        timing_info = {
            "detection_ms": 0.0,
            "recognition_ms": 0.0,
            "total_ms": 0.0
        }
        crops = []
        
        if image is None or image.size == 0:
            logger.warning("Empty image provided")
            return [], timing_info, crops
        
        # Start timing for detection phase
        t0 = time.perf_counter()
        
        # Step 1: Preprocess for detection
        input_tensor, shape_info = self._preprocess_detection(image)
        
        # Step 2: Run detection
        det_output = self.det_compiled([input_tensor])[self.det_output_layer]
        
        # Step 3: Post-process detection
        boxes = self._postprocess_detection(det_output, shape_info)
        
        # End detection timing
        t1 = time.perf_counter()
        timing_info["detection_ms"] = (t1 - t0) * 1000
        
        if len(boxes) == 0:
            logger.info("No text detected")
            timing_info["total_ms"] = timing_info["detection_ms"]
            return [], timing_info, crops
        
        # Start timing for recognition phase
        # Step 4: Sort boxes (top-to-bottom, left-to-right)
        boxes = self._sort_boxes(boxes)
        
        # Step 5: Crop text regions
        crops = self._crop_text_regions(image, boxes)
        
        # Step 6: Run recognition
        texts, scores = self._run_recognition(crops)
        
        # End recognition timing
        t2 = time.perf_counter()
        timing_info["recognition_ms"] = (t2 - t1) * 1000
        timing_info["total_ms"] = (t2 - t0) * 1000
        
        # Step 7: Combine results
        results = []
        for i, (box, text, score) in enumerate(zip(boxes, texts, scores)):
            if score >= self.rec_score_thresh:
                results.append({
                    "bbox": box.tolist(),
                    "text": text,
                    "score": float(score)
                })
        
        logger.info(f"Extracted {len(results)} text regions")
        return results, timing_info, crops
    
    def _preprocess_detection(
        self,
        image: np.ndarray
    ) -> tuple:
        """
        Preprocess image for text detection.
        
        Args:
            image: BGR image (H, W, C).
            
        Returns:
            Tuple of (input_tensor, shape_info):
                - input_tensor: Normalized tensor (1, 3, H, W)
                - shape_info: Dictionary with original and resize info
        """
        src_h, src_w = image.shape[:2]
        
        # Calculate resize ratio based on limit type
        if self.det_limit_type == "max":
            ratio = min(self.det_limit_side_len / max(src_h, src_w), 1.0)
        else:  # "min"
            ratio = max(self.det_limit_side_len / min(src_h, src_w), 1.0)
        
        new_h = int(src_h * ratio)
        new_w = int(src_w * ratio)
        
        # Ensure dimensions are divisible by 32 (required by DB model)
        new_h = max(32, (new_h // 32) * 32)
        new_w = max(32, (new_w // 32) * 32)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Convert BGR to RGB and normalize
        img = resized[:, :, ::-1].astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # Transpose HWC to CHW
        img = img.transpose(2, 0, 1)
        
        # Add batch dimension
        input_tensor = np.expand_dims(img, 0).astype(np.float32)
        
        shape_info = {
            "src_h": src_h,
            "src_w": src_w,
            "resize_h": new_h,
            "resize_w": new_w,
            "ratio_h": new_h / src_h,
            "ratio_w": new_w / src_w
        }
        
        return input_tensor, shape_info
    
    def _postprocess_detection(
        self,
        output: np.ndarray,
        shape_info: dict
    ) -> np.ndarray:
        """
        Post-process detection output to get bounding boxes.
        
        Args:
            output: Model output tensor (1, 1, H, W).
            shape_info: Dictionary with resize information.
            
        Returns:
            Array of bounding boxes, each as (4, 2) array of corner points.
        """
        pred = output[0, 0]  # Remove batch and channel dims
        
        # Apply threshold to get binary mask
        segmentation = pred > self.det_thresh
        
        # Find contours
        contours, _ = cv2.findContours(
            (segmentation * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        boxes = []
        for contour in contours:
            # Skip small contours
            if len(contour) < 4:
                continue
            
            # Get minimum area rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.float32)
            
            # Calculate box score (average of prediction values inside box)
            box_score = self._calculate_box_score(pred, contour)
            
            if box_score < self.det_box_thresh:
                continue
            
            # Expand box using unclip ratio
            box = self._unclip_box(box, self.det_unclip_ratio)
            
            if box is None:
                continue
            
            # Get new minimum area rectangle after unclipping
            rect = cv2.minAreaRect(box)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.float32)
            
            # Skip boxes that are too small
            if min(rect[1]) < 3:
                continue
            
            boxes.append(box)
        
        if len(boxes) == 0:
            return np.array([])
        
        boxes = np.array(boxes)
        
        # Scale boxes back to original image size
        boxes[:, :, 0] = boxes[:, :, 0] / shape_info["ratio_w"]
        boxes[:, :, 1] = boxes[:, :, 1] / shape_info["ratio_h"]
        
        # Clip to image boundaries
        boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, shape_info["src_w"])
        boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, shape_info["src_h"])
        
        return boxes
    
    def _calculate_box_score(
        self,
        pred: np.ndarray,
        contour: np.ndarray
    ) -> float:
        """
        Calculate the average prediction score inside a contour.
        
        Args:
            pred: Prediction map (H, W).
            contour: Contour points.
            
        Returns:
            Average score inside the contour.
        """
        h, w = pred.shape
        
        # Get bounding rectangle
        x, y, rect_w, rect_h = cv2.boundingRect(contour)
        
        # Clip to image bounds
        x_min = max(0, x)
        y_min = max(0, y)
        x_max = min(w, x + rect_w)
        y_max = min(h, y + rect_h)
        
        # Create mask for the contour
        mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
        contour_shifted = contour.copy()
        contour_shifted[:, :, 0] = contour[:, :, 0] - x_min
        contour_shifted[:, :, 1] = contour[:, :, 1] - y_min
        cv2.fillPoly(mask, [contour_shifted.astype(np.int32)], 1)
        
        # Calculate mean score
        roi = pred[y_min:y_max, x_min:x_max]
        score = np.sum(roi * mask) / (np.sum(mask) + 1e-6)
        
        return float(score)
    
    def _unclip_box(
        self,
        box: np.ndarray,
        unclip_ratio: float
    ) -> Optional[np.ndarray]:
        """
        Expand a box using the Vatti clipping algorithm.
        
        Args:
            box: Box corners as (4, 2) array.
            unclip_ratio: Expansion ratio.
            
        Returns:
            Expanded box as array, or None if expansion fails.
        """
        try:
            poly = Polygon(box)
            if not poly.is_valid or poly.area < 1:
                return None
            
            distance = poly.area * unclip_ratio / poly.length
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(
                box.astype(np.int32).tolist(),
                pyclipper.JT_ROUND,
                pyclipper.ET_CLOSEDPOLYGON
            )
            expanded = offset.Execute(distance)
            
            if len(expanded) == 0:
                return None
            
            return np.array(expanded[0], dtype=np.float32)
        except Exception:
            return None
    
    def _sort_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """
        Sort boxes from top-to-bottom, left-to-right.
        
        Args:
            boxes: Array of boxes, each as (4, 2) array.
            
        Returns:
            Sorted boxes array.
        """
        if len(boxes) == 0:
            return boxes
        
        # Sort by y-coordinate of top-left corner, then by x-coordinate
        indices = np.lexsort((boxes[:, 0, 0], boxes[:, 0, 1]))
        return boxes[indices]
    
    def _crop_text_regions(
        self,
        image: np.ndarray,
        boxes: np.ndarray
    ) -> list:
        """
        Crop and straighten text regions from image.
        
        Args:
            image: Original BGR image.
            boxes: Array of bounding boxes.
            
        Returns:
            List of cropped text region images.
        """
        crops = []
        
        for box in boxes:
            crop = self._get_rotate_crop_image(image, box)
            if crop is not None and crop.size > 0:
                crops.append(crop)
        
        return crops
    
    def _get_rotate_crop_image(
        self,
        image: np.ndarray,
        points: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Crop and rotate a text region to horizontal orientation.
        
        Args:
            image: Source image.
            points: Four corner points of the text region.
            
        Returns:
            Cropped and rotated image, or None if failed.
        """
        try:
            points = points.astype(np.float32)
            
            # Calculate width and height
            width = int(max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])
            ))
            height = int(max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])
            ))
            
            # Filter out regions that are too small for quality recognition:
            # - height < 24: would require >2x upscaling (48/24=2x max)
            # - width < height/2: too vertical/narrow for horizontal text
            if height < 24 or width < height / 2:
                return None
            
            # Define destination points
            dst_pts = np.array([
                [0, 0],
                [width, 0],
                [width, height],
                [0, height]
            ], dtype=np.float32)
            
            # Get perspective transform
            M = cv2.getPerspectiveTransform(points, dst_pts)
            
            # Apply transform
            cropped = cv2.warpPerspective(
                image, M, (width, height),
                borderMode=cv2.BORDER_REPLICATE
            )
            
            # If height > width * 1.5, it's vertical text - rotate to horizontal
            if height > width * 1.5:
                cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            return cropped
        except Exception:
            return None
    
    def _run_recognition(
        self,
        crops: list
    ) -> tuple:
        """
        Run recognition on cropped text regions.
        
        Args:
            crops: List of cropped text images.
            
        Returns:
            Tuple of (texts, scores) lists.
        """
        if len(crops) == 0:
            return [], []
        
        # Sort by width ratio for efficient batching
        width_ratios = [c.shape[1] / float(c.shape[0]) for c in crops]
        indices = np.argsort(np.array(width_ratios))
        
        # Initialize results
        rec_results = [("", 0.0)] * len(crops)
        
        # Process in batches
        for batch_start in range(0, len(crops), self.rec_batch_size):
            batch_end = min(batch_start + self.rec_batch_size, len(crops))
            batch_indices = indices[batch_start:batch_end]
            batch_crops = [crops[idx] for idx in batch_indices]
            
            # Preprocess batch
            batch_tensor = self._preprocess_recognition_batch(batch_crops)
            
            # Run inference
            rec_output = self.rec_compiled([batch_tensor])[self.rec_output_layer]
            
            # Decode results
            batch_texts, batch_scores = self._decode_recognition(rec_output)
            
            # Store results at original indices
            for i, (text, score) in enumerate(zip(batch_texts, batch_scores)):
                rec_results[batch_indices[i]] = (text, score)
        
        texts = [r[0] for r in rec_results]
        scores = [r[1] for r in rec_results]
        
        return texts, scores
    
    def _preprocess_recognition_batch(
        self,
        crops: list
    ) -> np.ndarray:
        """
        Preprocess a batch of cropped images for recognition.
        
        Args:
            crops: List of cropped text images.
            
        Returns:
            Batch tensor of shape (N, 3, H, W).
        """
        # Calculate max width ratio in this batch
        max_wh_ratio = max(c.shape[1] / float(c.shape[0]) for c in crops)
        target_width = min(
            int(self.rec_image_height * max_wh_ratio),
            self.rec_max_width
        )
        
        batch_tensors = []
        
        for crop in crops:
            # Resize to target height while maintaining aspect ratio
            h, w = crop.shape[:2]
            ratio = self.rec_image_height / float(h)
            resized_w = min(int(w * ratio), target_width)
            
            resized = cv2.resize(crop, (resized_w, self.rec_image_height))
            
            # Convert to float and normalize
            img = resized.astype(np.float32) / 255.0
            img = (img - 0.5) / 0.5
            
            # Transpose HWC to CHW
            img = img.transpose(2, 0, 1)
            
            # Pad to target width
            padded = np.zeros(
                (3, self.rec_image_height, target_width),
                dtype=np.float32
            )
            padded[:, :, :resized_w] = img
            
            batch_tensors.append(padded)
        
        return np.stack(batch_tensors, axis=0)
    
    def _decode_recognition(
        self,
        output: np.ndarray
    ) -> tuple:
        """
        Decode recognition output using CTC decoding.
        
        Args:
            output: Model output tensor (N, T, C).
            
        Returns:
            Tuple of (texts, scores) lists.
        """
        texts = []
        scores = []
        
        # Get predictions (argmax over character dimension)
        preds = np.argmax(output, axis=2)
        
        for batch_idx in range(preds.shape[0]):
            pred = preds[batch_idx]
            pred_scores = output[batch_idx]
            
            # CTC decode: remove duplicates and blanks
            text_indices = []
            text_scores = []
            prev_idx = 0
            
            for t in range(len(pred)):
                idx = pred[t]
                if idx != 0 and idx != prev_idx:  # 0 is blank token
                    text_indices.append(idx)
                    text_scores.append(pred_scores[t, idx])
                prev_idx = idx
            
            # Convert indices to characters
            if len(text_indices) > 0:
                text = ""
                for idx in text_indices:
                    if idx < len(self.character_dict):
                        text += self.character_dict[idx]
                score = float(np.mean(text_scores))
            else:
                text = ""
                score = 0.0
            
            texts.append(text)
            scores.append(score)
        
        return texts, scores
