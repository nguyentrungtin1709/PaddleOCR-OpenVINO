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
        rec_batch_size: int = 4,
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
        
        # Step 5: Process label structure - classify and filter boxes
        image_height = image.shape[0]
        final_boxes, box_types = self._process_label_structure(boxes, image_height, image)
        
        if len(final_boxes) == 0:
            logger.info("No valid boxes after label structure processing")
            timing_info["total_ms"] = timing_info["detection_ms"]
            return [], timing_info, crops
        
        # Step 6: Crop text regions from final boxes
        filtered_boxes, crops, filtered_types = self._crop_text_regions(image, final_boxes, box_types)
        
        if len(crops) == 0:
            logger.info("No valid text crops after filtering")
            timing_info["total_ms"] = timing_info["detection_ms"]
            return [], timing_info, crops
        
        # Step 7: Run recognition
        texts, scores = self._run_recognition(crops)
        
        # End recognition timing
        t2 = time.perf_counter()
        timing_info["recognition_ms"] = (t2 - t1) * 1000
        timing_info["total_ms"] = (t2 - t0) * 1000
        
        # Step 8: Combine results - include all filtered boxes
        # Use filtered_boxes which matches 1:1 with texts and scores
        results = []
        for i, (box, text, score, box_type) in enumerate(zip(filtered_boxes, texts, scores, filtered_types)):
            # Display <None> for empty text
            display_text = text if text.strip() else "<None>"
            results.append({
                "bbox": box.tolist(),
                "text": display_text,
                "score": float(score),
                "source": box_type  # 'detected' or 'created'
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
    
    def _process_label_structure(
        self,
        boxes: np.ndarray,
        image_height: int,
        image: np.ndarray
    ) -> tuple:
        """
        Process label structure to extract fixed 4 boxes.
        
        Label structure:
        - Upper region (35%): Position info (e.g., "1/1") + noise
        - Lower region (65%): Product code, size, color
        
        Args:
            boxes: Sorted array of detected boxes.
            image_height: Height of the image.
            image: Original image for creating intermediate box.
            
        Returns:
            Tuple of (boxes, box_types):
                - boxes: Array of 4 boxes [position, product, size, color]
                - box_types: List of strings ['detected' or 'created']
        """
        # Classify boxes into upper and lower regions
        upper_boxes, lower_boxes = self._classify_boxes_by_region(boxes, image_height)
        
        # Process upper region - filter noise, get 1 main box
        position_box = self._filter_upper_region(upper_boxes)
        
        # Process lower region - ensure 3 boxes (create intermediate if needed)
        lower_final_boxes, lower_box_types = self._process_lower_region(lower_boxes, image)
        
        # Combine results
        final_boxes = []
        box_types = []
        if position_box is not None:
            final_boxes.append(position_box)
            box_types.append('detected')
        final_boxes.extend(lower_final_boxes)
        box_types.extend(lower_box_types)
        
        if len(final_boxes) == 0:
            return np.array([]), []
        
        return np.array(final_boxes, dtype=np.float32), box_types
    
    def _classify_boxes_by_region(
        self,
        boxes: np.ndarray,
        image_height: int,
        upper_ratio: float = 0.35,
        threshold: float = 0.40
    ) -> tuple:
        """
        Classify boxes into upper (35%) and lower (65%) regions.
        
        A box belongs to upper region if at least 40% of its area
        is above the 35% mark.
        
        Args:
            boxes: Array of boxes.
            image_height: Height of the image.
            upper_ratio: Ratio defining upper region (default 0.35).
            threshold: Minimum ratio of box above mark to be upper (default 0.40).
            
        Returns:
            Tuple of (upper_boxes, lower_boxes) as lists.
        """
        mark_y = image_height * upper_ratio
        upper_boxes = []
        lower_boxes = []
        
        for box in boxes:
            # Get min and max y coordinates of the box
            min_y = np.min(box[:, 1])
            max_y = np.max(box[:, 1])
            box_height = max_y - min_y
            
            if box_height <= 0:
                continue
            
            # Calculate portion of box above the mark
            portion_above = max(0, mark_y - min_y)
            ratio_above = portion_above / box_height
            
            if ratio_above >= threshold:
                upper_boxes.append(box)
            else:
                lower_boxes.append(box)
        
        return upper_boxes, lower_boxes
    
    def _filter_upper_region(self, upper_boxes: list) -> Optional[np.ndarray]:
        """
        Filter noise in upper region and select the main position box.
        
        The main box is the one with the lowest bottom edge (closest to
        the 35% boundary), as noise tends to be higher up.
        
        Args:
            upper_boxes: List of boxes in upper region.
            
        Returns:
            The main position box, or None if no boxes.
        """
        if len(upper_boxes) == 0:
            return None
        
        if len(upper_boxes) == 1:
            return upper_boxes[0]
        
        # Find box with the largest bottom_y (lowest position)
        max_bottom_y = -1
        selected_box = None
        
        for box in upper_boxes:
            bottom_y = np.max(box[:, 1])  # Max y = bottom edge
            if bottom_y > max_bottom_y:
                max_bottom_y = bottom_y
                selected_box = box
        
        return selected_box
    
    def _process_lower_region(
        self,
        lower_boxes: list,
        image: np.ndarray,
        min_gap: float = 2.0
    ) -> tuple:
        """
        Process lower region boxes, creating intermediate box if needed.
        
        If only 2 boxes exist with a gap >= 2px between them,
        create an intermediate box for the missing size field.
        
        If 3 boxes exist and middle box is smaller than average of others,
        expand the middle box to match.
        
        Args:
            lower_boxes: List of boxes in lower region.
            image: Original image (for bounds checking).
            min_gap: Minimum gap to trigger intermediate box creation.
            
        Returns:
            Tuple of (boxes, box_types):
                - boxes: List of 3 boxes (or fewer if not enough detected)
                - box_types: List of strings ['detected' or 'created']
        """
        if len(lower_boxes) == 0:
            return [], []
        
        if len(lower_boxes) == 1:
            return [self._order_points(lower_boxes[0])], ['detected']
        
        # Sort all boxes by y-coordinate (top to bottom)
        sorted_boxes = sorted(lower_boxes, key=lambda b: np.min(b[:, 1]))
        
        if len(sorted_boxes) >= 3:
            # Take first 3 boxes (now correctly sorted top-to-bottom)
            boxes = sorted_boxes[:3]
            
            # Order points for all boxes
            ordered_boxes = [self._order_points(b) for b in boxes]
            
            # Check if middle box needs expansion
            expanded_middle = self._expand_middle_box_if_needed(
                ordered_boxes[0],  # box above (18000)
                ordered_boxes[1],  # middle box (size)
                ordered_boxes[2],  # box below (SAND)
                image
            )
            
            if expanded_middle is not None:
                return [ordered_boxes[0], expanded_middle, ordered_boxes[2]], ['detected', 'detected', 'detected']
            return ordered_boxes, ['detected', 'detected', 'detected']
        
        # Exactly 2 boxes - check if we need to create intermediate
        box_above = sorted_boxes[0]
        box_below = sorted_boxes[1]
        
        # Order points for both boxes to get consistent corners
        ordered_above = self._order_points(box_above)
        ordered_below = self._order_points(box_below)
        
        # Calculate gap between boxes
        # Bottom of upper box (BL and BR) to top of lower box (TL and TR)
        bottom_y_above = max(ordered_above[3][1], ordered_above[2][1])  # BL, BR
        top_y_below = min(ordered_below[0][1], ordered_below[1][1])      # TL, TR
        gap = top_y_below - bottom_y_above
        
        if gap < min_gap:
            # Gap too small, don't create intermediate box
            return [ordered_above, ordered_below], ['detected', 'detected']
        
        # Create intermediate box
        intermediate_box = self._create_intermediate_box(
            ordered_above, ordered_below, image
        )
        
        if intermediate_box is not None:
            # Insert intermediate box between the two
            return [ordered_above, intermediate_box, ordered_below], ['detected', 'created', 'detected']
        
        return [ordered_above, ordered_below], ['detected', 'detected']
    
    def _expand_middle_box_if_needed(
        self,
        box_above: np.ndarray,
        box_middle: np.ndarray,
        box_below: np.ndarray,
        image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Expand middle box to align left edge with surrounding boxes and maintain proper dimensions.
        
        Expansions applied:
        1. Left edge: Align to min(x_left of box_above, x_left of box_below) if middle is indented
        2. Height: Expand to average of surrounding boxes if too small
        3. Width: Expand to right if w < 1.5 * h
        
        Args:
            box_above: Ordered points [TL, TR, BR, BL] of upper box.
            box_middle: Ordered points [TL, TR, BR, BL] of middle box.
            box_below: Ordered points [TL, TR, BR, BL] of lower box.
            image: Original image for bounds checking.
            
        Returns:
            Expanded middle box, or None if no expansion needed.
        """
        try:
            # Calculate heights
            height_above = np.linalg.norm(box_above[3] - box_above[0])
            height_middle = np.linalg.norm(box_middle[3] - box_middle[0])
            height_below = np.linalg.norm(box_below[3] - box_below[0])
            avg_height = (height_above + height_below) / 2
            
            # Calculate current width
            width_middle = np.linalg.norm(box_middle[1] - box_middle[0])
            
            # Calculate left x-coordinates
            x_left_above = min(box_above[0][0], box_above[3][0])  # TL.x, BL.x
            x_left_middle = min(box_middle[0][0], box_middle[3][0])  # TL.x, BL.x
            x_left_below = min(box_below[0][0], box_below[3][0])  # TL.x, BL.x
            x_left_target = min(x_left_above, x_left_below)
            
            # Determine if expansion is needed
            need_height_expand = height_middle < avg_height
            need_left_expand = x_left_middle > x_left_target + 1  # Middle is indented (1px tolerance)
            new_height = avg_height if need_height_expand else height_middle
            
            # Check if width expansion is needed (w >= 1.5 * h)
            min_width = 1.5 * new_height
            need_width_expand = width_middle < min_width
            
            if not need_height_expand and not need_width_expand and not need_left_expand:
                return None
            
            # Get corners of middle box
            tl = box_middle[0].copy()
            tr = box_middle[1].copy()
            br = box_middle[2].copy()
            bl = box_middle[3].copy()
            
            # Expand left edge if needed (align to target x_left)
            if need_left_expand:
                left_expand = x_left_middle - x_left_target
                
                # Calculate horizontal direction (from TL to TR, normalized)
                width_dir = tr - tl
                width_len = np.linalg.norm(width_dir)
                if width_len < 1e-6:
                    width_dir = box_above[1] - box_above[0]
                    width_len = np.linalg.norm(width_dir)
                    if width_len < 1e-6:
                        return None
                width_dir = width_dir / width_len
                
                # Move left edge to the left (opposite of width direction)
                tl = tl - width_dir * left_expand
                bl = bl - width_dir * left_expand
                
                # Update width_middle after left expansion
                width_middle = np.linalg.norm(tr - tl)
            
            # Expand height if needed (equally top and bottom)
            if need_height_expand:
                expand_height = avg_height - height_middle
                expand_each = expand_height / 2
                
                # Calculate vertical direction (from TL to BL)
                height_dir = bl - tl
                height_len = np.linalg.norm(height_dir)
                if height_len < 1e-6:
                    return None
                height_dir = height_dir / height_len  # Normalize
                
                # Expand top edge upward (opposite of height direction)
                tl = tl - height_dir * expand_each
                tr = tr - height_dir * expand_each
                # Expand bottom edge downward (same as height direction)
                bl = bl + height_dir * expand_each
                br = br + height_dir * expand_each
            
            # Recalculate width after left expansion and check if right expansion still needed
            current_width = np.linalg.norm(tr - tl)
            min_width = 1.5 * new_height
            need_width_expand_right = current_width < min_width
            
            # Expand width to the right if needed
            if need_width_expand_right:
                expand_width = min_width - current_width
                
                # Calculate horizontal direction (from TL to TR)
                width_dir = tr - tl
                width_len = np.linalg.norm(width_dir)
                if width_len < 1e-6:
                    # Use direction from box_above if middle box is too narrow
                    width_dir = box_above[1] - box_above[0]
                    width_len = np.linalg.norm(width_dir)
                    if width_len < 1e-6:
                        return None
                width_dir = width_dir / width_len  # Normalize
                
                # Expand only to the right (keep left edge aligned)
                tr = tr + width_dir * expand_width
                br = br + width_dir * expand_width
            
            # Create expanded box
            expanded = np.array([tl, tr, br, bl], dtype=np.float32)
            
            # Clip to image boundaries
            h, w = image.shape[:2]
            expanded[:, 0] = np.clip(expanded[:, 0], 0, w)
            expanded[:, 1] = np.clip(expanded[:, 1], 0, h)
            
            return expanded
        except Exception:
            return None
    
    def _create_intermediate_box(
        self,
        box_above: np.ndarray,
        box_below: np.ndarray,
        image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Create an intermediate box between two boxes, left-aligned with the leftmost edge.
        
        The box is created between box_above and box_below with:
        - Left edge (TL.x, BL.x) aligned to the leftmost x of both boxes
        - Width = 1.5 × height
        - Height expanded to average if too small
        
        Args:
            box_above: Ordered points [TL, TR, BR, BL] of upper box.
            box_below: Ordered points [TL, TR, BR, BL] of lower box.
            image: Original image for bounds checking.
            
        Returns:
            New box as (4, 2) array [TL, TR, BR, BL], or None if invalid.
        """
        try:
            # Find the leftmost x-coordinate from both boxes
            x_left = min(
                box_above[0][0],  # TL of above
                box_above[3][0],  # BL of above
                box_below[0][0],  # TL of below
                box_below[3][0]   # BL of below
            )
            
            # Get y-coordinates for top and bottom edges
            # Top of intermediate = bottom of box_above
            tl_y = box_above[3][1]  # BL.y of above
            # Bottom of intermediate = top of box_below
            bl_y = box_below[0][1]  # TL.y of below
            
            # Calculate direction vector (horizontal, from box_above's bottom edge)
            direction = box_above[2] - box_above[3]  # BR - BL
            length = np.linalg.norm(direction)
            if length < 1e-6:
                return None
            direction = direction / length  # Normalize
            
            # Calculate heights
            current_height = abs(bl_y - tl_y)
            height_above = np.linalg.norm(box_above[3] - box_above[0])
            height_below = np.linalg.norm(box_below[3] - box_below[0])
            avg_height = (height_above + height_below) / 2
            
            # Use the larger of current height or average height
            final_height = max(current_height, avg_height)
            
            # Calculate width = 1.5 × height
            width = 1.5 * final_height
            
            # Create initial points with left-aligned x
            tl = np.array([x_left, tl_y], dtype=np.float32)
            bl = np.array([x_left, bl_y], dtype=np.float32)
            
            # If current height is smaller than average, expand equally
            if current_height < avg_height:
                expand_total = avg_height - current_height
                expand_each = expand_total / 2
                
                # Expand top edge upward, bottom edge downward
                tl[1] = tl[1] - expand_each
                bl[1] = bl[1] + expand_each
            
            # Create right edge points using horizontal direction
            tr = tl + direction * width
            br = bl + direction * width
            
            # Create box in order [TL, TR, BR, BL]
            intermediate = np.array([tl, tr, br, bl], dtype=np.float32)
            
            # Clip to image boundaries
            h, w = image.shape[:2]
            intermediate[:, 0] = np.clip(intermediate[:, 0], 0, w)
            intermediate[:, 1] = np.clip(intermediate[:, 1], 0, h)
            
            return intermediate
        except Exception:
            return None
    
    def _crop_text_regions(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        box_types: list = None
    ) -> tuple:
        """
        Crop and straighten text regions from image.
        
        Args:
            image: Original BGR image.
            boxes: Array of bounding boxes.
            box_types: List of box types ('detected' or 'created').
            
        Returns:
            Tuple of (filtered_boxes, crops, filtered_types):
                - filtered_boxes: Array of boxes that have valid crops
                - crops: List of cropped text region images
                - filtered_types: List of box types for valid crops
        """
        if box_types is None:
            box_types = ['detected'] * len(boxes)
            
        crops = []
        valid_indices = []
        
        for i, box in enumerate(boxes):
            crop = self._get_rotate_crop_image(image, box)
            if crop is not None and crop.size > 0:
                crops.append(crop)
                valid_indices.append(i)
        
        # Filter boxes and types to only include those with valid crops
        if len(valid_indices) > 0:
            filtered_boxes = boxes[valid_indices]
            filtered_types = [box_types[i] for i in valid_indices]
        else:
            filtered_boxes = np.array([])
            filtered_types = []
        
        return filtered_boxes, crops, filtered_types
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Orders 4 points in consistent order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
        
        This ensures correct perspective transform for text regions that may be
        slightly rotated. Assumes text is mostly horizontal (rotation < 45°).
        
        Args:
            pts: Array of shape (4, 2) with corner coordinates.
            
        Returns:
            Array of shape (4, 2) ordered as [TL, TR, BR, BL].
        """
        center = np.mean(pts, axis=0)
        
        def get_angle(p):
            return np.arctan2(p[1] - center[1], p[0] - center[0])
        
        sorted_pts = sorted(pts, key=get_angle)
        sorted_pts = np.array(sorted_pts, dtype="float32")
        
        sums = sorted_pts.sum(axis=1)
        top_left_idx = np.argmin(sums)
        ordered = np.roll(sorted_pts, -top_left_idx, axis=0)
        
        return ordered
    
    def _get_rotate_crop_image(
        self,
        image: np.ndarray,
        points: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Crop and straighten a text region to horizontal orientation.
        
        Args:
            image: Source image.
            points: Four corner points of the text region.
            
        Returns:
            Cropped and straightened image, or None if failed.
        """
        try:
            # Order points: TL, TR, BR, BL
            ordered = self._order_points(points.astype(np.float32))
            
            # Calculate width (TL-TR and BL-BR) and height (TL-BL and TR-BR)
            width = int(max(
                np.linalg.norm(ordered[0] - ordered[1]),  # TL to TR
                np.linalg.norm(ordered[3] - ordered[2])   # BL to BR
            ))
            height = int(max(
                np.linalg.norm(ordered[0] - ordered[3]),  # TL to BL
                np.linalg.norm(ordered[1] - ordered[2])   # TR to BR
            ))
            
            # Basic size check before expensive operations
            if width < 3 or height < 3:
                return None
            
            # Define destination points matching ordered source
            dst_pts = np.array([
                [0, 0],           # TL → top-left
                [width, 0],       # TR → top-right  
                [width, height],  # BR → bottom-right
                [0, height]       # BL → bottom-left
            ], dtype=np.float32)
            
            # Get perspective transform
            M = cv2.getPerspectiveTransform(ordered, dst_pts)
            
            # Apply transform
            cropped = cv2.warpPerspective(
                image, M, (width, height),
                borderMode=cv2.BORDER_REPLICATE
            )
            
            # No rotation needed - points are already ordered correctly
            # Text should now be horizontal and right-side up
            
            # Filter out regions that are too small for quality recognition
            h, w = cropped.shape[:2]
            # - height < 12: too small for any recognition
            # - width < height/2: too narrow for horizontal text
            if h < 12 or w < h / 2:
                return None
            
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
