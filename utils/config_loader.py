"""
Configuration loader for PaddleOCR OpenVINO pipeline.

This module provides dataclasses and a loader for reading configuration
from application.json. It follows the Dependency Inversion Principle (DIP)
by providing value objects that can be passed to core components.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class OpenVINOConfig:
    """OpenVINO Runtime configuration parameters."""
    device: str
    num_threads: int
    num_streams: int
    performance_hint: str
    enable_hyper_threading: bool
    enable_cpu_pinning: bool
    cache_dir: str


@dataclass(frozen=True)
class DetectionConfig:
    """Text detection model configuration parameters."""
    model_path: str
    limit_type: str
    limit_side_len: int
    thresh: float
    box_thresh: float
    unclip_ratio: float


@dataclass(frozen=True)
class RecognitionConfig:
    """Text recognition model configuration parameters."""
    model_path: str
    char_dict_path: str
    image_height: int
    max_width: int
    batch_size: int
    score_thresh: float


@dataclass(frozen=True)
class OutputConfig:
    """Output configuration parameters."""
    save_visualization: bool
    drop_score_threshold: float


class ConfigLoader:
    """
    Loads and parses configuration from application.json.
    
    This class reads the JSON configuration file and provides
    typed dataclass objects for each configuration section.
    
    Attributes:
        config_path: Path to the configuration file.
        _config: Parsed JSON configuration dictionary.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the application.json file.
            
        Raises:
            FileNotFoundError: If the configuration file does not exist.
            json.JSONDecodeError: If the configuration file is not valid JSON.
        """
        self.config_path = Path(config_path)
        self._base_dir = self.config_path.parent.parent  # Project root
        
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = json.load(f)
    
    def _resolve_path(self, relative_path: str) -> str:
        """
        Resolve a relative path to an absolute path based on project root.
        
        Args:
            relative_path: Path relative to project root.
            
        Returns:
            Absolute path as string.
        """
        return str(self._base_dir / relative_path)
    
    def get_openvino_config(self) -> OpenVINOConfig:
        """
        Get OpenVINO runtime configuration.
        
        Returns:
            OpenVINOConfig dataclass with runtime settings.
        """
        ov_cfg = self._config.get("openvino", {})
        return OpenVINOConfig(
            device=ov_cfg.get("device", "CPU"),
            num_threads=ov_cfg.get("numThreads", 2),
            num_streams=ov_cfg.get("numStreams", 1),
            performance_hint=ov_cfg.get("performanceHint", "LATENCY"),
            enable_hyper_threading=ov_cfg.get("enableHyperThreading", False),
            enable_cpu_pinning=ov_cfg.get("enableCpuPinning", True),
            cache_dir=ov_cfg.get("cacheDir", "")
        )
    
    def get_detection_config(self) -> DetectionConfig:
        """
        Get text detection model configuration.
        
        Returns:
            DetectionConfig dataclass with detection settings.
        """
        models_cfg = self._config.get("models", {})
        det_cfg = self._config.get("detection", {})
        
        return DetectionConfig(
            model_path=self._resolve_path(
                models_cfg.get("detectionModelPath", "models/det/inference.onnx")
            ),
            limit_type=det_cfg.get("limitType", "max"),
            limit_side_len=det_cfg.get("limitSideLen", 640),
            thresh=det_cfg.get("thresh", 0.15),
            box_thresh=det_cfg.get("boxThresh", 0.15),
            unclip_ratio=det_cfg.get("unclipRatio", 2.0)
        )
    
    def get_recognition_config(self) -> RecognitionConfig:
        """
        Get text recognition model configuration.
        
        Returns:
            RecognitionConfig dataclass with recognition settings.
        """
        models_cfg = self._config.get("models", {})
        rec_cfg = self._config.get("recognition", {})
        
        return RecognitionConfig(
            model_path=self._resolve_path(
                models_cfg.get("recognitionModelPath", "models/rec/inference.onnx")
            ),
            char_dict_path=self._resolve_path(
                models_cfg.get("characterDictPath", "fonts/ppocr_keys_v1.txt")
            ),
            image_height=rec_cfg.get("imageHeight", 48),
            max_width=rec_cfg.get("maxWidth", 320),
            batch_size=rec_cfg.get("batchSize", 6),
            score_thresh=rec_cfg.get("scoreThresh", 0.3)
        )
    
    def get_output_config(self) -> OutputConfig:
        """
        Get output configuration.
        
        Returns:
            OutputConfig dataclass with output settings.
        """
        out_cfg = self._config.get("output", {})
        return OutputConfig(
            save_visualization=out_cfg.get("saveVisualization", True),
            drop_score_threshold=out_cfg.get("dropScoreThreshold", 0.5)
        )
