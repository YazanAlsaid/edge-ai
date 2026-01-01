from typing import List, Generator, Optional, Tuple, Dict, Callable ,Any
from pathlib import Path
from loguru import logger
import json
import os
import sys
import numpy as np
import queue
import cv2
import time


def load_json_file(path: str) -> Dict[str, Any]:
    """
    Loads and parses a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: Parsed contents of the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        OSError: If the file cannot be read.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON format in file '{path}': {e.msg}", e.doc, e.pos)

    return data

def init_input_source(input_path: str):
    if not any(input_path.lower().endswith(ext) for ext in ('.mp4', '.avi', '.mov', '.mkv')):
        raise ValueError("Only video files are supported (.mp4, .avi, .mov, .mkv)")

    if not os.path.exists(input_path):
        raise FileNotFoundError("Video not found: {input_path}")

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("Failed to open video: {input_path}")

    return cap

def generate_color(class_id: int) -> tuple:
    """
    Generate a unique color for a given class ID.

    Args:
        class_id (int): The class ID to generate a color for.

    Returns:
        tuple: A tuple representing an RGB color.
    """
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, size=3).tolist())

def get_labels(labels_path: str) -> list:
        """
        Load labels from a file.

        Args:
            labels_path (str): Path to the labels file.

        Returns:
            list: List of class names.
        """
        with open(labels_path, 'r', encoding="utf-8") as f:
            class_names = f.read().splitlines()
        return class_names


def id_to_color(idx):
    np.random.seed(idx)
    return np.random.randint(0, 255, size=3, dtype=np.uint8)

####################################################################
# Frame Rate Tracker
####################################################################

class FrameRateTracker:
    def __init__(self):
        self._count = 0
        self._start_time = None

    def start(self) -> None:
        self._start_time = time.time()

    def increment(self, n: int = 1) -> None:
        self._count += n

    @property
    def count(self) -> int:
        return self._count

    @property
    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def fps(self) -> float:
        elapsed = self.elapsed
        return self._count / elapsed if elapsed > 0 else 0.0

    def frame_rate_summary(self) -> str:
        return f"Processed {self.count} frames at {self.fps:.2f} FPS"
