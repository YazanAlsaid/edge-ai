import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from common.toolbox import id_to_color


def inference_result_handler(original_frame: np.ndarray, infer_results: list,
                            labels: List[str], config_data: Dict,
                            tracker: Optional = None,
                            draw_trail: bool = False) -> np.ndarray:
    """
    Haupt-Postprocessing-Funktion.
    Gibt annotiertes Frame zurück.
    """
    from .detection_utils import extract_detections, draw_detections

    # Detections extrahieren
    detections = extract_detections(original_frame, infer_results, config_data)

    # Frame mit Detections/Tracking zeichnen
    annotated_frame = draw_detections(
        detections, original_frame, labels, tracker, draw_trail
    )

    return annotated_frame

