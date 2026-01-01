import sys
import os
import cv2
import numpy as np
import time
import queue
from typing import Generator, Tuple, Dict
from pathlib import Path

# Add backend to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # backend/
sys.path.insert(0, parent_dir)

# Import your existing modules
try:
    from common.hailo_inference import HailoInfer
    from common.toolbox import FrameRateTracker, get_labels, load_json_file
    from preprocessing.preprocess import preprocess_for_streamlit
    from postprocessing.postprocess import inference_result_handler
    from common.tracker.ByteTrack.byte_tracker import BYTETracker
    from types import SimpleNamespace
except ImportError as e:
    print(f"Import Error: {e}")
    print("Current sys.path:", sys.path)
    raise


class LiveInferenceStream:
    """
    Live inference stream optimized for Streamlit.
    Uses the EXACT SAME pattern as your batch pipeline.
    """

    def __init__(self,
                 hef_path: str,
                 labels_path: str,
                 config_path: str,
                 enable_tracking: bool = True,
                 device_id: str = "SHARED"):

        self.hef_path = hef_path
        self.labels_path = labels_path
        self.config_path = config_path
        self.enable_tracking = enable_tracking
        
        # Initialize Hailo inference engine - SAME as run_inference.py
        self.hailo_infer = HailoInfer(hef_path, batch_size=1)
        self.input_height, self.input_width, _ = self.hailo_infer.get_input_shape()

        # Load labels and config
        self.labels = get_labels(labels_path)
        self.config_data = load_json_file(config_path)

        # Initialize tracker if enabled
        self.tracker = None
        if enable_tracking:
            tracker_config = self.config_data.get("visualization_params", {}).get("tracker", {})
            self.tracker = BYTETracker(SimpleNamespace(**tracker_config))

        # Performance tracking
        self.fps_tracker = FrameRateTracker()
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process single frame - optimized for Streamlit.
        """
        # Start timing
        start_time = time.time()

        # Preprocess
        processed = preprocess_for_streamlit(frame, self.input_width, self.input_height)

        # Run inference - uses the SAME pattern as batch pipeline
        inference_result = self._run_inference(processed)

        # Postprocess - uses your existing postprocess function
        annotated_frame = inference_result_handler(
            original_frame=frame,
            infer_results=inference_result,
            labels=self.labels,
            config_data=self.config_data,
            tracker=self.tracker,
            draw_trail=False
        )

        # Convert BGR to RGB for Streamlit
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Update stats
        self.frame_count += 1
        self.fps_tracker.increment()

        # Prepare statistics
        stats = {
            'frame_number': self.frame_count,
            'processing_time_ms': processing_time * 1000,
            'fps': self.fps_tracker.fps,
            'timestamp': time.strftime("%H:%M:%S")
        }

        return annotated_rgb, stats

    def _run_inference(self, processed_frame: np.ndarray):
        """
        Run inference EXACTLY like Hailo examples using HailoInfer.run()
        This is the CORRECT version that matches your batch pipeline!
        """
        result_queue = queue.Queue()

        def inference_callback(completion_info, bindings_list):
            if completion_info.exception:
                result_queue.put(("error", completion_info.exception))
                return

            bindings = bindings_list[0]  # batch_size = 1

            if len(bindings._output_names) == 1:
                result = bindings.output().get_buffer()
            else:
                result = {
                    name: np.expand_dims(
                        bindings.output(name).get_buffer(), axis=0
                    )
                    for name in bindings._output_names
                }

            result_queue.put(("success", result))

        # ✅ WICHTIG: Verwende HailoInfer.run() genau wie in der Batch-Pipeline
        self.hailo_infer.run(
            [processed_frame],  # batch = 1
            inference_callback
        )

        try:
            status, result = result_queue.get(timeout=30.0)
        except queue.Empty:
            raise TimeoutError("Inference timed out")

        if status == "error":
            raise RuntimeError(result)

        return result

    def stream_video(self, video_path: str) -> Generator[Tuple[np.ndarray, Dict], None, None]:
        """
        Generator that streams video frames.
        """
        cap = cv2.VideoCapture(video_path)
        self.fps_tracker.start()

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                annotated_frame, stats = self.process_frame(frame)
                yield annotated_frame, stats

        finally:
            cap.release()
            self.hailo_infer.close()

    def get_video_info(self, video_path: str) -> Dict:
        """
        Extract video information.
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': fps,
            'frame_count': frame_count,
            'duration': frame_count / fps if fps > 0 else 0
        }

        cap.release()
        return info
    
    def reset(self):
        """
        Reset the stream for new video processing.
        """
        self.frame_count = 0
        self.fps_tracker = FrameRateTracker()
        
        # Re-initialize tracker if tracking is enabled
        if self.tracker and self.enable_tracking:
            tracker_config = self.config_data.get("visualization_params", {}).get("tracker", {})
            self.tracker = BYTETracker(SimpleNamespace(**tracker_config))
    
    def close(self):
        """
        Clean up resources.
        """
        if hasattr(self, 'hailo_infer') and self.hailo_infer:
            self.hailo_infer.close()
