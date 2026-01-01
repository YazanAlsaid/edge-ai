import tempfile
import os
import sys

sys.path.append(os.path.abspath("../backend"))
from inference.run_inference import run_inference_full_video


def run_video_inference_from_upload(
    uploaded_file,
    hef_path: str,
    labels_path: str,
    config_path: str,
    enable_tracking: bool = True,
    return_stats: bool = False,
):
    """
    Streamlit Upload -> TEMP Video -> Backend Inference
    """

    # -------- TEMP INPUT --------
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in.write(uploaded_file.read())
        input_video_path = tmp_in.name

    try:
        # -------- BACKEND --------
        result = run_inference_full_video(
            input_video_path=input_video_path,
            hef_path=hef_path,
            labels_path=labels_path,
            config_path=config_path,
            enable_tracking=enable_tracking,
        )

    finally:
        if os.path.exists(input_video_path):
            os.unlink(input_video_path)

    # -------- RETURN --------
    if return_stats:
        result_video_path, frame_count = result
        return result_video_path, {"frames": frame_count}

    return result[0]
