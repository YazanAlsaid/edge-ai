import streamlit as st
import os
import time

from layout import sidebar_layout, main_layout
from video_inference import run_video_inference_from_upload

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_PATH = "../backend/models/yolo11s_kitti_quant.hef"
LABELS_PATH = "../backend/configs/kitti_labels.txt"
CONFIG_PATH = "../backend/configs/config.json"

st.set_page_config(layout="wide")

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "result_video_path" not in st.session_state:
    st.session_state.result_video_path = None

if "inference_stats" not in st.session_state:
    st.session_state.inference_stats = None

# --------------------------------------------------
# UI
# --------------------------------------------------
uploaded_video, enable_tracking, show_fps, run, clear = sidebar_layout()

# --------------------------------------------------
# CLEAR BUTTON
# --------------------------------------------------
if clear:
    if st.session_state.result_video_path and os.path.exists(st.session_state.result_video_path):
        os.unlink(st.session_state.result_video_path)

    st.session_state.result_video_path = None
    st.session_state.inference_stats = None
    st.experimental_rerun()

# --------------------------------------------------
# RUN INFERENCE
# --------------------------------------------------
if uploaded_video and run:
    start_time = time.time()

    with st.spinner("Inference läuft – bitte warten..."):
        result_path, stats = run_video_inference_from_upload(
            uploaded_file=uploaded_video,
            hef_path=MODEL_PATH,
            labels_path=LABELS_PATH,
            config_path=CONFIG_PATH,
            enable_tracking=enable_tracking,
            return_stats=True
        )

    total_time = time.time() - start_time

    st.session_state.result_video_path = result_path
    st.session_state.inference_stats = {
        "fps": stats["frames"] / total_time if total_time > 0 else 0,
        "frames": stats["frames"],
        "time": total_time,
    }

    st.success("Inference abgeschlossen")

# --------------------------------------------------
# MAIN VIEW
# --------------------------------------------------
main_layout(
    result_video_path=st.session_state.result_video_path,
    show_fps=show_fps,
    inference_stats=st.session_state.inference_stats
)
