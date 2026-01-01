import cv2
import tempfile
import os

from inference.live_inference_stream import LiveInferenceStream


def run_inference_full_video(
    input_video_path: str,
    hef_path: str,
    labels_path: str,
    config_path: str,
    enable_tracking: bool = True,
):
    """
    Führt Inferenz auf dem KOMPLETTEN Video aus
    und gibt (output_video_path, frame_count) zurück.
    """

    stream = LiveInferenceStream(
        hef_path=hef_path,
        labels_path=labels_path,
        config_path=config_path,
        enable_tracking=enable_tracking,
    )

    # 🔑 WICHTIG: Tracker für jeden Run zurücksetzen
    stream.reset()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_video_path = tmp_out.name
    tmp_out.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps if fps > 0 else 10,
        (width, height)
    )

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_rgb, _ = stream.process_frame(frame)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            writer.write(annotated_bgr)

            frame_count += 1

    finally:
        cap.release()
        writer.release()
        stream.close()

    return output_video_path, frame_count
