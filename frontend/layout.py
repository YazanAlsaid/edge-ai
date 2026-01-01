import streamlit as st


def sidebar_layout():
    st.sidebar.header("⚙️ Einstellungen")

    uploaded_video = st.sidebar.file_uploader(
        "Video hochladen",
        type=["mp4", "avi", "mov"]
    )

    enable_tracking = st.sidebar.checkbox(
        "Tracking aktivieren",
        value=True
    )

    show_fps = st.sidebar.checkbox(
        "FPS anzeigen",
        value=True
    )

    run_button = st.sidebar.button("▶ Run Inference")
    clear_button = st.sidebar.button("🧹 Clear Ergebnis")

    return uploaded_video, enable_tracking, show_fps, run_button, clear_button


def main_layout(result_video_path, show_fps, inference_stats=None):
    st.title("Video Inference")

    if result_video_path:
        tabs = st.tabs(["🎞 Ergebnis Video", "📐 Depth (später)", "🧭 BEV (später)"])

        # ---------------- Video Tab ----------------
        with tabs[0]:
            st.subheader("Ergebnis Video")
            st.video(result_video_path)

            if show_fps and inference_stats:
                st.markdown(
                    f"""
                    **Durchschnittliche FPS:** {inference_stats['fps']:.2f}  
                    **Gesamtframes:** {inference_stats['frames']}  
                    **Gesamtzeit:** {inference_stats['time']:.2f} s
                    """
                )

        # ---------------- Placeholder Tabs ----------------
        with tabs[1]:
            st.info("Depth View")

        with tabs[2]:
            st.info("Bird-Eye View")
