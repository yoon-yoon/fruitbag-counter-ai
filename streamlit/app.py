import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from io import BytesIO

from video_analyzer_yolo import convert_to_h264, analyze_video_with_tracking  # YOLO + counting 분석 함수

st.set_page_config(page_title="YOLO Line-Counter", layout="centered") # 740px
# st.set_page_config(page_title="YOLO Line-Counter", layout="wide")
st.title("🔍 YOLO + Line Crossing Counting")
st.markdown("""
        <style>
        body {
            background-color: #e6e6fa; /* 연보라색 */
        }

        [data-testid="stAppViewContainer"] {
            background-color: #e6e6fa; /* 전체 페이지 배경 */
        }

        [data-testid="stHeader"] {
            background-color: #e6e6fa;
        }

        [data-testid="stSidebar"] {
            background-color: #d8bfd8; /* 사이드바 배경 */
        }

        .stMarkdown, .stText, .stSlider {
            color: #000000; /* 글자 색 */
            font-size: 18px;
        }
        </style>
    """, unsafe_allow_html=True)


st.subheader("🎥 분석 영상 업로드")
video_file = st.file_uploader("영상을 업로드하세요:)", type=["mp4", "avi", "mov"])

line_coords = []
start_sec, end_sec = 0.0, 0.0
fps = 30

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()

    st.markdown("---")
    st.subheader("🖊️ Line 설정")

    if success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        canvas_width, canvas_height = frame_pil.size

        # 원하는 캔버스 너비 (예: 740px)
        target_width = 700
        orig_width, orig_height = frame_pil.size
        scale_ratio = target_width / orig_width
        canvas_width = target_width
        canvas_height = int(orig_height * scale_ratio)

        # st.image(frame_rgb, caption="🎯 첫 프레임", use_column_width=True)

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=3,
            stroke_color="#00f",
            background_image=frame_pil.resize((canvas_width, canvas_height)),
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="line",
            key="canvas"
        )

        # 좌표 복원 (scale up to original resolution)
        if canvas_result.json_data and canvas_result.json_data["objects"]:
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "line":
                    left = obj["left"]
                    top = obj["top"]
                    x1 = (left + obj["x1"]) / scale_ratio
                    y1 = (top + obj["y1"]) / scale_ratio
                    x2 = (left + obj["x2"]) / scale_ratio
                    y2 = (top + obj["y2"]) / scale_ratio
                    line_coords.append(((x1, y1), (x2, y2)))

            st.success(f"✅ 총 {len(line_coords)}개의 선이 설정되었습니다.")
        else:
            st.info("선을 그려주세요!")

        st.markdown("---")

        # st.video(video_path)  # 이게 슬라이더보다 위에 있어야 함
        print(f"line_coords : {line_coords}")

        st.subheader("⏱️ 분석 구간 선택")
        start_sec, end_sec = st.slider(
            "분석할 영상 구간 (초 단위)",
            min_value=0.0,
            max_value=round(duration, 2),
            value=(0.0, round(duration, 2)),
            step=0.1
        )

        st.markdown("---")

        # 모델 클래스 이름 미리 정의 또는 model 불러와서 가져오기
        # 예시: COCO 클래스
        coco_class_names = [
            "pear_bag", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        
        # 🎯 사용자 클래스 선택
        st.subheader("👀 추적할 객체 클래스 선택")
        selected_class_names = st.multiselect("클래스를 하나 이상 선택하세요", coco_class_names, default=["person"])
        wanted_class_ids = [i for i, name in enumerate(coco_class_names) if name in selected_class_names]

        st.markdown("---")
        st.subheader("🤖 AI 분석")
        
if video_file and line_coords and st.button("📊 분석 시작"):
    with st.spinner("YOLO 분석 중..."):
        result = analyze_video_with_tracking(
            # video_path, line_coords, start_sec, end_sec, fps, wanted_class_ids
            video_path, line_coords, start_sec, end_sec, fps, selected_class_names
        )

        print(f"result 1 : {result}")
        st.session_state["analyzed"] = True
        st.session_state["result"] = result

if st.session_state.get("analyzed"):
    result = st.session_state["result"]

    st.markdown("---")

    import io

    st.markdown("### ✅ 분석 결과 요약")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔍 추적된 객체 수", f"{result['total_tracked_objects']}개")
    with col2:
        st.metric("🚦 Line 건넌 횟수", f"{result['line_cross_count']}회")

    st.markdown("---")
    st.video(result['output_video_path'])

    # ✅ 분석 결과 텍스트 생성 (문자열 → 바이트)
    summary_text = f"""📄 분석 결과 요약
    --------------------------
    🔍 총 추적된 객체 수: {result['total_tracked_objects']}개
    🚦 총 Line 건넌 횟수: {result['line_cross_count']}회
    """
    summary_bytes = summary_text.encode('utf-8')  # ⚠️ 여기 중요

    # ✅ 텍스트 파일 다운로드
    st.download_button(
        label="📄 분석 결과 텍스트 다운로드",
        data=summary_bytes,
        file_name="analysis_summary.txt",
        mime="text/plain"
    )

    # 🎞️ 결과 영상 다운로드
    with open(result['output_video_path'], "rb") as f:
        video_bytes = f.read()

    st.download_button(
        label="📥 결과 영상 다운로드",
        data=video_bytes,
        file_name="result_video.mp4",
        mime="video/mp4"
    )
