import cv2
import numpy as np
import subprocess
from ultralytics import YOLO
from shapely.geometry import LineString
from collections import defaultdict
import os
import time


def convert_to_h264(input_path, output_path):
    """ffmpeg로 H.264 코덱으로 변환"""
    try:
        subprocess.run([
            "/home/jovyan/anaconda3/envs/yolo_env/ffmpeg", "-y", "-i", input_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            output_path
        ], check=True)
        print(f"[INFO] 변환 완료: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg 변환 실패: {e}")


def generate_timestamp_filename(base_dir=".", prefix="analyzed"):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)
    full_stamp = f"{timestamp}_{ms:03d}"
    raw_path = os.path.join(base_dir, f"{prefix}_{full_stamp}_raw.mp4")
    h264_path = os.path.join(base_dir, f"{prefix}_{full_stamp}_h264.mp4")
    return raw_path, h264_path

def analyze_video_with_tracking(video_path, line_coords, start_sec, end_sec, fps, wanted_class_names):
    import streamlit as st

    # 모델 경로 결정
    is_pear_bag = "pear_bag" in wanted_class_names
    model_path = "../ai/runs/detect/train7/weights/best.pt" if is_pear_bag else "./yolo11n.pt"
    model = YOLO(model_path)

    # 클래스 이름 → ID 변환
    model_class_names = model.names  # e.g., {0: 'person', 1: 'car', ...}
    name_to_id = {v: k for k, v in model_class_names.items()}
    wanted_class_ids = [name_to_id[name] for name in wanted_class_names if name in name_to_id]

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    total_frames = end_frame - start_frame + 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    raw_out_path, final_out_path = generate_timestamp_filename(base_dir="./temp_video")
    out = cv2.VideoWriter(raw_out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    track_history = defaultdict(list)
    counted_ids = set()
    line_geoms = [LineString([p1, p2]) for p1, p2 in line_coords]
    total_tracked_ids = set()
    line_cross_count = 0

    # Streamlit 진행바
    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_idx = start_frame
    processed = 0

    while cap.isOpened() and frame_idx <= end_frame:
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, classes=wanted_class_ids if wanted_class_ids else None)
        frame_idx += 1
        if not results:
            continue

        result = results[0]
        boxes = result.boxes
        annotated_frame = frame.copy()

        if boxes and boxes.id is not None:
            cls = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.int().cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()

            for box, class_id, track_id in zip(xyxy, cls, ids):
                if class_id not in wanted_class_ids:
                    continue

                x1, y1, x2, y2 = map(int, box)
                label = f"{model_class_names[class_id]} {int(track_id)}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Tracking path 및 Line Crossing
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                total_tracked_ids.add(track_id)
                track = track_history[track_id]
                if track:
                    last_point = track[-1]
                    movement_line = LineString([last_point, (cx, cy)])
                    for line in line_geoms:
                        if movement_line.intersects(line) and (track_id, line.wkt) not in counted_ids:
                            line_cross_count += 1
                            counted_ids.add((track_id, line.wkt))
                track.append((cx, cy))
                if len(track) > 30:
                    track.pop(0)

                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 255, 255), thickness=2)

        # 선 및 카운팅 정보 시각화
        for p1, p2 in line_coords:
            cv2.line(annotated_frame, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 255), 2)

        cv2.rectangle(annotated_frame, (10, 10), (500, 80), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"Tracked IDs: {len(total_tracked_ids)}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Line Cross Count: {line_cross_count}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        out.write(annotated_frame)

        # Streamlit 진행 표시
        processed += 1
        progress_bar.progress(min(processed / total_frames, 1.0))
        status_text.text(f"🔄 ({processed}/{total_frames} 프레임 완료)")

    cap.release()
    out.release()
    convert_to_h264(raw_out_path, final_out_path)

    progress_bar.progress(1.0)
    status_text.text("✅ 분석 완료!")

    return {
        "total_tracked_objects": len(total_tracked_ids),
        "line_cross_count": line_cross_count,
        "output_video_path": final_out_path
    }




# def analyze_video_with_tracking(video_path, line_coords, start_sec, end_sec, fps, wanted_class_names):
#     import streamlit as st  # 함수 내부에서 임포트해야 에러 없음

#     is_pear_bag = "pear_bag" in wanted_class_names
#     model_path = "../ai/runs/detect/train3/weights/best.pt" if is_pear_bag else "./yolo11n.pt"
#     model = YOLO(model_path)

#     model_class_names = model.names
#     name_to_id = {v: k for k, v in model_class_names.items()}
#     wanted_class_ids = [name_to_id[name] for name in wanted_class_names if name in name_to_id]

#     cap = cv2.VideoCapture(video_path)
#     assert cap.isOpened(), "Error reading video file"

#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     start_frame = int(start_sec * fps)
#     end_frame = int(end_sec * fps)
#     total_frames = end_frame - start_frame + 1
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

#     raw_out_path, final_out_path = generate_timestamp_filename(base_dir="./temp_video")
#     out = cv2.VideoWriter(raw_out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

#     track_history = defaultdict(list)
#     counted_ids = set()
#     line_geoms = [LineString([p1, p2]) for p1, p2 in line_coords]
#     total_tracked_ids = set()
#     line_cross_count = 0

#     # Streamlit 진행바 & 상태 텍스트
#     progress_bar = st.progress(0)
#     status_text = st.empty()

#     frame_idx = start_frame
#     processed = 0
#     while cap.isOpened() and frame_idx <= end_frame:
#         success, frame = cap.read()
#         if not success:
#             break

#         results = model.track(frame, persist=True, classes=wanted_class_ids if wanted_class_ids else None)
#         frame_idx += 1  # 위치 조심!

#         if not results:
#             continue

#         result = results[0]
#         boxes = result.boxes
#         annotated_frame = result.plot()

#         if boxes and boxes.id is not None:
#             xywh = boxes.xywh.cpu().numpy()
#             ids = boxes.id.int().cpu().numpy()
#             for (x, y, w, h), track_id in zip(xywh, ids):
#                 cx, cy = float(x), float(y)
#                 total_tracked_ids.add(track_id)
#                 track = track_history[track_id]
#                 if track:
#                     last_point = track[-1]
#                     movement_line = LineString([last_point, (cx, cy)])
#                     for line in line_geoms:
#                         if movement_line.intersects(line) and (track_id, line.wkt) not in counted_ids:
#                             line_cross_count += 1
#                             counted_ids.add((track_id, line.wkt))
#                 track.append((cx, cy))
#                 if len(track) > 30:
#                     track.pop(0)

#                 points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
#                 cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 255, 255), thickness=2)

#         for p1, p2 in line_coords:
#             cv2.line(annotated_frame, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 255), 2)

#         cv2.rectangle(annotated_frame, (10, 10), (320, 80), (0, 0, 0), -1)
#         cv2.putText(annotated_frame, f"Tracked IDs: {len(total_tracked_ids)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
#         cv2.putText(annotated_frame, f"Line Cross Count: {line_cross_count}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

#         out.write(annotated_frame)

#         # 🔄 진행 상태 업데이트
#         processed += 1
#         progress_bar.progress(min(processed / total_frames, 1.0))
#         status_text.text(f"🔄 분석 중... ({processed}/{total_frames} 프레임 완료)")

#     cap.release()
#     out.release()
#     convert_to_h264(raw_out_path, final_out_path)

#     # 🔚 완료 표시
#     progress_bar.progress(1.0)
#     status_text.text("✅ 분석 완료!")

#     return {
#         "total_tracked_objects": len(total_tracked_ids),
#         "line_cross_count": line_cross_count,
#         "output_video_path": final_out_path
#     }