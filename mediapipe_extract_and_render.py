import os
import cv2
import json
import argparse
from pathlib import Path
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

POSE_CONNECTIONS = mp_holistic.POSE_CONNECTIONS
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

# -------------------- FaceMesh contour loops (no cross bars) --------------------
# Mouth (outer + optional inner)
MOUTH_OUTER = [61, 146, 91, 181, 84, 314, 405, 321, 375, 291]        # closed loop
MOUTH_INNER = [78, 95, 88, 178, 87, 317, 402, 318, 324, 308]         # closed loop (optional)

# NEW: Upper lip (outer) as an open polyline (NO nose tip)
UPPER_LIP_LOOP = [61, 40, 37, 267, 270, 409, 291]

# Eyes (outer loops)
LEFT_EYE_LOOP  = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_LOOP = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

# Eyebrows (outer arcs — readable sets)
LEFT_BROW_LOOP  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW_LOOP = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

def normalized_to_pixel(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h), landmark.z)

def smooth_landmarks(prev_landmarks, cur_landmarks, alpha=0.8):
    if prev_landmarks is None:
        return cur_landmarks
    out = {}
    for k in cur_landmarks.keys():
        cur = cur_landmarks[k]
        prev = prev_landmarks.get(k)
        if cur is None:
            out[k] = None
            continue
        if prev is None:
            out[k] = cur
            continue
        new_list = []
        for pcur, pprev in zip(cur, prev):
            x = alpha * pprev[0] + (1 - alpha) * pcur[0]
            y = alpha * pprev[1] + (1 - alpha) * pcur[1]
            z = alpha * pprev[2] + (1 - alpha) * pcur[2]
            new_list.append((x, y, z))
        out[k] = new_list
    return out

def center_and_scale_landmarks(landmarks, reference_pairs=[(11, 12)]):
    pose = landmarks.get('pose')
    if pose is None:
        return landmarks
    def midpoint(a, b):
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    try:
        center = midpoint(pose[23], pose[24])
    except Exception:
        center = midpoint(pose[11], pose[12])
    scales = []
    for a, b in reference_pairs:
        pa, pb = pose[a], pose[b]
        scales.append(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
    s = float(np.mean(scales)) if len(scales) > 0 else 1.0
    if s == 0:
        s = 1.0
    out = {}
    for k, lst in landmarks.items():
        if lst is None:
            out[k] = None
            continue
        normed = [((p[0] - center[0]) / s, (p[1] - center[1]) / s, p[2] / s) for p in lst]
        out[k] = normed
    return out

# ---------------- Drawing ----------------
def draw_skeleton_on_transparent(img_w, img_h, landmarks_pixel, draw_face=False, draw_inner_mouth=True):
    canvas = np.zeros((img_h, img_w, 4), dtype=np.uint8)

    def draw_point(pt, radius=4):
        if pt is None:
            return
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), radius, (255, 255, 255, 255), -1)

    def draw_line(a, b, thickness=2):
        if a is None or b is None:
            return
        cv2.line(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 255, 255, 255), thickness)

    def draw_closed_loop(points, thickness=2):
        n = len(points)
        for i in range(n):
            a = points[i]
            b = points[(i + 1) % n]
            if a is None or b is None:
                continue
            cv2.line(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 255, 255, 255), thickness)

    def draw_open_polyline(points, thickness=2):
        for i in range(len(points) - 1):
            a = points[i]
            b = points[i + 1]
            if a is None or b is None:
                continue
            cv2.line(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 255, 255, 255), thickness)

    # Filter wrist–hand redundancies; also skip pose face so FaceMesh handles face
    EXCLUDED_POSE_CONNECTIONS = {
        (15, 17), (16, 18), (17, 19), (18, 20),
        (11, 15), (12, 16)
    }
    POSE_FACE_IDXS = set(range(0, 11))  # hide pose face; FaceMesh provides cleaner face

    # Pose body (no face via pose)
    pose = landmarks_pixel.get('pose')
    if pose is not None:
        for (start, end) in POSE_CONNECTIONS:
            if (start, end) in EXCLUDED_POSE_CONNECTIONS or (end, start) in EXCLUDED_POSE_CONNECTIONS:
                continue
            if start in POSE_FACE_IDXS or end in POSE_FACE_IDXS:
                continue
            if start < len(pose) and end < len(pose):
                draw_line(pose[start], pose[end], thickness=3)
        for idx, p in enumerate(pose):
            if idx in POSE_FACE_IDXS:
                continue
            draw_point(p, radius=3)

    # Hands
    for hand_key in ('left_hand', 'right_hand'):
        hand = landmarks_pixel.get(hand_key)
        if hand is not None:
            for (s, e) in HAND_CONNECTIONS:
                if s < len(hand) and e < len(hand):
                    draw_line(hand[s], hand[e], thickness=3)
            for p in hand:
                draw_point(p, radius=4)

    # Face (FaceMesh): mouth, eyes, eyebrows + dedicated upper lip
    if draw_face:
        face = landmarks_pixel.get('face')
        if face is not None and len(face) > 0:
            # Mouth outer (closed)
            mouth_outer = [face[i] if i < len(face) else None for i in MOUTH_OUTER]
            draw_closed_loop(mouth_outer, thickness=2)

            # Optional mouth inner (closed)
            if draw_inner_mouth:
                mouth_inner = [face[i] if i < len(face) else None for i in MOUTH_INNER]
                draw_closed_loop(mouth_inner, thickness=2)

            # NEW: Upper lip emphasis (open polyline)
            upper_lip = [face[i] if i < len(face) else None for i in UPPER_LIP_LOOP]
            draw_open_polyline(upper_lip, thickness=2)

            # Eyes (closed)
            left_eye = [face[i] if i < len(face) else None for i in LEFT_EYE_LOOP]
            right_eye = [face[i] if i < len(face) else None for i in RIGHT_EYE_LOOP]
            draw_closed_loop(left_eye, thickness=2)
            draw_closed_loop(right_eye, thickness=2)

            # Eyebrows (open arcs)
            left_brow = [face[i] if i < len(face) else None for i in LEFT_BROW_LOOP]
            right_brow = [face[i] if i < len(face) else None for i in RIGHT_BROW_LOOP]
            for i in range(len(left_brow) - 1):
                a, b = left_brow[i], left_brow[i + 1]
                if a and b:
                    cv2.line(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255,255,255,255), 2)
            for i in range(len(right_brow) - 1):
                a, b = right_brow[i], right_brow[i + 1]
                if a and b:
                    cv2.line(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255,255,255,255), 2)

    return canvas

# ---------------- Extraction ----------------
def extract_landmarks_from_results(results, w, h):
    out = {}
    # Pose
    out['pose'] = [(lm.x * w, lm.y * h, lm.z) for lm in results.pose_landmarks.landmark] if results.pose_landmarks else None
    # Hands
    out['left_hand']  = [(lm.x * w, lm.y * h, lm.z) for lm in results.left_hand_landmarks.landmark] if results.left_hand_landmarks else None
    out['right_hand'] = [(lm.x * w, lm.y * h, lm.z) for lm in results.right_hand_landmarks.landmark] if results.right_hand_landmarks else None
    # Face
    out['face'] = [(lm.x * w, lm.y * h, lm.z) for lm in results.face_landmarks.landmark] if results.face_landmarks else None
    return out

def process_video_file(path, output_dir, draw_skeleton=True, save_json=True, smooth=True, center_scale=True, draw_face=True):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Failed to open {path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_name = Path(path).stem
    frames_out_dir = Path(output_dir) / video_name / 'frames'
    frames_out_dir.mkdir(parents=True, exist_ok=True)
    json_out_path = Path(output_dir) / video_name / f"{video_name}_landmarks.json"

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as holistic:

        frame_idx = 0
        prev_landmarks = None
        landmarks_all_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            raw_landmarks = extract_landmarks_from_results(results, w, h)

            smoothed = smooth_landmarks(prev_landmarks, raw_landmarks, alpha=0.85) if smooth else raw_landmarks
            prev_landmarks = smoothed

            normalized = center_and_scale_landmarks(smoothed) if center_scale else None

            frame_record = {
                'frame_index': frame_idx,
                'raw': {k: ([{'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])} for p in v] if v else None)
                        for k, v in raw_landmarks.items()},
                'smoothed': {k: ([{'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])} for p in v] if v else None)
                             for k, v in smoothed.items()},
                'normalized_centered': {k: ([{'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])} for p in v] if v else None)
                                        for k, v in (normalized or {}).items()}
            }
            landmarks_all_frames.append(frame_record)

            if draw_skeleton:
                canvas = draw_skeleton_on_transparent(w, h, smoothed, draw_face=draw_face)
                out_path = frames_out_dir / f"frame_{frame_idx:06d}.png"
                success = cv2.imwrite(str(out_path), canvas)
                print(f"Saving frame {frame_idx} -> {out_path}, success={success}, nonzero={np.sum(canvas) > 0}")

            frame_idx += 1

    cap.release()

    if save_json:
        with open(json_out_path, 'w', encoding='utf-8') as f:
            json.dump({'video': video_name, 'fps': fps, 'width': w, 'height': h, 'frames': landmarks_all_frames},
                      f, ensure_ascii=False)

    print(f"Processed {path}: {frame_idx} frames. JSON -> {json_out_path}. Frames -> {frames_out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--draw_skeleton', action='store_true')
    parser.add_argument('--save_json', action='store_true')
    parser.add_argument('--smooth', action='store_true')
    parser.add_argument('--draw_face', action='store_true')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = [p for p in input_dir.iterdir() if p.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv')]
    if len(video_files) == 0:
        print('No video files found in input_dir')

    for vf in video_files:
        process_video_file(
            vf, output_dir,
            draw_skeleton=args.draw_skeleton,
            save_json=args.save_json,
            smooth=args.smooth,
            draw_face=args.draw_face
        )

    print('\nDone. To assemble transparent frames into a webm with alpha:')
    print("ffmpeg -framerate <FPS> -i frame_%06d.png -c:v libvpx-vp9 -pix_fmt yuva420p out_with_alpha.webm")
    print("Replace <FPS> with the original video's frames-per-second (check the JSON output).")
