import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import time
import argparse
import csv
import threading
import queue
import sqlite3
import numpy as np
from datetime import datetime
from collections import deque

import onnxruntime as ort

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# ---------------------------
# SQLite 資料庫初始化
# ---------------------------
def init_db(db_path="action_logs.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS action_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            duration_frames INTEGER,
            predicted_class INTEGER,
            class_name TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    return conn

def log_event_to_db(conn, duration_frames, pred_class, class_name, confidence):
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO action_events (timestamp, duration_frames, predicted_class, class_name, confidence)
        VALUES (?, ?, ?, ?, ?)
    """, (timestamp, duration_frames, pred_class, class_name, confidence))
    conn.commit()
    print(f"✅ [資料庫寫入成功] {timestamp} | 偵測到 {class_name} (信心度: {confidence:.2f})")

# ---------------------------
# Skeleton utils
# ---------------------------
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

def safe_norm(v, eps=1e-6):
    return np.sqrt(np.sum(v * v, axis=-1) + eps)

def angle_between(u, v, eps=1e-6):
    nu = safe_norm(u, eps)
    nv = safe_norm(v, eps)
    dot = np.sum(u * v, axis=-1) / (nu * nv + eps)
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)

def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def normalize_skeleton_frame(frame, eps=1e-6):
    # 最佳模型以 No-CS-Normalization 策略訓練（關閉 Center-Scale Normalization）
    # 以保留人體絕對高度特徵用於辨識 Class 0（背景非搬運動作）
    # 此函式保留以供未來比較，但部署時不呼叫
    f = frame.astype(np.float32).copy()
    xy = f[:, :2]
    has_z = f.shape[-1] >= 4
    z = f[:, 2:3] if has_z else None

    ls = xy[LEFT_SHOULDER]
    rs = xy[RIGHT_SHOULDER]
    lh = xy[LEFT_HIP]
    rh = xy[RIGHT_HIP]

    center = (ls + rs + lh + rh) / 4.0
    xy = xy - center

    shoulder_w = np.linalg.norm(ls - rs)
    hip_w = np.linalg.norm(lh - rh)
    scale = shoulder_w + hip_w
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0

    xy = xy / scale
    f[:, :2] = xy
    if has_z and z is not None:
        f[:, 2:3] = z / scale
    return f

def resample_seq(seq_TJ3, out_len):
    T = seq_TJ3.shape[0]
    if T == out_len:
        return seq_TJ3
    if T > out_len:
        idxs = np.linspace(0, T - 1, num=out_len, dtype=int)
        return seq_TJ3[idxs]
    pad_len = out_len - T
    pad = np.tile(seq_TJ3[-1:], (pad_len, 1, 1))
    return np.concatenate([seq_TJ3, pad], axis=0)

def compute_valid_ratio(frame, vis_thr=0.5):
    xy = frame[:, :2]
    vis = frame[:, -1]
    finite_xy = np.isfinite(xy).all(axis=-1)
    valid = finite_xy & (vis >= vis_thr)
    return float(valid.mean())

def compute_episode_features(seq, vis_thr=0.5):
    T = seq.shape[0]
    xy = seq[:, :, :2]
    has_z = seq.shape[-1] >= 4
    vis = seq[:, :, -1]

    finite_xy = np.isfinite(xy).all(axis=-1)
    valid_kp = finite_xy & (vis >= vis_thr)
    valid_ratio = valid_kp.mean(axis=1).astype(np.float32)

    def P(i):
        return xy[:, i, :]

    hip_c = 0.5 * (P(LEFT_HIP) + P(RIGHT_HIP))
    sh_c = 0.5 * (P(LEFT_SHOULDER) + P(RIGHT_SHOULDER))

    trunk = sh_c - hip_c
    up = np.tile(np.array([[0.0, -1.0]], dtype=np.float32), (T, 1))
    flex = angle_between(trunk, up)
    flex = np.nan_to_num(flex, nan=0.0, posinf=0.0, neginf=0.0)

    twist_deg = np.zeros(T, dtype=np.float32)
    for t in range(T):
        ls = seq[t, LEFT_SHOULDER]
        rs = seq[t, RIGHT_SHOULDER]
        lh = seq[t, LEFT_HIP]
        rh = seq[t, RIGHT_HIP]
        
        if has_z:
            sh_vec = np.array([ls[0]-rs[0], ls[2]-rs[2]], dtype=np.float32)
            hip_vec = np.array([lh[0]-rh[0], lh[2]-rh[2]], dtype=np.float32)
        else:
            sh_vec = np.array([ls[0]-rs[0], ls[1]-rs[1]], dtype=np.float32)
            hip_vec = np.array([lh[0]-rh[0], lh[1]-rh[1]], dtype=np.float32)
            
        sh_norm = np.linalg.norm(sh_vec)
        hip_norm = np.linalg.norm(hip_vec)
        if sh_norm > 1e-6 and hip_norm > 1e-6:
            cos_a = np.dot(sh_vec, hip_vec) / (sh_norm * hip_norm)
            cos_a = np.clip(cos_a, -1.0, 1.0)
            ang = np.degrees(np.arccos(cos_a))
            twist_deg[t] = min(ang, 180.0 - ang)
            
    twist_deg = np.nan_to_num(twist_deg, nan=0.0, posinf=0.0, neginf=0.0)

    w_c = 0.5 * (P(LEFT_WRIST) + P(RIGHT_WRIST))
    hand_dist = safe_norm(w_c - hip_c).astype(np.float32)
    hand_dist = np.nan_to_num(hand_dist, nan=0.0, posinf=0.0, neginf=0.0)

    def p95(x):
        return float(np.percentile(x, 95)) if len(x) > 0 else 0.0

    twist_ratio_over_20 = float((twist_deg >= 20.0).sum()) / max(1, T)
    extra_score = 3 if twist_ratio_over_20 >= 0.10 else (1 if twist_ratio_over_20 > 0 else 0)

    return {
        "flex_p95_deg": p95(flex) * 180.0 / np.pi,
        "twist_p95_deg": p95(twist_deg),
        "twist_ratio_over_20": twist_ratio_over_20,
        "extra_score": extra_score,
        "hand_dist_p95": p95(hand_dist),
        "valid_ratio_p95": p95(valid_ratio),
    }

# ---------------------------
# ONNX 推論核心
# ---------------------------
def infer_onnx(ort_session, seq_TJ3):
    """
    seq_TJ3: (T,J,3) float32
    """
    x = np.expand_dims(seq_TJ3, axis=0).astype(np.float32)
    inputs = {ort_session.get_inputs()[0].name: x}
    logits = ort_session.run(None, inputs)[0]
    
    exp_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    prob = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    pred = int(np.argmax(prob[0]))
    return pred, prob[0]

# ---------------------------
# RTSP capture helper
# ---------------------------
def open_source(source, rtsp_transport="tcp"):
    if str(source).isdigit():
        return cv2.VideoCapture(int(source))
    if source.lower().endswith(".mp4") or source.lower().endswith(".avi"):
        return cv2.VideoCapture(source)
        
    ffmpeg_opts = (
        f"rtsp_transport;{rtsp_transport}|fflags;nobuffer|flags;low_delay|stimeout;5000000"
    )
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_opts
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

class RTSPLatestFrameGrabber:
    def __init__(self, source, rtsp_transport="tcp", reconnect_fail_reads=10):
        self.source = source
        self.rtsp_transport = rtsp_transport
        self.reconnect_fail_reads = reconnect_fail_reads
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.latest_ts = 0.0
        self.last_ok_time = 0.0
        self.cap_fps = 0.0
        self.src_fps = 0.0
        self.fail_count = 0
        self.reconnect_backoff = 0.2

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        read_count = 0
        t0 = time.time()

        while not self.stop_event.is_set():
            if self.cap is None:
                cap = open_source(self.source, self.rtsp_transport)
                if cap is None:
                    time.sleep(self.reconnect_backoff)
                    self.reconnect_backoff = min(1.0, self.reconnect_backoff + 0.2)
                    continue
                with self.lock:
                    self.cap = cap
                    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
                    self.src_fps = src_fps if np.isfinite(src_fps) else 0.0
                    self.cap_fps = self.src_fps if self.src_fps > 0 else 0.0
                    self.fail_count = 0
                self.reconnect_backoff = 0.2

            ok, frame = self.cap.read()
            now = time.time()
            if not ok:
                with self.lock:
                    self.fail_count += 1
                    too_many_failures = self.fail_count >= self.reconnect_fail_reads
                if too_many_failures:
                    with self.lock:
                        if self.cap is not None:
                            self.cap.release()
                        self.cap = None
                        self.fail_count = 0
                    time.sleep(self.reconnect_backoff)
                    self.reconnect_backoff = min(1.0, self.reconnect_backoff + 0.2)
                    continue
                time.sleep(0.005)
                continue

            with self.lock:
                self.latest_ts = now
                self.last_ok_time = now
                self.fail_count = 0
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass
            except queue.Empty:
                pass
            self.reconnect_backoff = 0.2

            read_count += 1
            if read_count >= 30:
                t1 = time.time()
                dt = t1 - t0
                fps = read_count / max(1e-6, dt)
                with self.lock:
                    self.cap_fps = fps
                read_count = 0
                t0 = t1

    def get_frame(self, timeout=0.03):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_cap_fps(self):
        with self.lock:
            if self.src_fps > 0:
                return float(self.src_fps)
            return float(self.cap_fps)

    def get_source_fps(self):
        with self.lock:
            return float(self.src_fps)

    def get_stream_status(self):
        with self.lock:
            return float(self.latest_ts), float(self.last_ok_time), int(self.fail_count)

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

class VideoFileGrabber:
    """用來讀取本地影片檔，不丟幀，且播放完畢自動結束"""
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(self.src_fps) or self.src_fps <= 0:
            self.src_fps = 30.0
        self.frame_delay = 1.0 / self.src_fps
        self.last_read_time = time.time()
        self.is_finished = False

    def start(self):
        pass

    def get_frame(self, timeout=0):
        if self.is_finished: return None
        # 控制播放速度，接近真實時間
        now = time.time()
        elapsed = now - self.last_read_time
        if elapsed < self.frame_delay:
            time.sleep(self.frame_delay - elapsed)
            
        ret, frame = self.cap.read()
        self.last_read_time = time.time()
        
        if not ret:
            self.is_finished = True
            return None
        return frame

    def get_cap_fps(self):
        return float(self.src_fps)

    def get_source_fps(self):
        return float(self.src_fps)

    def get_stream_status(self):
        return time.time(), time.time(), 0

    def stop(self):
        if self.cap:
            self.cap.release()

def draw_mediapipe_pose(frame_bgr, pose_results):
    if pose_results is None or pose_results.pose_landmarks is None:
        return frame_bgr

    landmark_list = landmark_pb2.NormalizedLandmarkList(
        landmark=[
            landmark_pb2.NormalizedLandmark(
                x=lmk.x, y=lmk.y, z=lmk.z, visibility=getattr(lmk, "visibility", 0.0)
            )
            for lmk in pose_results.pose_landmarks.landmark
        ]
    )

    mp.solutions.drawing_utils.draw_landmarks(
        image=frame_bgr,
        landmark_list=landmark_list,
        connections=mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=2),
        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(thickness=2),
    )
    return frame_bgr

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="RTSP url, WebCam ID, or MP4 path")
    parser.add_argument("--onnx_path", type=str, default="../models/robust_transformer_best.onnx", help="ONNX model path")
    
    parser.add_argument("--show", type=int, default=1, help="1: show cv2 window, 0: headless")
    parser.add_argument("--live_stride", type=int, default=2, choices=[1, 2], help="Update skeleton every N frames. Use 2 to match training sample_stride=2.")
    parser.add_argument("--pose_stride", type=int, default=1, help="run pose every N processed frames")
    parser.add_argument("--mp_complexity", type=int, default=0, help="mediapipe pose model_complexity")
    parser.add_argument("--show_fps_detail", type=int, default=1, help="1: show cap/proc fps")
    parser.add_argument("--draw_pose", type=int, default=0, help="1: draw mediapipe pose overlay")
    parser.add_argument("--rtsp_transport", type=str, default="tcp", choices=["tcp", "udp"])
    parser.add_argument("--grab_timeout", type=float, default=0.005, help="frame queue get timeout (seconds)")

    parser.add_argument("--seg_len", type=int, default=16, help="binary window length (frames)")
    parser.add_argument("--seg_step", type=int, default=4, help="binary inference step (frames)")
    parser.add_argument("--episode_len", type=int, default=64, help="episode resample length for ONNX model")

    parser.add_argument("--vis_thr", type=float, default=0.5)
    parser.add_argument("--min_valid_ratio", type=float, default=0.6)
    parser.add_argument("--force_cut_vr", type=float, default=0.2, help="force cut event if window valid_ratio drops below this")

    parser.add_argument("--end_patience", type=int, default=3, help="need N consecutive low probs to end")
    parser.add_argument("--bin_ema_alpha", type=float, default=0.3, help="EMA alpha for binary probability smoothing")

    parser.add_argument("--pre_roll", type=int, default=16, help="keep some frames before event start")
    parser.add_argument("--max_event_frames", type=int, default=600, help="force cut if too long")
    parser.add_argument("--min_event_frames", type=int, default=10, help="minimum valid event frames to be considered an action")

    parser.add_argument("--save_video", type=int, default=0, help="1: save output video, 0: disable")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--out_fps", type=float, default=0.0)
    parser.add_argument("--out_codec", type=str, default="mp4v")
    parser.add_argument("--out_prefix", type=str, default="cam1_live")
    parser.add_argument("--save_csv", type=int, default=0)
    parser.add_argument("--csv_dir", type=str, default="outputs")
    parser.add_argument("--csv_prefix", type=str, default="window_stats")

    parser.add_argument("--in_event_thr", type=float, default=0.5)
    parser.add_argument("--out_event_thr", type=float, default=0.5)

    args = parser.parse_args()

    # 1. 啟動 SQLite 資料庫
    db_conn = init_db()

    # 2. 載入 ONNX 引擎
    print(f"🚀 正在載入 ONNX 模型: {args.onnx_path}")
    ort_session = ort.InferenceSession(args.onnx_path, providers=['CPUExecutionProvider'])

    class_names = {
        0: "背景／非搬運 (Normal)",
        1: "桌面高度蹲舉 (Squat)",
        2: "髖鉸鏈搬運 (Hip-Hinge)",
        3: "屈膝非對稱搬運 (Asymmetric)",
        4: "⚠️ 直膝彎腰搬運 (Upright/High Risk)",
    }

    # 3. 啟動 MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=args.mp_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    # 4. 判斷輸入來源是影片檔還是 RTSP/WebCam
    is_file = args.source.lower().endswith((".mp4", ".avi", ".mov"))
    if is_file:
        print("[INFO] 偵測到輸入為本地影片檔，採用 VideoFileGrabber (不丟幀模式)")
        grabber = VideoFileGrabber(args.source)
    else:
        print("[INFO] 偵測到輸入為串流或攝影機，採用 RTSPLatestFrameGrabber (低延遲丟幀模式)")
        grabber = RTSPLatestFrameGrabber(args.source, rtsp_transport=args.rtsp_transport, reconnect_fail_reads=30)
        
    grabber.start()
    print("[INFO] proc_fps may exceed camera fps briefly at startup due to buffered frames.")

    video_writer = None
    video_path = None
    frame_size = None
    video_enabled = int(args.save_video) == 1
    video_codec = args.out_codec
    if video_enabled and len(video_codec) != 4:
        print(f"[WARN] invalid out_codec='{video_codec}', fallback to 'mp4v'")
        video_codec = "mp4v"

    csv_file = None
    csv_writer = None
    csv_enabled = int(args.save_csv) == 1
    if csv_enabled:
        try:
            os.makedirs(args.csv_dir, exist_ok=True)
            csv_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(args.csv_dir, f"{args.csv_prefix}_{csv_ts}.csv")
            csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "timestamp", "window_start_idx", "window_end_idx", "valid_ratio", 
                "motion_p_raw", "motion_p_smooth", "in_event", "pred_class",
                "pred_prob", "current_twist_deg", "event_extra_score"
            ])
        except Exception as e:
            print(f"[WARN] cannot create csv: {e}")
            csv_enabled = False

    frame_idx = 0
    skel_buf = deque(maxlen=max(args.seg_len, args.pre_roll, 64))
    valid_buf = deque(maxlen=64)

    in_event = False
    event_frames = []
    event_start_idx = None
    end_counter = 0
    event_active_class = 0
    event_active_name = ""

    last_prob_motion = 0.0
    last_prob_motion_smooth = 0.0
    last_class_pred = 0
    last_valid_ratio = 0.0
    last_pose_results = None
    last_skel_raw = None
    last_extra_score = 0
    window_start_idx = 0

    last_start_result_text = ""
    last_5_results = []

    t0 = time.time()
    proc_fps = 0.0
    proc_count = 0
    pose_count = 0
    pose_tick = 0
    last_good_frame = None
    queue_empty_since = None

    while True:
        frame = grabber.get_frame(timeout=float(args.grab_timeout))
        latest_ts, last_ok_time, fail_count = grabber.get_stream_status()
        now = time.time()
        age_ms = (now - latest_ts) * 1000.0 if latest_ts > 0 else -1.0
        rtsp_ok_gap_s = (now - last_ok_time) if last_ok_time > 0 else -1.0

        if frame is None:
            if is_file and hasattr(grabber, 'is_finished') and grabber.is_finished:
                print("\n[INFO] 影片播放完畢，自動退出！")
                break
                
            if queue_empty_since is None:
                queue_empty_since = now
            if args.show == 1:
                if last_good_frame is not None:
                    frame = last_good_frame.copy()
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                if now - queue_empty_since > 0.5:
                    cv2.putText(frame, "Waiting for frame...", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                debug_line = f"age_ms={age_ms:.1f}  ok_gap_s={rtsp_ok_gap_s:.2f}"
                cv2.putText(frame, debug_line, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)
        else:
            queue_empty_since = None
            
            # ★ 強制鏡像翻轉：迎合訓練集收集時的鏡像宇宙，解決 Domain Shift 錯位問題
            frame = cv2.flip(frame, 1)
            
            last_good_frame = frame.copy()
            frame_idx += 1

            proc_count += 1
            if proc_count >= 10:
                t1 = time.time()
                dt = t1 - t0
                proc_fps = proc_count / max(1e-6, dt)
                proc_count = 0
                pose_count = 0
                t0 = t1

            if frame_idx % args.live_stride != 0:
                if args.show == 1:
                    line1 = f"FPS={proc_fps:.1f}  valid={last_valid_ratio:.2f}  motion_p={last_prob_motion:.2f}  in_event={int(in_event)}"
                    cv2.putText(frame, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if in_event and event_start_idx is not None:
                        cv2.putText(frame, f"EVENT running... frames={len(event_frames)}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                    if last_start_result_text:
                        cv2.putText(frame, last_start_result_text, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                    y_offset = 114
                    for past_result in last_5_results:
                        cv2.putText(frame, past_result, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 255), 2)
                        y_offset += 25
            else:
                if video_enabled and video_writer is None:
                    h0, w0 = frame.shape[:2]
                    frame_size = (w0, h0)
                    os.makedirs(args.out_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = os.path.join(args.out_dir, f"{args.out_prefix}_{ts}.mp4")
                    out_fps = args.out_fps if args.out_fps > 0 else 20.0
                    fourcc = cv2.VideoWriter_fourcc(*video_codec)
                    writer = cv2.VideoWriter(video_path, fourcc, out_fps, frame_size)
                    if writer.isOpened(): video_writer = writer

                pose_tick += 1
                run_pose = (pose_tick % max(1, int(args.pose_stride)) == 0) or (last_skel_raw is None)
                if run_pose:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = pose.process(img_rgb)
                    last_pose_results = res
                    pose_count += 1

                    skel = np.zeros((33, 4), dtype=np.float32)
                    if res.pose_landmarks is not None:
                        lm = res.pose_landmarks.landmark
                        for i in range(33):
                            skel[i, 0] = lm[i].x
                            skel[i, 1] = lm[i].y
                            skel[i, 2] = lm[i].z
                            skel[i, 3] = getattr(lm[i], "visibility", 0.0)
                        last_skel_raw = skel.copy()
                    else:
                        last_skel_raw = None
                else:
                    res = last_pose_results
                    skel = None if last_skel_raw is None else last_skel_raw.copy()

                if skel is not None:
                    skel = np.nan_to_num(skel, nan=0.0, posinf=0.0, neginf=0.0)
                    # No-Norm：不做 Center-Scale Normalization，保留絕對高度特徵
                    skel_n = skel.astype(np.float32)
                    vr = compute_valid_ratio(skel_n, vis_thr=args.vis_thr)
                    last_valid_ratio = vr

                    skel_buf.append(skel_n)
                    valid_buf.append(vr)

                    if len(skel_buf) >= args.seg_len and (frame_idx % args.seg_step == 0):
                        seg = np.stack(list(skel_buf)[-args.seg_len:], axis=0).astype(np.float32)
                        window_valid_ratio = float(np.mean(list(valid_buf)[-min(len(valid_buf), args.seg_len):]))

                        CORE_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
                        seg_13 = seg[:, CORE_JOINTS, :]
                        seg_13_model = seg_13[:, :, [0, 1, 3]]
                        
                        # 強制補齊到 64 幀送給 ONNX
                        seg_13_model = resample_seq(seg_13_model, 64)
                        pred, prob = infer_onnx(ort_session, seg_13_model)
                        
                        is_low_visibility = window_valid_ratio < 0.35
                        v_list = list(valid_buf)[-min(len(valid_buf), args.seg_len):]
                        has_consecutive_drops = False
                        if len(v_list) >= 2:
                            for i in range(len(v_list) - 1):
                                if v_list[i] < 0.1 and v_list[i+1] < 0.1:
                                    has_consecutive_drops = True
                                    break
                                    
                        if is_low_visibility or has_consecutive_drops:
                            prob = np.zeros_like(prob)
                            prob[0] = 1.0
                            pred = 0

                        last_class_pred = pred
                        p_motion_raw = 1.0 - float(prob[0])
                        last_prob_motion = p_motion_raw
                        
                        scale = np.clip(window_valid_ratio / max(1e-6, float(args.min_valid_ratio)), 0.0, 1.0)
                        alpha = float(args.bin_ema_alpha)
                        last_prob_motion_smooth = alpha * p_motion_raw + (1.0 - alpha) * last_prob_motion_smooth
                        current_p_decide = last_prob_motion_smooth * float(scale) if not in_event else last_prob_motion_smooth
                        window_start_idx = max(1, frame_idx - (args.seg_len - 1) * args.live_stride)

                        if not in_event:
                            if current_p_decide >= args.in_event_thr:
                                in_event = True
                                end_counter = 0
                                event_start_idx = window_start_idx
                                pre = list(skel_buf)[-min(len(skel_buf), args.pre_roll):]
                                event_frames = [x.copy() for x in pre]
                                event_active_class = last_class_pred 
                                event_active_name = class_names.get(event_active_class, str(event_active_class))
                                last_start_result_text = f"Event Started -> Triggered by {event_active_name}..."
                        else:
                            event_frames.append(skel_n.copy())
                            if current_p_decide <= args.out_event_thr:
                                end_counter += 1
                            else:
                                end_counter = 0

                            too_long = len(event_frames) >= args.max_event_frames
                            vr_too_low = window_valid_ratio < args.force_cut_vr

                            if end_counter >= args.end_patience or too_long or vr_too_low:
                                in_event = False
                                end_counter = 0
                                if vr_too_low and len(event_frames) > 5:
                                    event_frames = event_frames[:-3]

                                if len(event_frames) < args.min_event_frames:
                                    event_frames = []
                                    event_start_idx = None
                                    continue

                                ep = np.stack(event_frames, axis=0).astype(np.float32)
                                ep_rs = resample_seq(ep, 64)

                                CORE_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
                                ep_rs_13 = ep_rs[:, CORE_JOINTS, :]
                                ep_rs_13_model = ep_rs_13[:, :, [0, 1, 3]]
                                
                                c_pred, c_prob = infer_onnx(ort_session, ep_rs_13_model)
                                c_name = class_names.get(c_pred, str(c_pred))
                                
                                # ★ 寫入資料庫
                                if c_pred != 0:
                                    log_event_to_db(db_conn, len(event_frames), c_pred, c_name, float(c_prob[c_pred]))

                                last_start_result_text = f"Finished. Duration: {len(event_frames)} frames"
                                new_result_text = f"[{datetime.now().strftime('%H:%M:%S')}] Class: {c_pred} ({c_name}) (Prob={c_prob[c_pred]:.2f})"
                                last_5_results.insert(0, new_result_text)
                                if len(last_5_results) > 5: last_5_results.pop()

                                event_frames = []
                                event_start_idx = None

                    if args.show == 1 or video_writer is not None:
                        if args.draw_pose == 1:
                            frame = draw_mediapipe_pose(frame, res)
                        h, w = frame.shape[:2]
                        line1 = f"FPS={proc_fps:.1f}  valid={vr:.2f}  motion_p={last_prob_motion_smooth:.2f}  in_event={int(in_event)}"
                        cv2.putText(frame, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        if in_event and event_start_idx is not None:
                            cv2.putText(frame, f"EVENT running... frames={len(event_frames)}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                        if last_start_result_text:
                            cv2.putText(frame, last_start_result_text, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                        y_offset = 114
                        for past_result in last_5_results:
                            cv2.putText(frame, past_result, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)
                            y_offset += 25
                        if window_start_idx > 0:
                            window_info = f"Frame: {frame_idx} (Window: {window_start_idx}-{frame_idx})"
                        else:
                            window_info = f"Frame: {frame_idx} (Initializing...)"
                        cv2.putText(frame, window_info, (w - 450, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        if video_writer is not None:
                            video_writer.write(frame)

        if args.show == 1:
            if frame is not None:
                cv2.imshow("cam1_live_advanced", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            time.sleep(0.001)

    grabber.stop()
    if video_writer is not None: video_writer.release()
    if csv_file is not None: csv_file.close()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()
