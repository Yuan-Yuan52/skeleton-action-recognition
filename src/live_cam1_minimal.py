import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import time
import argparse
import csv
import threading
import queue
import numpy as np
from datetime import datetime
from collections import deque

import torch
import torch.nn.functional as F

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from models_transformer import SpatialTemporalTransformer

# ---------------------------
# Skeleton utils
# ---------------------------
# Mediapipe Pose 33 indices (same as you used)
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


def normalize_skeleton_frame(frame33x3, eps=1e-6):
    """
    frame33x3: (33,3) with x,y in [0,1] (mediapipe normalized), vis in [0,1]
    Do per-frame center/scale normalization (same spirit as your training).
    """
    f = frame33x3.astype(np.float32).copy()
    xy = f[:, :2]

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
    return f


def resample_seq(seq_TJ3, out_len):
    """
    seq_TJ3: (T,J,3)
    """
    T = seq_TJ3.shape[0]
    if T == out_len:
        return seq_TJ3
    if T > out_len:
        idxs = np.linspace(0, T - 1, num=out_len, dtype=int)
        return seq_TJ3[idxs]
    # pad
    pad_len = out_len - T
    pad = np.tile(seq_TJ3[-1:], (pad_len, 1, 1))
    return np.concatenate([seq_TJ3, pad], axis=0)


def compute_valid_ratio(frame33x3, vis_thr=0.5):
    xy = frame33x3[:, :2]
    vis = frame33x3[:, 2]
    finite_xy = np.isfinite(xy).all(axis=-1)
    valid = finite_xy & (vis >= vis_thr)
    return float(valid.mean())


def compute_episode_features(seq_TJ3, vis_thr=0.5):
    """
    Compute per-frame flex/twist/hand_dist/valid_ratio, then aggregate p95.
    seq_TJ3 is assumed already normalized (center+scale).
    Returns dict with p95 stats.
    """
    T = seq_TJ3.shape[0]
    xy = seq_TJ3[:, :, :2]
    vis = seq_TJ3[:, :, 2]

    finite_xy = np.isfinite(xy).all(axis=-1)
    valid_kp = finite_xy & (vis >= vis_thr)
    valid_ratio = valid_kp.mean(axis=1).astype(np.float32)

    def P(i):
        return xy[:, i, :]

    hip_c = 0.5 * (P(LEFT_HIP) + P(RIGHT_HIP))
    sh_c = 0.5 * (P(LEFT_SHOULDER) + P(RIGHT_SHOULDER))

    # flex
    trunk = sh_c - hip_c
    up = np.tile(np.array([[0.0, -1.0]], dtype=np.float32), (T, 1))
    flex = angle_between(trunk, up)
    flex = np.nan_to_num(flex, nan=0.0, posinf=0.0, neginf=0.0)  # radians

    # twist (2D proxy)
    sh_vec = P(RIGHT_SHOULDER) - P(LEFT_SHOULDER)
    hip_vec = P(RIGHT_HIP) - P(LEFT_HIP)
    sh_ang = np.arctan2(sh_vec[:, 1], sh_vec[:, 0])
    hip_ang = np.arctan2(hip_vec[:, 1], hip_vec[:, 0])
    twist = np.abs(wrap_pi(sh_ang - hip_ang))
    twist = np.nan_to_num(twist, nan=0.0, posinf=0.0, neginf=0.0)  # radians

    # hand_dist (wrist center to hip center)
    w_c = 0.5 * (P(LEFT_WRIST) + P(RIGHT_WRIST))
    hand_dist = safe_norm(w_c - hip_c).astype(np.float32)
    hand_dist = np.nan_to_num(hand_dist, nan=0.0, posinf=0.0, neginf=0.0)

    def p95(x):
        if len(x) == 0:
            return 0.0
        return float(np.percentile(x, 95))

    return {
        "flex_p95_deg": p95(flex) * 180.0 / np.pi,
        "twist_p95_deg": p95(twist) * 180.0 / np.pi,
        "hand_dist_p95": p95(hand_dist),
        "valid_ratio_p95": p95(valid_ratio),
    }


# ---------------------------
# Model utils
# ---------------------------
# ★ 修改點 1: 增加 in_channels 參數，用於支援進階模型的 6 個通道 (包含速度特徵)
def load_gru_ckpt(ckpt_path, num_joints, num_classes, device, in_channels=3):
    model = SpatialTemporalTransformer(
        num_joints=num_joints, 
        in_channels=in_channels, 
        d_model=256, 
        nhead=8, 
        num_spatial_layers=3, 
        num_temporal_layers=4,
        num_classes=num_classes, 
        dropout=0.3
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
# ★ 修改點 2: 增加 use_velocity 參數。開啟後，神經網路會自己幫您計算速度差值(Velocity)
def infer_class(model, seq_TJ3, device, use_velocity=False):
    """
    seq_TJ3: (T,J,3) float32
    """
    x = torch.from_numpy(seq_TJ3).unsqueeze(0).to(device)  # (1,T,J,3)
    
    # 🌟 動態計算速度並合併至輸入特徵中
    if use_velocity:
        velocity = torch.zeros_like(x)
        # v[t] = x[t] - x[t-1]
        velocity[:, 1:, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
        # 串接後變成 (1, T, 33, 6)
        x = torch.cat([x, velocity], dim=-1)
        
    logits = model(x)  # (1,C)
    prob = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred = int(prob.argmax())
    return pred, prob


# ---------------------------
# RTSP capture helper
# ---------------------------
def open_rtsp(rtsp_url, rtsp_transport="tcp"):
    ffmpeg_opts = (
        f"rtsp_transport;{rtsp_transport}|fflags;nobuffer|flags;low_delay|stimeout;5000000"
    )
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_opts
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class RTSPLatestFrameGrabber:
    """
    Background RTSP reader that always keeps only the newest frame.
    """
    def __init__(self, rtsp_url, rtsp_transport="tcp", reconnect_fail_reads=10):
        self.rtsp_url = rtsp_url
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
                cap = open_rtsp(self.rtsp_url, self.rtsp_transport)
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

def draw_mediapipe_pose(frame_bgr, pose_results):
    """
    frame_bgr: OpenCV BGR image
    pose_results: output of mp.solutions.pose.Pose.process()
    """
    if pose_results is None or pose_results.pose_landmarks is None:
        return frame_bgr

    # MediaPipe drawing wants a NormalizedLandmarkList proto
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
    parser.add_argument("--rtsp", type=str, required=True)

    # ★ 修改點 3: 簡化 checkpoints 參數，現在只要提供 binary(二元模型) 以及 class4(進階五分類)
    # ★ 修改點 3: 現在我們只有一個模型 (5分類)
    parser.add_argument("--class4_ckpt", type=str, required=True, help="Advanced 5-class model checkpoint path")

    parser.add_argument("--show", type=int, default=1, help="1: show cv2 window, 0: headless")
    parser.add_argument("--live_stride", type=int, default=2, choices=[1, 2],
                        help="Update skeleton every N frames. Use 2 to match training sample_stride=2.")
    parser.add_argument("--pose_stride", type=int, default=1, help="run pose every N processed frames")
    parser.add_argument("--mp_complexity", type=int, default=0, help="mediapipe pose model_complexity")
    parser.add_argument("--show_fps_detail", type=int, default=1, help="1: show cap/proc fps")
    parser.add_argument("--draw_pose", type=int, default=0, help="1: draw mediapipe pose overlay")
    parser.add_argument("--rtsp_transport", type=str, default="tcp", choices=["tcp", "udp"],
                        help="RTSP transport protocol for OpenCV FFmpeg capture")
    parser.add_argument("--grab_timeout", type=float, default=0.005, help="frame queue get timeout (seconds)")

    # window lengths
    parser.add_argument("--seg_len", type=int, default=16, help="binary window length (frames)")
    parser.add_argument("--seg_step", type=int, default=4, help="binary inference step (frames)")
    # ★ 修改點 4: default 改為 24 幀，對齊 advanced 模型的訓練條件
    parser.add_argument("--episode_len", type=int, default=24, help="episode resample length for class4 / advanced model")

    # thresholds
    parser.add_argument("--vis_thr", type=float, default=0.5)
    parser.add_argument("--min_valid_ratio", type=float, default=0.6)
    parser.add_argument("--force_cut_vr", type=float, default=0.2, help="force cut event if window valid_ratio drops below this")

    parser.add_argument("--end_patience", type=int, default=3, help="need N consecutive low probs to end")
    parser.add_argument("--bin_ema_alpha", type=float, default=0.3, help="EMA alpha for binary probability smoothing")

    # pre-roll and max duration
    parser.add_argument("--pre_roll", type=int, default=16, help="keep some frames before event start")
    parser.add_argument("--max_event_frames", type=int, default=600, help="force cut if too long")

    # optional video output
    parser.add_argument("--save_video", type=int, default=0, help="1: save output video, 0: disable")
    parser.add_argument("--out_dir", type=str, default="outputs", help="output directory for saved videos")
    parser.add_argument("--out_fps", type=float, default=0.0, help="0: use capture fps or fallback to 20")
    parser.add_argument("--out_codec", type=str, default="mp4v", help="fourcc codec, e.g. mp4v")
    parser.add_argument("--out_prefix", type=str, default="cam1_live", help="output filename prefix")
    parser.add_argument("--save_csv", type=int, default=0, help="1: save window stats csv, 0: disable")
    parser.add_argument("--csv_dir", type=str, default="outputs", help="output directory for csv logs")
    parser.add_argument("--csv_prefix", type=str, default="window_stats", help="csv filename prefix")

    # threshold tuning for class 0 (no motion) vs others (motion)
    parser.add_argument("--in_event_thr", type=float, default=0.5, help="cumulative prob(class 1~4) >= thr => event start")
    parser.add_argument("--out_event_thr", type=float, default=0.5, help="prob(class 0) >= thr => event end")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> device:", device)

    # ★ 修改點 4: 只需要載入這一個進階大腦，並限定輸入維度為 13 個核心關節
    class4_model = load_gru_ckpt(args.class4_ckpt, num_joints=13, num_classes=5, device=device, in_channels=3)

    # 定義類別名稱 (0 就是無運動 / 標準動作，1~4是對應的姿勢)
    class_names = {
        0: "Class 0 (Normal/Idle)",
        1: "Class 1 (Error A)",
        2: "Class 2 (Error B)",
        3: "Class 3 (Error C)",
        4: "Class 4 (Error D)",
    }

    # mediapipe init
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=args.mp_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    grabber = RTSPLatestFrameGrabber(args.rtsp, rtsp_transport=args.rtsp_transport, reconnect_fail_reads=30)
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
                "timestamp", 
                "frame_idx", 
                "valid_ratio", 
                "motion_p_raw", 
                "motion_p_smooth", 
                "in_event",
                "pred_class",
                "pred_prob"
            ])
        except Exception as e:
            print(f"[WARN] cannot create csv log file: {e}. disable csv output.")
            csv_enabled = False
            csv_file = None
            csv_writer = None

    # buffers
    frame_idx = 0
    skel_buf = deque(maxlen=max(args.seg_len, args.pre_roll, 64))  # normalized skeleton frames
    valid_buf = deque(maxlen=64)

    # event state machine
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

    last_start_result_text = ""
    last_5_results = []
    last_feat = None

    # fps tracking
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
            if queue_empty_since is None:
                queue_empty_since = now

            if args.show == 1:
                if last_good_frame is not None:
                    frame = last_good_frame.copy()
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                if now - queue_empty_since > 0.5:
                    cv2.putText(frame, "Waiting for RTSP frame...", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                debug_line = f"age_ms={age_ms:.1f}  rtsp_ok_gap_s={rtsp_ok_gap_s:.2f}"
                cv2.putText(frame, debug_line, (10, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)
                if fail_count > 0:
                    warn_line = f"reconnecting... read_fail_count={fail_count}"
                    cv2.putText(frame, warn_line, (10, 84),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
        else:
            queue_empty_since = None
            last_good_frame = frame.copy()

            frame_idx += 1

            # fps calc 
            proc_count += 1
            if proc_count >= 10:
                t1 = time.time()
                dt = t1 - t0
                proc_fps = proc_count / max(1e-6, dt)
                proc_count = 0
                pose_count = 0
                t0 = t1

            # Skip heavy processing on non-stride frames; keep UI responsive.
            if frame_idx % args.live_stride != 0:
                if args.show == 1:
                    line1 = f"FPS={proc_fps:.1f}  valid={last_valid_ratio:.2f}  motion_p={last_prob_motion:.2f}  in_event={int(in_event)}"
                    cv2.putText(frame, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if int(args.show_fps_detail) == 1:
                        cap_fps_now = grabber.get_cap_fps()
                        fps_line = f"cap_fps={cap_fps_now:.1f}  proc_fps={proc_fps:.1f}"
                        cv2.putText(frame, fps_line, (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    debug_line = f"age_ms={age_ms:.1f}  rtsp_ok_gap_s={rtsp_ok_gap_s:.2f}"
                    cv2.putText(frame, debug_line, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)

                    if in_event and event_start_idx is not None:
                        dur = len(event_frames)
                        line2 = f"EVENT running... frames={dur}"
                        cv2.putText(frame, line2, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                    if last_start_result_text:
                        cv2.putText(frame, last_start_result_text, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                    
                    y_offset = 114
                    for past_result in last_5_results:
                        cv2.putText(frame, past_result, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 255), 2)
                        y_offset += 25
            else:
                # lazy init video writer
                if video_enabled and video_writer is None:
                    h0, w0 = frame.shape[:2]
                    frame_size = (w0, h0)
                    try:
                        os.makedirs(args.out_dir, exist_ok=True)
                    except Exception as e:
                        print(f"[WARN] cannot create out_dir '{args.out_dir}': {e}. disable video output.")
                        video_enabled = False

                    if video_enabled:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        video_path = os.path.join(args.out_dir, f"{args.out_prefix}_{ts}.mp4")

                        if args.out_fps > 0:
                            out_fps = float(args.out_fps)
                        else:
                            cap_fps = grabber.get_cap_fps()
                            src_fps = grabber.get_source_fps()
                            if np.isfinite(cap_fps) and cap_fps > 1.0:
                                out_fps = cap_fps
                            elif np.isfinite(src_fps) and src_fps > 1.0:
                                out_fps = src_fps
                            else:
                                out_fps = 20.0
                                
                            # ★ 修正點：配合 live_stride 的丟幀機制，調降輸出 FPS
                            out_fps = out_fps / args.live_stride

                        fourcc = cv2.VideoWriter_fourcc(*video_codec)
                        writer = cv2.VideoWriter(video_path, fourcc, out_fps, frame_size)
                        if not writer.isOpened():
                            video_enabled = False
                        else:
                            video_writer = writer

                pose_tick += 1
                run_pose = (pose_tick % max(1, int(args.pose_stride)) == 0) or (last_skel_raw is None)
                if run_pose:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = pose.process(img_rgb)
                    last_pose_results = res
                    pose_count += 1

                    skel = np.zeros((33, 3), dtype=np.float32)
                    if res.pose_landmarks is not None:
                        lm = res.pose_landmarks.landmark
                        for i in range(33):
                            skel[i, 0] = lm[i].x
                            skel[i, 1] = lm[i].y
                            skel[i, 2] = lm[i].visibility
                        last_skel_raw = skel.copy()
                    else:
                        last_skel_raw = None
                else:
                    res = last_pose_results
                    skel = None if last_skel_raw is None else last_skel_raw.copy()

                if skel is None:
                    pass
                else:
                    # normalize per frame
                    skel = np.nan_to_num(skel, nan=0.0, posinf=0.0, neginf=0.0)
                    skel_n = normalize_skeleton_frame(skel)
                    vr = compute_valid_ratio(skel_n, vis_thr=args.vis_thr)
                    last_valid_ratio = vr

                    skel_buf.append(skel_n)
                    valid_buf.append(vr)

                    # infer motion every seg_step frames
                    if len(skel_buf) >= args.seg_len and (frame_idx % args.seg_step == 0):
                        seg = np.stack(list(skel_buf)[-args.seg_len:], axis=0).astype(np.float32)  # (T,33,3)
                        window_valid_ratio = float(np.mean(list(valid_buf)[-min(len(valid_buf), args.seg_len):]))

                        # ★ 擷取 13 個核心關節 (鼻子 + 肩膀/手肘/手腕/臀部/膝蓋/腳踝) 給模型看
                        CORE_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
                        seg_13 = seg[:, CORE_JOINTS, :]

                        # 直接用 Advanced Class4 型預測
                        pred, prob = infer_class(class4_model, seg_13, device, use_velocity=False)
                        last_class_pred = pred
                        
                        # 定義「運動機率」為非 Class 0 的機率總和，即 1.0 - prob[0]
                        p_motion_raw = 1.0 - float(prob[0])
                        last_prob_motion = p_motion_raw
                        
                        # ---------------------------------------------------------
                        # ★ 修正點：Valid Ratio 帶來的降載效應
                        # 如果有人走出去，骨架模糊，那它應該是要讓 p_motion 變低 (不想觸發 Start)
                        # 但"已經在動作中"時，p_motion_scaled 變低會導致直接強制結束 (End)
                        # 所以我們將 Scale 的判斷跟 Smoothing 分開。
                        # ---------------------------------------------------------
                        scale = np.clip(window_valid_ratio / max(1e-6, float(args.min_valid_ratio)), 0.0, 1.0)
                        
                        # EMA smoothing 放平滑最原始的機率
                        alpha = float(args.bin_ema_alpha)
                        alpha = min(1.0, max(0.0, alpha))
                        last_prob_motion_smooth = alpha * p_motion_raw + (1.0 - alpha) * last_prob_motion_smooth
                        
                        # 如果"還沒開始"，我們乘上 scale 讓他不要亂觸發。
                        # 如果"已經開始了"，我們就不乘 scale，避免只因為骨架模糊兩幀就被強制卡掉動作。
                        current_p_decide = last_prob_motion_smooth * float(scale) if not in_event else last_prob_motion_smooth
                        
                        if csv_enabled and csv_writer is not None:
                            csv_writer.writerow([
                                datetime.now().isoformat(timespec="milliseconds"),
                                frame_idx,
                                f"{window_valid_ratio:.6f}",
                                f"{p_motion_raw:.6f}",
                                f"{current_p_decide:.6f}",
                                int(in_event),
                                last_class_pred,
                                f"{prob[last_class_pred]:.6f}"
                            ])

                        # ★ 狀態機更新
                        if not in_event:
                            # 判斷開始動作：motion 機率大於閾值
                            if current_p_decide >= args.in_event_thr:
                                in_event = True
                                end_counter = 0
                                event_start_idx = frame_idx

                                pre = list(skel_buf)[-min(len(skel_buf), args.pre_roll):]
                                event_frames = [x.copy() for x in pre]
                                
                                # 改成忠實記錄當下那個觸發開始畫面的 Pred 類別
                                event_active_class = last_class_pred 
                                event_active_name = class_names.get(event_active_class, str(event_active_class))
                                last_start_result_text = f"Event Started -> Triggered by {event_active_name}..."
                                
                        else:
                            # 動作進行中
                            event_frames.append(skel_n.copy())

                            # 判斷結束：motion 機率太低（代表回到了 Class 0 狀態）
                            if current_p_decide <= args.out_event_thr:
                                end_counter += 1
                            else:
                                end_counter = 0

                            too_long = len(event_frames) >= args.max_event_frames
                            vr_too_low = window_valid_ratio < args.force_cut_vr

                            # ★ 動作結束：整段送去鑑定，這部分邏輯不變
                            if end_counter >= args.end_patience or too_long or vr_too_low:
                                in_event = False
                                end_counter = 0

                                # 如果是因為丟失骨架強制切斷 (vr_too_low)，我們應該把最後幾幀糊掉的從 event_frames 裡剔除
                                if vr_too_low and len(event_frames) > 5:
                                    event_frames = event_frames[:-3]

                                # 將收集到的整段影像重採樣至 24 幀
                                ep = np.stack(event_frames, axis=0).astype(np.float32)  # (T,33,3)
                                ep_rs = resample_seq(ep, args.episode_len)

                                # 再次用模型進行最終仔細判定 (擷取 13 核心關節)
                                CORE_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
                                ep_rs_13 = ep_rs[:, CORE_JOINTS, :]
                                c_pred, c_prob = infer_class(class4_model, ep_rs_13, device, use_velocity=False)
                                c_name = class_names.get(c_pred, str(c_pred))

                                feat = compute_episode_features(ep, vis_thr=args.vis_thr)
                                last_feat = feat

                                # ★ 修正點：更新畫面上顯示的最終判定字串 (保留歷史前 5 次)
                                # 我們採用「方法 1：信任即時播報」，把動作發生當下最強烈的判定記錄下來
                                last_start_result_text = f"Finished. Duration: {len(event_frames)} frames"
                                
                                # ★ 修正點：完全信任這段 64 幀長度 (episode) 的最終綜合鑑定結果 c_pred
                                # 不要再用一開始觸發時的短暫預測 (event_active_class)，因為那通常不準
                                final_val = c_pred
                                final_name = class_names.get(c_pred, str(c_pred))
                                
                                new_result_text = (
                                    f"[{datetime.now().strftime('%H:%M:%S')}] Class: {final_val} ({final_name}) (Prob={c_prob[c_pred]:.2f})"
                                )
                                last_5_results.insert(0, new_result_text)
                                if len(last_5_results) > 5:
                                    last_5_results.pop()

                                # 清空狀態
                                event_frames = []
                                event_start_idx = None

                    should_render = (args.show == 1) or (video_writer is not None)
                    if should_render:
                        if args.draw_pose == 1:
                            frame = draw_mediapipe_pose(frame, res)

                        h, w = frame.shape[:2]
                        line1 = f"FPS={proc_fps:.1f}  valid={vr:.2f}  motion_p={last_prob_motion_smooth:.2f}  in_event={int(in_event)}"
                        cv2.putText(frame, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        if in_event and event_start_idx is not None:
                            dur = len(event_frames)
                            line2 = f"EVENT running... frames={dur}"
                            cv2.putText(frame, line2, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                        if last_start_result_text:
                            cv2.putText(frame, last_start_result_text, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                        
                        y_offset = 114
                        for past_result in last_5_results:
                            cv2.putText(frame, past_result, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)
                            y_offset += 25
                            
                        # 推遲其他除錯資訊往下移，避免蓋到歷史清單
                        info_y_offset = max(114 + len(last_5_results)*25 + 15, y_offset)
                        if int(args.show_fps_detail) == 1:
                            cap_fps_now = grabber.get_cap_fps()
                            fps_line = f"cap_fps={cap_fps_now:.1f}  proc_fps={proc_fps:.1f}"
                            cv2.putText(frame, fps_line, (10, info_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                            info_y_offset += 30
                        debug_line = f"age_ms={age_ms:.1f}  rtsp_ok_gap_s={rtsp_ok_gap_s:.2f}"
                        cv2.putText(frame, debug_line, (10, info_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)

                        if video_writer is not None:
                            video_writer.write(frame)

        if args.show == 1:
            if frame is not None:
                cv2.imshow("cam1_live_minimal", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            time.sleep(0.001)

    grabber.stop()
    if video_writer is not None:
        video_writer.release()
        print(f"[INFO] video saved: {video_path}")
    if csv_file is not None:
        csv_file.close()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()
