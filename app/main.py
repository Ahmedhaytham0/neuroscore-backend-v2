from __future__ import annotations

import os
import gc
import threading
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from scipy.fft import fft, fftfreq

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

FINGER_MODEL_PATH = MODEL_DIR / "finger_to_nose_random_forest.joblib"

ROMBERG_MODEL_PATH = MODEL_DIR / "romberg_model.joblib"

TANDEM_MODEL_PATH = MODEL_DIR / "tandem_back_ensemble.joblib"

FINGER_FEATURES = [
    "ftn_dist_mean",
    "ftn_dist_std",
    "ftn_dist_min",
    "ftn_dist_max",
    "ftn_dist_range",
    "ftn_jitter",
    "endpoint_accuracy",
    "miss_rate",
    "path_length",
    "path_efficiency",
    "sway_x",
    "sway_y",
    "vel_mean",
    "vel_std",
    "vel_max",
    "acc_mean",
    "acc_std",
    "tremor_freq",
    "tremor_power",
    "tremor_ratio",
    "signal_entropy",
    "elbow_angle_mean",
    "elbow_angle_std",
    "elbow_angle_range",
    "body_sway_x",
    "body_sway_y",
    "movement_irregularity",
    "jerk_mean",
]

ROMBERG_FEATURES = [
    "sh_tilt_mean",
    "sh_tilt_std",
    "sh_tilt_max",
    "sh_tilt_range",
    "sh_tilt_p75",
    "sh_tilt_p90",
    "hip_tilt_mean",
    "hip_tilt_std",
    "hip_tilt_max",
    "hip_tilt_range",
    "hip_tilt_p75",
    "hip_tilt_p90",
    "hip_osc_std",
    "hip_osc_range",
    "wrist_mean",
    "wrist_std",
    "elbow_std",
    "tilt_vel_mean",
    "tilt_vel_std",
    "lean_gt_15",
    "lean_gt_25",
    "lean_gt_40",
    "hip_gt_15",
    "hip_gt_25",
]

TANDEM_FEATURES = [
    "sh_tilt_mean",
    "sh_tilt_std",
    "sh_tilt_max",
    "sh_tilt_range",
    "sh_tilt_p75",
    "sh_tilt_p90",
    "hip_tilt_mean",
    "hip_tilt_std",
    "hip_tilt_max",
    "hip_tilt_range",
    "hip_tilt_p75",
    "hip_tilt_p90",
    "hip_osc_std",
    "hip_osc_range",
    "trunk_ap_lean_mean",
    "trunk_ap_lean_std",
    "trunk_ap_lean_max",
    "trunk_ap_lean_range",
    "ankle_rhythm_std",
    "ankle_rhythm_freq",
    "ankle_rhythm_entropy",
    "step_asymmetry",
    "wrist_spread_mean",
    "wrist_spread_std",
    "elbow_spread_std",
    "sh_tilt_vel_mean",
    "sh_tilt_vel_std",
    "lean_gt15",
    "lean_gt25",
    "lean_gt40",
    "hip_gt15",
    "hip_gt25",
    "knee_spread_mean",
    "knee_spread_std",
    "spine_lat_mean",
    "spine_lat_std",
    "hip_vel_mean",
    "hip_vel_std",
    "hip_acc_mean",
]

RESOLUTIONS = [(640, 480), (960, 540), (1280, 720), (480, 270)]

app = FastAPI(title="Parkinson Video AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Do not load all models at startup. Railway memory is limited, so models are
# loaded lazily inside the required endpoint, then released after prediction.
PROCESS_LOCK = threading.Lock()

finger_model_name = "Random Forest"
finger_model_metrics = {
    "test_acc": 0.905263,
    "test_auc": 0.949231,
    "sensitivity": 0.892308,
    "specificity": 0.933333,
}
romberg_model_name = "Gradient Boosting"
tandem_model_name = "Random Forest"


def _unwrap_model(obj: Any) -> Any:
    return obj["model"] if isinstance(obj, dict) and "model" in obj else obj


def _load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return _unwrap_model(joblib.load(path))


def _try_start_processing() -> None:
    # Prevent multiple heavy video/model requests from running together and
    # exhausting Railway RAM. The Flutter app / Swagger gets a clear response.
    if not PROCESS_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Server is currently processing another video. Please try again in a moment.",
        )


def _finish_processing() -> None:
    try:
        PROCESS_LOCK.release()
    except RuntimeError:
        pass
    gc.collect()


mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# MediaPipe Pose landmark indices used by the new Finger-to-Nose model.
NOSE = 0
L_WRIST, R_WRIST = 15, 16
L_ELBOW, R_ELBOW = 13, 14
L_SHOULDER, R_SHOULDER = 11, 12
L_INDEX, R_INDEX = 19, 20
L_HIP, R_HIP = 23, 24



def _round_dict(d: Dict[str, Any], ndigits: int = 6) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            out[k] = round(float(v), ndigits)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _label_payload(
    test_name: str,
    prob: np.ndarray,
    features: Dict[str, float],
    frames_used: int,
    chart_data: Dict[str, List[float]] | None = None,
) -> Dict[str, Any]:
    p_healthy = float(prob[0])
    p_patient = float(prob[1])
    label = "PATIENT" if p_patient >= 0.5 else "HEALTHY"
    confidence = max(p_healthy, p_patient) * 100.0
    score = p_healthy * 100.0

    return {
        "test": test_name,
        "label": label,
        "prediction": label,
        "score": round(score, 2),
        "confidence": round(confidence, 2),
        "p_healthy": round(p_healthy, 6),
        "p_patient": round(p_patient, 6),
        "frames_used": int(frames_used),
        "features": _round_dict(features),
        "chart_data": chart_data or {},
    }


def ultra_recovery_processing(frame: np.ndarray) -> np.ndarray:
    gamma = 1.8
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    frame = cv2.LUT(frame, table)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)



def _safe_fft_features(signal: np.ndarray, fps: float) -> Tuple[float, float, float]:
    n = len(signal)

    if n < 10:
        return 0.0, 0.0, 0.0

    sig = signal - np.mean(signal)
    yf = np.abs(fft(sig))[: n // 2]
    xf = fftfreq(n, 1.0 / fps)[: n // 2]

    if len(yf) <= 1:
        return 0.0, 0.0, 0.0

    idx = int(np.argmax(yf[1:]) + 1)
    dom_freq = float(abs(xf[idx]))
    dom_power = float(yf[idx])
    total_pow = float(np.sum(yf) + 1e-9)

    mask = (xf >= 3) & (xf <= 12)
    tremor_pow = float(np.sum(yf[mask]))
    tremor_ratio = tremor_pow / total_pow

    return dom_freq, dom_power / total_pow, tremor_ratio


def _elbow_angle(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    v1 = shoulder - elbow
    v2 = wrist - elbow

    cos_a = np.einsum("ij,ij->i", v1, v2) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9
    )

    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def _extract_ftn_vector(landmarks: np.ndarray, fps: float = 25.0) -> List[float]:
    shoulder_c = (landmarks[:, L_SHOULDER] + landmarks[:, R_SHOULDER]) / 2
    hip_c = (landmarks[:, L_HIP] + landmarks[:, R_HIP]) / 2

    body_height = float(np.mean(np.linalg.norm(shoulder_c - hip_c, axis=1)))
    if body_height < 1e-5:
        body_height = 1.0

    nose_xy = landmarks[:, NOSE, :2]

    l_wrist = landmarks[:, L_WRIST, :2]
    r_wrist = landmarks[:, R_WRIST, :2]

    l_dist_w = float(np.mean(np.linalg.norm(l_wrist - nose_xy, axis=1)))
    r_dist_w = float(np.mean(np.linalg.norm(r_wrist - nose_xy, axis=1)))

    if l_dist_w <= r_dist_w:
        wrist = l_wrist
        fingertip = landmarks[:, L_INDEX, :2]
        elbow_lm = landmarks[:, L_ELBOW, :2]
        shoulder_lm = landmarks[:, L_SHOULDER, :2]
    else:
        wrist = r_wrist
        fingertip = landmarks[:, R_INDEX, :2]
        elbow_lm = landmarks[:, R_ELBOW, :2]
        shoulder_lm = landmarks[:, R_SHOULDER, :2]

    ftn_dist = np.linalg.norm(fingertip - nose_xy, axis=1) / body_height

    ftn_dist_mean = float(np.mean(ftn_dist))
    ftn_dist_std = float(np.std(ftn_dist))
    ftn_dist_min = float(np.min(ftn_dist))
    ftn_dist_max = float(np.max(ftn_dist))
    ftn_dist_range = ftn_dist_max - ftn_dist_min
    ftn_jitter = float(np.mean(np.abs(np.diff(ftn_dist)))) if len(ftn_dist) > 1 else 0.0

    endpoint_accuracy = float(np.mean(ftn_dist < 0.05))
    miss_rate = float(np.mean(ftn_dist > 0.15))

    wrist_norm = wrist / body_height
    steps = np.linalg.norm(np.diff(wrist_norm, axis=0), axis=1)

    path_length = float(np.sum(steps)) if len(steps) else 0.0
    straight_d = float(np.linalg.norm(wrist_norm[-1] - wrist_norm[0]))
    path_efficiency = straight_d / (path_length + 1e-9)

    sway_x = float(np.var(wrist_norm[:, 0]))
    sway_y = float(np.var(wrist_norm[:, 1]))

    velocity = np.linalg.norm(np.diff(wrist, axis=0), axis=1) / body_height
    vel_mean = float(np.mean(velocity)) if len(velocity) else 0.0
    vel_std = float(np.std(velocity)) if len(velocity) else 0.0
    vel_max = float(np.max(velocity)) if len(velocity) else 0.0

    acceleration = np.diff(velocity)
    acc_mean = float(np.mean(acceleration)) if len(acceleration) else 0.0
    acc_std = float(np.std(acceleration)) if len(acceleration) else 0.0

    tremor_freq, tremor_power, tremor_ratio = _safe_fft_features(ftn_dist, fps)
    signal_entropy = float(np.std(ftn_dist))

    angles = _elbow_angle(shoulder_lm, elbow_lm, wrist)
    elbow_angle_mean = float(np.mean(angles))
    elbow_angle_std = float(np.std(angles))
    elbow_angle_range = float(angles.max() - angles.min())

    body_sway_x = float(np.var(shoulder_c[:, 0]) / body_height)
    body_sway_y = float(np.var(shoulder_c[:, 1]) / body_height)

    movement_irregularity = (
        float(np.mean(np.abs(np.diff(wrist[:, 0])) + np.abs(np.diff(wrist[:, 1])))) / body_height
        if len(wrist) > 1
        else 0.0
    )

    jerk_mean = float(np.mean(np.abs(np.diff(acceleration)))) if len(acceleration) > 1 else 0.0

    return [
        ftn_dist_mean,
        ftn_dist_std,
        ftn_dist_min,
        ftn_dist_max,
        ftn_dist_range,
        ftn_jitter,
        endpoint_accuracy,
        miss_rate,
        path_length,
        path_efficiency,
        sway_x,
        sway_y,
        vel_mean,
        vel_std,
        vel_max,
        acc_mean,
        acc_std,
        tremor_freq,
        tremor_power,
        tremor_ratio,
        signal_entropy,
        elbow_angle_mean,
        elbow_angle_std,
        elbow_angle_range,
        body_sway_x,
        body_sway_y,
        movement_irregularity,
        jerk_mean,
    ]


def extract_finger_features(video_path: str, n_frames: int = 30) -> Tuple[Dict[str, float], int, Dict[str, List[float]]]:
    cap = cv2.VideoCapture(str(video_path))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if total < 5:
        cap.release()
        raise ValueError("Video has too few frames.")

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    lm_seq: List[np.ndarray] = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as pose:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()

            if not ret:
                continue

            for res in RESOLUTIONS:
                resized = cv2.resize(frame, res)
                enhanced = ultra_recovery_processing(resized)
                rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                r = pose.process(rgb)

                if r.pose_landmarks:
                    arr = np.array(
                        [[lm.x, lm.y, lm.z] for lm in r.pose_landmarks.landmark],
                        dtype=float,
                    )
                    lm_seq.append(arr)
                    break

    cap.release()

    if len(lm_seq) < 5:
        raise ValueError("Not enough landmarks detected. Make sure the upper body, arm, and face are visible.")

    vector = _extract_ftn_vector(np.stack(lm_seq), fps=fps)
    features = {name: round(float(value), 6) for name, value in zip(FINGER_FEATURES, vector)}

    chart = {
        "ftn_distance_signal": [],
        "note": ["New 28-feature Finger-to-Nose model"],
    }

    return features, len(lm_seq), chart



def _pose_sequence(video_path: str, n_frames: int = 30) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < 2:
        cap.release()
        raise ValueError("Video has too few frames.")

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    lm_seq: List[np.ndarray] = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as pose:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()

            if not ret:
                continue

            for res in RESOLUTIONS:
                rgb = cv2.cvtColor(cv2.resize(frame, res), cv2.COLOR_BGR2RGB)
                r = pose.process(rgb)

                if r.pose_landmarks:
                    arr = np.array(
                        [[l.x, l.y, l.z, l.visibility] for l in r.pose_landmarks.landmark],
                        dtype=float,
                    )
                    lm_seq.append(arr)
                    break

    cap.release()

    if len(lm_seq) < 5:
        raise ValueError("Not enough body landmarks detected. Make sure the full body is visible.")

    return lm_seq


def extract_romberg_features(video_path: str, n_frames: int = 30) -> Tuple[Dict[str, float], int, Dict[str, List[float]]]:
    lm_seq = _pose_sequence(video_path, n_frames)
    lm3 = [s[:, :3] for s in lm_seq]

    torso_widths = np.array([abs(lm[11, 0] - lm[12, 0]) for lm in lm3])
    torso_w = float(np.mean(torso_widths) + 1e-6)

    sh_tilts = np.array([
        np.degrees(np.arctan2(abs(lm[11, 1] - lm[12, 1]), abs(lm[11, 0] - lm[12, 0]) + 1e-6))
        for lm in lm3
    ])

    hip_tilts = np.array([
        np.degrees(np.arctan2(abs(lm[23, 1] - lm[24, 1]), abs(lm[23, 0] - lm[24, 0]) + 1e-6))
        for lm in lm3
    ])

    hip_x = np.array([(lm[23, 0] + lm[24, 0]) / 2 for lm in lm3])
    hip_x_c = hip_x - np.mean(hip_x)

    hip_osc_std = np.std(hip_x_c) / torso_w
    hip_osc_range = (np.max(hip_x_c) - np.min(hip_x_c)) / torso_w

    wrist_spread = np.array([abs(lm[15, 0] - lm[16, 0]) for lm in lm3])
    wrist_spread_mean = np.mean(wrist_spread) / torso_w
    wrist_spread_std = np.std(wrist_spread) / torso_w

    elbow_spread = np.array([abs(lm[13, 0] - lm[14, 0]) for lm in lm3])
    elbow_spread_std = np.std(elbow_spread) / torso_w

    sh_tilt_diff = np.abs(np.diff(sh_tilts))
    sh_tilt_velocity_mean = sh_tilt_diff.mean() if len(sh_tilt_diff) else 0.0
    sh_tilt_velocity_std = sh_tilt_diff.std() if len(sh_tilt_diff) else 0.0

    values = np.array([
        sh_tilts.mean(),
        sh_tilts.std(),
        sh_tilts.max(),
        sh_tilts.max() - sh_tilts.min(),
        np.percentile(sh_tilts, 75),
        np.percentile(sh_tilts, 90),

        hip_tilts.mean(),
        hip_tilts.std(),
        hip_tilts.max(),
        hip_tilts.max() - hip_tilts.min(),
        np.percentile(hip_tilts, 75),
        np.percentile(hip_tilts, 90),

        hip_osc_std,
        hip_osc_range,

        wrist_spread_mean,
        wrist_spread_std,
        elbow_spread_std,

        sh_tilt_velocity_mean,
        sh_tilt_velocity_std,

        np.mean(sh_tilts > 15),
        np.mean(sh_tilts > 25),
        np.mean(sh_tilts > 40),

        np.mean(hip_tilts > 15),
        np.mean(hip_tilts > 25),
    ], dtype=float)

    features = {name: round(float(val), 6) for name, val in zip(ROMBERG_FEATURES, values)}

    chart = {
        "shoulder_tilt_signal": [round(float(x), 3) for x in sh_tilts[:120]],
        "hip_tilt_signal": [round(float(x), 3) for x in hip_tilts[:120]],
        "hip_sway_signal": [round(float(x), 6) for x in hip_x_c[:120]],
    }

    return features, len(lm_seq), chart


def extract_tandem_features(video_path: str, n_frames: int = 30) -> Tuple[Dict[str, float], int, Dict[str, List[float]]]:
    lm_seq = _pose_sequence(video_path, n_frames)
    lm3 = [s[:, :3] for s in lm_seq]

    torso_widths = np.array([abs(lm[11, 0] - lm[12, 0]) for lm in lm3])
    torso_w = float(np.mean(torso_widths) + 1e-6)

    sh_tilts = np.array([
        np.degrees(np.arctan2(abs(lm[11, 1] - lm[12, 1]), abs(lm[11, 0] - lm[12, 0]) + 1e-6))
        for lm in lm3
    ])

    hip_tilts = np.array([
        np.degrees(np.arctan2(abs(lm[23, 1] - lm[24, 1]), abs(lm[23, 0] - lm[24, 0]) + 1e-6))
        for lm in lm3
    ])

    hip_x = np.array([(lm[23, 0] + lm[24, 0]) / 2 for lm in lm3])
    hip_x_c = hip_x - np.mean(hip_x)

    hip_osc_std = np.std(hip_x_c) / torso_w
    hip_osc_range = (np.max(hip_x_c) - np.min(hip_x_c)) / torso_w

    wrist_spread = np.array([abs(lm[15, 0] - lm[16, 0]) for lm in lm3])
    wrist_spread_mean = np.mean(wrist_spread) / torso_w
    wrist_spread_std = np.std(wrist_spread) / torso_w

    elbow_spread = np.array([abs(lm[13, 0] - lm[14, 0]) for lm in lm3])
    elbow_spread_std = np.std(elbow_spread) / torso_w

    sh_tilt_diff = np.abs(np.diff(sh_tilts))
    sh_tilt_velocity_mean = sh_tilt_diff.mean() if len(sh_tilt_diff) else 0.0
    sh_tilt_velocity_std = sh_tilt_diff.std() if len(sh_tilt_diff) else 0.0

    large_lean_15 = np.mean(sh_tilts > 15)
    large_lean_25 = np.mean(sh_tilts > 25)
    large_lean_40 = np.mean(sh_tilts > 40)
    large_hip_15 = np.mean(hip_tilts > 15)
    large_hip_25 = np.mean(hip_tilts > 25)

    sh_mids = np.array([(lm[11] + lm[12]) / 2 for lm in lm3])
    hip_mids = np.array([(lm[23] + lm[24]) / 2 for lm in lm3])
    spine = sh_mids - hip_mids

    spine_lateral = np.abs(spine[:, 0]) / (np.abs(spine[:, 1]) + 1e-6)
    trunk_ap = np.abs(spine[:, 1])

    ankle_dist = np.array([
        abs(lm[27, 0] - lm[28, 0])
        for lm in lm3
    ])

    ankle_rhythm_std = np.std(ankle_dist)

    fft_vals = np.abs(np.fft.fft(ankle_dist))
    ankle_rhythm_freq = np.argmax(fft_vals[1:]) + 1 if len(fft_vals) > 1 else 0

    prob = ankle_dist / (np.sum(ankle_dist) + 1e-6)
    ankle_rhythm_entropy = -np.sum(prob * np.log(prob + 1e-6))

    left_steps = np.array([lm[27, 1] for lm in lm3])
    right_steps = np.array([lm[28, 1] for lm in lm3])
    step_asymmetry = np.mean(np.abs(left_steps - right_steps))

    knee_spread = np.array([
        abs(lm[25, 0] - lm[26, 0])
        for lm in lm3
    ])

    knee_spread_mean = np.mean(knee_spread) / torso_w
    knee_spread_std = np.std(knee_spread) / torso_w

    hip_vel = np.diff(hip_x_c)
    hip_acc = np.diff(hip_vel)

    hip_vel_mean = np.mean(np.abs(hip_vel)) if len(hip_vel) else 0.0
    hip_vel_std = np.std(hip_vel) if len(hip_vel) else 0.0
    hip_acc_mean = np.mean(np.abs(hip_acc)) if len(hip_acc) else 0.0

    values = np.array([
        sh_tilts.mean(),
        sh_tilts.std(),
        sh_tilts.max(),
        sh_tilts.max() - sh_tilts.min(),
        np.percentile(sh_tilts, 75),
        np.percentile(sh_tilts, 90),

        hip_tilts.mean(),
        hip_tilts.std(),
        hip_tilts.max(),
        hip_tilts.max() - hip_tilts.min(),
        np.percentile(hip_tilts, 75),
        np.percentile(hip_tilts, 90),

        hip_osc_std,
        hip_osc_range,

        trunk_ap.mean(),
        trunk_ap.std(),
        trunk_ap.max(),
        trunk_ap.max() - trunk_ap.min(),

        ankle_rhythm_std,
        ankle_rhythm_freq,
        ankle_rhythm_entropy,

        step_asymmetry,

        wrist_spread_mean,
        wrist_spread_std,
        elbow_spread_std,

        sh_tilt_velocity_mean,
        sh_tilt_velocity_std,

        large_lean_15,
        large_lean_25,
        large_lean_40,

        large_hip_15,
        large_hip_25,

        knee_spread_mean,
        knee_spread_std,

        spine_lateral.mean(),
        spine_lateral.std(),

        hip_vel_mean,
        hip_vel_std,
        hip_acc_mean,
    ], dtype=float)

    features = {name: round(float(val), 6) for name, val in zip(TANDEM_FEATURES, values)}

    chart = {
        "shoulder_tilt_signal": [round(float(x), 3) for x in sh_tilts[:120]],
        "hip_tilt_signal": [round(float(x), 3) for x in hip_tilts[:120]],
        "hip_sway_signal": [round(float(x), 6) for x in hip_x_c[:120]],
        "spine_lateral_signal": [round(float(x), 6) for x in spine_lateral[:120]],
    }

    return features, len(lm_seq), chart


async def _save_upload(video: UploadFile) -> str:
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await video.read()
        tmp.write(content)
        return tmp.name


class ChatRequest(BaseModel):
    message: str


def _medical_chatbot_answer(message: str) -> str:
    text = (message or "").strip().lower()

    disclaimer = "\n\nتنبيه مهم: هذه المعلومات للتوعية فقط وليست بديلاً عن استشارة الطبيب أو التشخيص الطبي المباشر."

    if not text:
        return "من فضلك اكتب سؤالك عن التصلب المتعدد MS فقط." + disclaimer

    # The chatbot is intentionally restricted to Multiple Sclerosis (MS) only.
    # Parkinson-related questions are refused instead of answered.
    parkinson_keywords = [
        "parkinson", "parkinson's", "باركنسون", "شلل رعاش", "الرعاش", "رعاش",
        "tremor", "tremors", "dopamine", "levodopa", "ل-dopa", "ليفودوبا",
    ]

    if any(k in text for k in parkinson_keywords):
        return (
            "أنا غير مخصص للإجابة عن مرض باركنسون. أقدر أساعدك فقط في الأسئلة المتعلقة بالتصلب المتعدد MS / Multiple Sclerosis."
            + disclaimer
        )

    non_ms_keywords = [
        "football", "movie", "game", "سياسة", "ماتش", "فيلم", "أغنية", "طبخ", "برمجة",
        "diabetes", "سكر", "ضغط", "hypertension", "heart", "قلب", "سرطان", "cancer",
    ]
    if any(k in text for k in non_ms_keywords):
        return "أنا مخصص فقط للأسئلة الطبية المتعلقة بالتصلب المتعدد MS / Multiple Sclerosis." + disclaimer

    ms_keywords = [
        "ms", "multiple sclerosis", "sclerosis", "تصلب", "التصلب", "متعدد",
        "تصلب متعدد", "التصلب المتعدد", "mri", "relapse", "انتكاسة", "myelin", "مايلين",
        "fatigue", "إرهاق", "ارهاق", "تنميل", "numbness", "vision", "رؤية",
        "balance", "اتزان", "walking", "مشي", "spasticity", "تيبس", "تشنج",
    ]

    # Emergency / red flag guidance is allowed because it is safety-critical, but still MS-focused.
    if any(k in text for k in ["emergency", "طوارئ", "خطر", "خطير", "مفاجئ", "sudden", "doctor", "دكتور", "طبيب"]):
        return (
            "لو مريض MS ظهرت عليه أعراض شديدة أو مفاجئة مثل ضعف مفاجئ، فقدان أو تشوش شديد في الرؤية، سقوط متكرر، ألم شديد، أو تدهور سريع، الأفضل التواصل مع طبيب أعصاب أو الطوارئ فورًا."
            + disclaimer
        )

    if any(k in text for k in ms_keywords):
        if any(k in text for k in ["symptom", "symptoms", "أعراض", "اعراض", "signs"]):
            return (
                "أعراض التصلب المتعدد MS قد تشمل: تنميل أو ضعف بالأطراف، مشاكل في الاتزان أو المشي، إرهاق شديد، زغللة أو مشاكل بالرؤية، تيبس أو تشنجات عضلية، وصعوبة في التركيز أو الذاكرة."
                + disclaimer
            )
        if any(k in text for k in ["treatment", "علاج", "دواء", "ادوية", "أدوية", "therapy"]):
            return (
                "علاج MS يختلف حسب الحالة، وقد يشمل أدوية لتقليل نشاط المرض، علاج الانتكاسات، علاج طبيعي، وتمارين لتحسين الاتزان والحركة. تحديد العلاج ونوع الدواء لازم يكون بواسطة طبيب أعصاب."
                + disclaimer
            )
        if any(k in text for k in ["test", "اختبار", "تشخيص", "diagnosis", "mri", "تحليل"]):
            return (
                "تشخيص MS يعتمد عادة على تقييم طبيب الأعصاب، MRI، التاريخ المرضي، وأحيانًا تحاليل أو فحوصات إضافية. اختبارات التطبيق تساعد في المتابعة وتقييم بعض الوظائف، لكنها لا تعتبر تشخيصًا نهائيًا."
                + disclaimer
            )
        if any(k in text for k in ["fatigue", "إرهاق", "ارهاق", "tired", "تعب"]):
            return (
                "الإرهاق من الأعراض الشائعة في MS. قد يتحسن بتنظيم النوم، تقسيم المجهود، العلاج الطبيعي، ومراجعة الطبيب لاستبعاد أسباب أخرى مثل الأنيميا أو مشاكل الغدة أو تأثير الأدوية."
                + disclaimer
            )
        if any(k in text for k in ["relapse", "attack", "انتكاسة", "هجمة"]):
            return (
                "انتكاسة MS تعني ظهور أعراض عصبية جديدة أو زيادة واضحة في أعراض قديمة لمدة غالبًا أكثر من 24 ساعة، بدون سبب واضح مثل حرارة أو عدوى. عند الاشتباه في انتكاسة لازم التواصل مع طبيب الأعصاب."
                + disclaimer
            )
        return (
            "التصلب المتعدد MS هو مرض مناعي يؤثر على الجهاز العصبي المركزي، وقد يسبب مشاكل في الحركة، الاتزان، الإحساس، الرؤية، والإرهاق. المتابعة المنتظمة مع طبيب الأعصاب مهمة لتقييم تطور الحالة وخطة العلاج."
            + disclaimer
        )

    return (
        "أنا مخصص فقط للأسئلة المتعلقة بالتصلب المتعدد MS / Multiple Sclerosis. اسألني مثلًا عن أعراض MS، التشخيص، الانتكاسة، الإرهاق، العلاج، أو معنى اختبارات المتابعة داخل التطبيق."
        + disclaimer
    )


@app.post("/chat")
def chat(request: ChatRequest) -> Dict[str, str]:
    return {"response": _medical_chatbot_answer(request.message)}



@app.get("/")
def root() -> Dict[str, str]:
    return {
        "status": "ok",
        "message": "Parkinson AI backend is running",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "finger_model": FINGER_MODEL_PATH.exists(),
        "finger_model_name": finger_model_name,
        "finger_model_metrics": _round_dict({k: v for k, v in finger_model_metrics.items() if v is not None}),
        "romberg_model": ROMBERG_MODEL_PATH.exists(),
        "romberg_model_name": romberg_model_name,
        "tandem_model": TANDEM_MODEL_PATH.exists(),
        "tandem_model_name": tandem_model_name,
    }


@app.post("/analyze/finger")
async def analyze_finger(video: UploadFile = File(...)) -> Dict[str, Any]:
    _try_start_processing()
    path = await _save_upload(video)
    model = None

    try:
        features, frames_used, chart = extract_finger_features(path)

        X = pd.DataFrame(
            [[features[k] for k in FINGER_FEATURES]],
            columns=FINGER_FEATURES,
        )

        model = _load_model(FINGER_MODEL_PATH)
        prob = model.predict_proba(X)[0]

        payload = _label_payload("finger_to_nose", prob, features, frames_used, chart)
        payload["model_name"] = finger_model_name
        payload["features_count"] = len(FINGER_FEATURES)

        return payload

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    finally:
        model = None
        try:
            os.remove(path)
        except OSError:
            pass
        _finish_processing()


@app.post("/analyze/romberg")
async def analyze_romberg(video: UploadFile = File(...)) -> Dict[str, Any]:
    _try_start_processing()
    path = await _save_upload(video)
    model = None

    try:
        features, frames_used, chart = extract_romberg_features(path)

        X = np.array(
            [[features[k] for k in ROMBERG_FEATURES]],
            dtype=float,
        )

        model = _load_model(ROMBERG_MODEL_PATH)
        prob = model.predict_proba(X)[0]

        payload = _label_payload("romberg", prob, features, frames_used, chart)
        payload["model_name"] = romberg_model_name
        payload["features_count"] = len(ROMBERG_FEATURES)

        return payload

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    finally:
        model = None
        try:
            os.remove(path)
        except OSError:
            pass
        _finish_processing()


@app.post("/analyze/tandem")
async def analyze_tandem(video: UploadFile = File(...)) -> Dict[str, Any]:
    _try_start_processing()
    path = await _save_upload(video)
    model = None

    try:
        features, frames_used, chart = extract_tandem_features(path)

        X = np.array(
            [[features[k] for k in TANDEM_FEATURES]],
            dtype=float,
        )

        model = _load_model(TANDEM_MODEL_PATH)
        prob = model.predict_proba(X)[0]

        payload = _label_payload("tandem", prob, features, frames_used, chart)
        payload["model_name"] = tandem_model_name
        payload["features_count"] = len(TANDEM_FEATURES)

        return payload

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    finally:
        model = None
        try:
            os.remove(path)
        except OSError:
            pass
        _finish_processing()
