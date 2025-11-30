"""
WebcamMouseCtrl: suivi des mains (une ou deux) avec MediaPipe Hands + filtrage One Euro.
Affiche les landmarks, la bbox, et un curseur lisse sur le bout de l'index pour chaque main.
Overlay FPS. Pas de clic/pinch ni controle souris dans cette version.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.007, d_cutoff: float = 1.5) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            return x

        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev if self.x_prev is not None else x

        self.t_prev = t
        dx = (x - self.x_prev) / dt if self.x_prev is not None else 0.0

        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * (self.x_prev if self.x_prev is not None else x)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


@dataclass
class HandObservation:
    landmarks: Tuple[Tuple[float, float], ...]
    handedness: str
    score: float
    bbox: Tuple[float, float, float, float]
    index_tip: Tuple[float, float]
    raw_landmarks: object  # NormalizedLandmarkList, kept generic to avoid heavy typing imports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hand tracking preview with MediaPipe Hands + One Euro smoothing")
    parser.add_argument("--camera-index", type=int, default=0, help="Index of the webcam (0 default)")
    parser.add_argument("--width", type=int, default=640, help="Requested capture width")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height")
    parser.add_argument("--max-hands", type=int, default=2, help="Maximum number of hands to track (lower = faster)")
    parser.add_argument("--model-complexity", type=int, default=0, choices=[0, 1], help="MediaPipe Hands model complexity (0 = faster)")
    parser.add_argument(
        "--inference-scale",
        type=float,
        default=0.7,
        help="Downscale factor applied before running MediaPipe (0.5-0.8 recommended for FPS gain)",
    )
    parser.add_argument("--no-mirror", action="store_true", help="Disable horizontal flip of the webcam preview")
    parser.add_argument("--show-fps", action="store_true", help="Overlay FPS counter")
    parser.add_argument("--min-cutoff", type=float, default=1.2, help="One Euro min cutoff")
    parser.add_argument("--beta", type=float, default=0.007, help="One Euro beta (speed coefficient)")
    parser.add_argument("--d-cutoff", type=float, default=1.5, help="One Euro derivative cutoff")
    return parser.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    return cap


def extract_hand_observations(result) -> List[HandObservation]:
    if not result.multi_hand_landmarks:
        return []

    observations: List[HandObservation] = []
    handedness_list = result.multi_handedness or []

    for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
        coords = tuple((lm.x, lm.y) for lm in hand_landmarks.landmark)

        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        handedness = "Unknown"
        score = 0.0
        if idx < len(handedness_list):
            handed_cls = handedness_list[idx].classification[0]
            handedness = handed_cls.label
            score = handed_cls.score

        observations.append(
            HandObservation(
                landmarks=coords,
                handedness=handedness,
                score=score,
                bbox=(min_x, min_y, max_x, max_y),
                index_tip=coords[8],
                raw_landmarks=hand_landmarks,
            )
        )

    return observations


def draw_hud(frame, observations: List[HandObservation], cursors: Dict[str, Tuple[float, float]], fps: float) -> None:
    h, w = frame.shape[:2]
    overlay = "ESC quit"
    if fps > 0:
        overlay = f"{overlay} | FPS {fps:0.1f}"
    cv2.putText(frame, overlay, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    for obs in observations:
        min_x, min_y, max_x, max_y = obs.bbox
        x0, y0 = int(min_x * w), int(min_y * h)
        x1, y1 = int(max_x * w), int(max_y * h)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 255), 2)

        label = f"{obs.handedness} {obs.score:0.2f}"
        cv2.putText(frame, label, (x0 + 4, max(y0 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            obs.raw_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style(),
        )

    for hand_id, pos in cursors.items():
        cx, cy = int(pos[0] * w), int(pos[1] * h)
        cv2.drawMarker(frame, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.putText(frame, hand_id, (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


def main() -> int:
    args = parse_args()

    cv2.setUseOptimized(True)

    try:
        cap = open_camera(args.camera_index, args.width, args.height)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}")
        return 1

    hands = mp.solutions.hands.Hands(
        max_num_hands=args.max_hands,
        model_complexity=args.model_complexity,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    prev_ts = time.time()
    fps = 0.0
    cursor_norms: Dict[str, Tuple[float, float]] = {}
    x_filters: Dict[str, OneEuroFilter] = {}
    y_filters: Dict[str, OneEuroFilter] = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[warn] Failed to read frame from camera; exiting")
                break

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            proc_frame = frame
            if 0.05 < args.inference_scale < 0.999:
                new_w = max(1, int(frame.shape[1] * args.inference_scale))
                new_h = max(1, int(frame.shape[0] * args.inference_scale))
                proc_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            now = time.time()
            dt = now - prev_ts
            prev_ts = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

            rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            observations = extract_hand_observations(result)
            cursor_norms.clear()

            h, w = frame.shape[:2]
            for idx, obs in enumerate(observations):
                hand_id = f"{obs.handedness or 'Hand'}-{idx}"
                if hand_id not in x_filters:
                    x_filters[hand_id] = OneEuroFilter(min_cutoff=args.min_cutoff, beta=args.beta, d_cutoff=args.d_cutoff)
                    y_filters[hand_id] = OneEuroFilter(min_cutoff=args.min_cutoff, beta=args.beta, d_cutoff=args.d_cutoff)

                norm_x, norm_y = obs.index_tip
                target_x = x_filters[hand_id](norm_x * w, now)
                target_y = y_filters[hand_id](norm_y * h, now)
                target_x = clamp(target_x, 0, w - 1)
                target_y = clamp(target_y, 0, h - 1)
                cursor_norms[hand_id] = (target_x / w, target_y / h)

            draw_hud(frame, observations, cursor_norms, fps if args.show_fps else 0.0)
            cv2.imshow("WebcamMouseCtrl", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
