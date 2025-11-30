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

try:
    import pyautogui

    pyautogui.FAILSAFE = False
except Exception:  # noqa: BLE001
    pyautogui = None

try:
    from cursor_control import CursorController
except Exception:
    CursorController = None  # type: ignore

PINCH_RELEASE_DELAY = 0.25  # seconds of tolerance before considering pinch released

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
    parser.add_argument("--no-draw-hands", action="store_true", help="Disable drawing landmarks/bboxes to save FPS")
    parser.add_argument(
        "--control-cursor",
        action="store_true",
        help="Move the OS cursor with the highest-score hand index tip (requires pyautogui)",
    )
    parser.add_argument(
        "--process-every",
        type=int,
        default=1,
        help="Run hand inference every Nth frame to save CPU (1 = every frame)",
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


def draw_hud(
    frame,
    observations: List[HandObservation],
    cursors: Dict[str, Tuple[float, float]],
    fps: float,
    pinch_states: Dict[str, bool],
    render_hands: bool,
) -> None:
    h, w = frame.shape[:2]
    overlay = "ESC quit"
    if fps > 0:
        overlay = f"{overlay} | FPS {fps:0.1f}"
    cv2.putText(frame, overlay, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    if render_hands:
        line_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        for obs in observations:
            label = f"{obs.handedness} {obs.score:0.2f}"
            cv2.putText(frame, label, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                obs.raw_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                line_spec,
                line_spec,
            )

    # Removed cursor markers and labels for a minimal HUD.


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

    cursor_controller = None
    if args.control_cursor:
        if CursorController is None or pyautogui is None:
            print("[warn] Cursor control requested but pyautogui/cursor_control not available; ignoring.")
        else:
            cursor_controller = CursorController()

    prev_ts = time.time()
    fps = 0.0
    cursor_norms: Dict[str, Tuple[float, float]] = {}
    x_filters: Dict[str, OneEuroFilter] = {}
    y_filters: Dict[str, OneEuroFilter] = {}
    pinch_states: Dict[str, bool] = {}
    pinch_off_started: Dict[str, Optional[float]] = {}
    warned_no_click = False
    frame_idx = 0
    last_observations: List[HandObservation] = []
    last_cursor_norms: Dict[str, Tuple[float, float]] = {}

    try:
        process_every = max(1, args.process_every)
        while True:
            frame_idx += 1
            ok, frame = cap.read()
            if not ok:
                print("[warn] Failed to read frame from camera; exiting")
                break

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            now = time.time()
            dt = now - prev_ts
            prev_ts = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

            if cursor_controller is not None and fps > 0:
                # Tie cursor update rate to current FPS, allow higher ceiling for snappier motion.
                cursor_controller.settings.max_rate_hz = max(20.0, min(fps, 240.0))

            should_process = frame_idx % process_every == 0
            h, w = frame.shape[:2]

            if should_process:
                proc_frame = frame
                if 0.05 < args.inference_scale < 0.999:
                    new_w = max(1, int(frame.shape[1] * args.inference_scale))
                    new_h = max(1, int(frame.shape[0] * args.inference_scale))
                    proc_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                observations = extract_hand_observations(result)
                cursor_norms = {}
                new_pinch_states: Dict[str, bool] = {}

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

                    thumb_tip = obs.landmarks[4]
                    index_tip = obs.landmarks[8]
                    pinch_dist = math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
                    span = math.hypot(obs.bbox[2] - obs.bbox[0], obs.bbox[3] - obs.bbox[1])
                    span = span if span > 1e-6 else 1.0
                    pinch_ratio = pinch_dist / span

                    prev_state = pinch_states.get(hand_id, False)
                    pinch_on = pinch_ratio < 0.17
                    pinch_off = pinch_ratio > 0.24

                    if prev_state:
                        if pinch_off:
                            start = pinch_off_started.get(hand_id)
                            if start is None:
                                pinch_off_started[hand_id] = now
                                new_pinch_states[hand_id] = True
                            elif now - start >= PINCH_RELEASE_DELAY:
                                new_pinch_states[hand_id] = False
                            else:
                                new_pinch_states[hand_id] = True
                        else:
                            pinch_off_started[hand_id] = None
                            new_pinch_states[hand_id] = True
                    else:
                        if pinch_on:
                            pinch_off_started[hand_id] = None
                            new_pinch_states[hand_id] = True
                        else:
                            pinch_off_started[hand_id] = None
                            new_pinch_states[hand_id] = False

                pinch_states = new_pinch_states
                pinch_off_started = {k: v for k, v in pinch_off_started.items() if k in pinch_states}
                last_observations = observations
                last_cursor_norms = cursor_norms
            else:
                observations = last_observations
                cursor_norms = last_cursor_norms

            if cursor_controller is not None and observations and should_process:
                # Only drive the cursor with the right hand index tip.
                right_indices = [i for i, obs in enumerate(observations) if obs.handedness.lower() == "right"]
                if right_indices:
                    best_idx = max(right_indices, key=lambda i: observations[i].score)
                    best_id = f"{observations[best_idx].handedness or 'Hand'}-{best_idx}"
                    pos = cursor_norms.get(best_id)
                    if pos:
                        cursor_controller.move_normalized(pos[0], pos[1])

            draw_hud(
                frame,
                observations,
                cursor_norms,
                fps if args.show_fps else 0.0,
                pinch_states,
                not args.no_draw_hands,
            )
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
