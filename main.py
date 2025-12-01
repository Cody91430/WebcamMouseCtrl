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
WHEEL_ROTATION_SCALE = 3.5  # amplify hand rotation to spin the wheel faster
CAMERA_SAFE_MARGIN = 0.10   # 10% margin near webcam edges to avoid jumpy extremes
WHEEL_ANGLE_SMOOTH = 0.25   # smoothing for wheel rotation
WHEEL_CENTER_SMOOTH = 0.35  # smoothing for wheel position

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


def map_to_screen_norm(
    x_px: float,
    y_px: float,
    frame_w: float,
    frame_h: float,
    screen_size: Optional[Tuple[int, int]],
    margin: float = CAMERA_SAFE_MARGIN,
) -> Tuple[float, float]:
    """Clamp to a safe area, normalize, and adapt aspect ratio to the screen."""
    safe_left = margin * frame_w
    safe_right = frame_w * (1.0 - margin)
    safe_top = margin * frame_h
    safe_bottom = frame_h * (1.0 - margin)

    clamped_x = clamp(x_px, safe_left, safe_right)
    clamped_y = clamp(y_px, safe_top, safe_bottom)

    safe_w = max(1.0, safe_right - safe_left)
    safe_h = max(1.0, safe_bottom - safe_top)

    norm_x = (clamped_x - safe_left) / safe_w
    norm_y = (clamped_y - safe_top) / safe_h

    if screen_size:
        screen_w, screen_h = screen_size
        cam_aspect = safe_w / safe_h
        screen_aspect = screen_w / screen_h if screen_h else cam_aspect
        if cam_aspect > screen_aspect:
            scale = screen_aspect / cam_aspect
            norm_y = norm_y * scale + (1 - scale) / 2.0
        elif cam_aspect < screen_aspect:
            scale = cam_aspect / screen_aspect
            norm_x = norm_x * scale + (1 - scale) / 2.0

    return clamp(norm_x, 0.0, 1.0), clamp(norm_y, 0.0, 1.0)


def smooth_angle(prev: Optional[float], target: float, alpha: float) -> float:
    """Smoothly interpolate angles, handling wrap-around."""
    if prev is None:
        return target
    delta = math.atan2(math.sin(target - prev), math.cos(target - prev))
    return prev + alpha * delta


def smooth_point(prev: Optional[Tuple[float, float]], target: Tuple[float, float], alpha: float) -> Tuple[float, float]:
    if prev is None:
        return target
    return (prev[0] + alpha * (target[0] - prev[0]), prev[1] + alpha * (target[1] - prev[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hand tracking preview with MediaPipe Hands + One Euro smoothing")
    parser.add_argument("--camera-index", type=int, default=0, help="Index of the webcam (0 default)")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width (16:9 default 1280)")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height (16:9 default 720)")
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


LETTER_WHEEL = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def select_letter_from_wheel(angle: float, letters: List[str]) -> Optional[str]:
    """Return the letter closest to the top (12 o'clock) for the given wheel angle."""
    if not letters:
        return None
    target_angle = -math.pi / 2  # 12 o'clock
    best_letter = None
    best_diff = float("inf")
    n = len(letters)
    for i, letter in enumerate(letters):
        theta = angle + 2 * math.pi * i / n - math.pi / 2.0
        diff = abs((theta - target_angle + math.pi) % (2 * math.pi) - math.pi)
        if diff < best_diff:
            best_diff = diff
            best_letter = letter
    return best_letter


def put_text_with_shadow(
    frame,
    text: str,
    org: Tuple[int, int],
    font_scale: float,
    color: Tuple[int, int, int],
    shadow_color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
    shadow_offset: Tuple[int, int] = (2, 2),
) -> None:
    """Draw text with a small shadow for better contrast."""
    x_off, y_off = shadow_offset
    cv2.putText(frame, text, (org[0] + x_off, org[1] + y_off), cv2.FONT_HERSHEY_SIMPLEX, font_scale, shadow_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def hand_rotation_angle(obs: "HandObservation", w: int, h: int) -> float:
    """Estimate hand rotation (radians) using wrist-to-middle-MCP vector."""
    wrist = obs.landmarks[0]
    middle_base = obs.landmarks[9]
    dx = (middle_base[0] - wrist[0]) * w
    dy = (middle_base[1] - wrist[1]) * h
    return math.atan2(dy, dx)


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
    wheel_overlay: Optional[Dict[str, object]] = None,
    typed_text: str = "",
) -> None:
    h, w = frame.shape[:2]
    overlay = "ESC quit"
    if fps > 0:
        overlay = f"{overlay} | FPS {fps:0.1f}"
    put_text_with_shadow(frame, overlay, (12, 28), 0.8, (255, 255, 255))
    if typed_text:
        put_text_with_shadow(frame, f"TXT: {typed_text[-30:]}", (12, h - 20), 0.7, (255, 255, 255))

    if render_hands:
        shadow_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 0), thickness=4, circle_radius=4)
        line_spec = mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
        wheel_hand_id = wheel_overlay.get("hand_id") if wheel_overlay else None
        for idx, obs in enumerate(observations):
            hand_id = f"{obs.handedness or 'Hand'}-{idx}"
            if wheel_hand_id and hand_id == wheel_hand_id:
                continue
            label = f"{obs.handedness} {obs.score:0.2f}"
            put_text_with_shadow(frame, label, (12, 52 + 24 * idx), 0.7, (255, 255, 255))
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                obs.raw_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                shadow_spec,
                shadow_spec,
            )
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                obs.raw_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                line_spec,
                line_spec,
            )

    if wheel_overlay:
        cx, cy = wheel_overlay.get("center", (w // 2, h // 2))  # type: ignore[assignment]
        angle = wheel_overlay.get("angle", 0.0)  # type: ignore[assignment]
        letters = wheel_overlay.get("letters", LETTER_WHEEL)  # type: ignore[assignment]
        base_radius = int(wheel_overlay.get("radius", min(w, h) // 4))  # type: ignore[arg-type]
        inner_offset = max(6, base_radius // 6)
        wheel_color = (0, 255, 0)
        text_color = (255, 255, 255)
        shadow_color = (0, 0, 0)
        n = len(letters)
        selected_letter = None
        selected_idx: Optional[int] = None
        wheel_hand_id = wheel_overlay.get("hand_id")
        letter_draw_ops: List[Tuple[str, int, int]] = []
        selected_diff = float("inf")
        target_angle = -math.pi / 2  # 12 o'clock
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        letter_points: List[Tuple[int, int]] = []
        # Draw star first (shadow + white) so letters appear above it.
        # Points will be collected during letter placement to avoid re-computing angles.
        temp_points: List[Tuple[int, int]] = []
        for i in range(len(letters)):
            radius = base_radius if i % 2 == 0 else max(10, base_radius - inner_offset)
            theta = angle + 2 * math.pi * i / n - math.pi / 2.0
            lx = int(cx + radius * math.cos(theta))
            ly = int(cy + radius * math.sin(theta))
            temp_points.append((lx, ly))

        if len(temp_points) > 1:
            shadow_shift = (1, 1)
            for i in range(len(temp_points)):
                p1 = temp_points[i]
                p2 = temp_points[(i + 1) % len(temp_points)]
                cv2.line(frame, (p1[0] + shadow_shift[0], p1[1] + shadow_shift[1]), (p2[0] + shadow_shift[0], p2[1] + shadow_shift[1]), shadow_color, 2, cv2.LINE_AA)
            for i in range(len(temp_points)):
                p1 = temp_points[i]
                p2 = temp_points[(i + 1) % len(temp_points)]
                cv2.line(frame, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)

        for i, letter in enumerate(letters):
            lx, ly = temp_points[i]
            theta = angle + 2 * math.pi * i / n - math.pi / 2.0
            diff = abs((theta - target_angle + math.pi) % (2 * math.pi) - math.pi)
            if diff < selected_diff:
                selected_diff = diff
                selected_letter = letter
                selected_idx = i
            # Shadow letter slightly offset down-right, then main letter.
            (text_w, text_h), _ = cv2.getTextSize(letter, font, font_scale, font_thickness)
            text_x = lx - text_w // 2
            text_y = ly + text_h // 2
            letter_draw_ops.append((letter, text_x, text_y))
            letter_points.append((lx, ly))

        # Marker on selected letter position (behind the glyph so text stays readable).
        if selected_idx is not None and 0 <= selected_idx < len(letter_points):
            mx, my = letter_points[selected_idx]
            marker_color = (0, 255, 255)
            cv2.circle(frame, (mx, my), 14, shadow_color, 4, cv2.LINE_AA)
            cv2.circle(frame, (mx, my), 14, marker_color, 2, cv2.LINE_AA)

        # Draw letters after all lines so they stay on top.
        shadow_offset = (2, 2)
        for letter, text_x, text_y in letter_draw_ops:
            cv2.putText(frame, letter, (text_x + shadow_offset[0], text_y + shadow_offset[1]), font, font_scale, shadow_color, font_thickness, cv2.LINE_AA)
            cv2.putText(frame, letter, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        if selected_letter:
            center_scale = 1.6
            center_thickness = 3
            (tw, th), _ = cv2.getTextSize(selected_letter, font, center_scale, center_thickness)
            tx = cx - tw // 2
            ty = cy + th // 2
            put_text_with_shadow(frame, selected_letter, (tx, ty), center_scale, text_color, shadow_color, center_thickness)

    # Removed cursor markers and labels for a minimal HUD.


def main() -> int:
    args = parse_args()

    cv2.setUseOptimized(True)
    window_name = "WebcamMouseCtrl"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)
    try:
        # Keep the preview window above other windows for easier testing.
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

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
    pinch_middle_states: Dict[str, bool] = {}
    pinch_off_started: Dict[str, Optional[float]] = {}
    pinch_middle_off_started: Dict[str, Optional[float]] = {}
    warned_no_click = False
    prev_pinch_states: Dict[str, bool] = {}
    prev_pinch_middle_states: Dict[str, bool] = {}
    frame_idx = 0
    typed_text = ""
    last_observations: List[HandObservation] = []
    last_cursor_norms: Dict[str, Tuple[float, float]] = {}
    wheel_info: Optional[Dict[str, object]] = None
    last_wheel_info: Optional[Dict[str, object]] = None
    wheel_hand_id: Optional[str] = None
    last_wheel_hand_id: Optional[str] = None
    wheel_base_angle: Optional[float] = None
    wheel_angle_smooth: Optional[float] = None
    wheel_center_smooth: Optional[Tuple[float, float]] = None

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
                new_pinch_middle_states: Dict[str, bool] = {}

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
                    screen_size = cursor_controller.screen_size if cursor_controller is not None else None
                    mapped_x, mapped_y = map_to_screen_norm(target_x, target_y, w, h, screen_size)
                    cursor_norms[hand_id] = (mapped_x, mapped_y)

                    thumb_tip = obs.landmarks[4]
                    index_tip = obs.landmarks[8]
                    middle_tip = obs.landmarks[12]
                    pinch_dist = math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
                    pinch_dist_mid = math.hypot(thumb_tip[0] - middle_tip[0], thumb_tip[1] - middle_tip[1])
                    span = math.hypot(obs.bbox[2] - obs.bbox[0], obs.bbox[3] - obs.bbox[1])
                    span = span if span > 1e-6 else 1.0
                    pinch_ratio = pinch_dist / span
                    pinch_ratio_mid = pinch_dist_mid / span

                    prev_state = pinch_states.get(hand_id, False)
                    prev_state_mid = pinch_middle_states.get(hand_id, False)
                    pinch_on = pinch_ratio < 0.17
                    pinch_off = pinch_ratio > 0.24
                    pinch_on_mid = pinch_ratio_mid < 0.17
                    pinch_off_mid = pinch_ratio_mid > 0.24

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
                    if prev_state_mid:
                        if pinch_off_mid:
                            start = pinch_middle_off_started.get(hand_id)
                            if start is None:
                                pinch_middle_off_started[hand_id] = now
                                new_pinch_middle_states[hand_id] = True
                            elif now - start >= PINCH_RELEASE_DELAY:
                                new_pinch_middle_states[hand_id] = False
                            else:
                                new_pinch_middle_states[hand_id] = True
                        else:
                            pinch_middle_off_started[hand_id] = None
                            new_pinch_middle_states[hand_id] = True
                    else:
                        if pinch_on_mid:
                            pinch_middle_off_started[hand_id] = None
                            new_pinch_middle_states[hand_id] = True
                        else:
                            pinch_middle_off_started[hand_id] = None
                            new_pinch_middle_states[hand_id] = False

                pinch_states = new_pinch_states
                pinch_middle_states = new_pinch_middle_states
                pinch_off_started = {k: v for k, v in pinch_off_started.items() if k in pinch_states}
                pinch_middle_off_started = {k: v for k, v in pinch_middle_off_started.items() if k in pinch_middle_states}

                # Left-hand pinch -> left click on rising edge.
                if pyautogui is not None:
                    for hand_id, state in pinch_states.items():
                        if hand_id.lower().startswith("left") and state and not prev_pinch_states.get(hand_id, False):
                            try:
                                pyautogui.click(button="left")
                            except Exception:
                                pass
                    # Left-hand middle pinch -> right click on rising edge.
                    for hand_id, state in pinch_middle_states.items():
                        if hand_id.lower().startswith("left") and state and not prev_pinch_middle_states.get(hand_id, False):
                            try:
                                pyautogui.click(button="right")
                            except Exception:
                                pass
                elif not warned_no_click:
                    print("[warn] pyautogui not available; pinch click disabled.")
                    warned_no_click = True

                # Right-hand index pinch -> show rotating letter wheel.
                wheel_info = None
                right_indices = [i for i, obs in enumerate(observations) if obs.handedness.lower() == "right"]
                selected_letter = None
                wheel_hand_id = None
                if right_indices:
                    best_idx = max(right_indices, key=lambda i: observations[i].score)
                    best_id = f"{observations[best_idx].handedness or 'Hand'}-{best_idx}"
                    if pinch_states.get(best_id):
                        pos = cursor_norms.get(best_id)
                        if pos:
                            cx = pos[0] * w
                            cy = pos[1] * h
                            raw_angle = hand_rotation_angle(observations[best_idx], w, h) * WHEEL_ROTATION_SCALE
                            if wheel_base_angle is None:
                                wheel_base_angle = raw_angle
                            angle = raw_angle - wheel_base_angle
                            wheel_angle_smooth = smooth_angle(wheel_angle_smooth, angle, WHEEL_ANGLE_SMOOTH)
                            wheel_center_smooth = smooth_point(wheel_center_smooth, (cx, cy), WHEEL_CENTER_SMOOTH)
                            cx_int = int(wheel_center_smooth[0])
                            cy_int = int(wheel_center_smooth[1])
                            angle_to_use = wheel_angle_smooth if wheel_angle_smooth is not None else angle
                            radius = max(50, min(w, h) // 5)
                            selected_letter = select_letter_from_wheel(angle_to_use, LETTER_WHEEL)
                            wheel_hand_id = best_id
                            wheel_info = {
                                "center": (cx_int, cy_int),
                                "angle": angle_to_use,
                                "letters": LETTER_WHEEL,
                                "radius": radius,
                                "selected_letter": selected_letter,
                                "hand_id": wheel_hand_id,
                            }

                # Capture the letter on a rising pinch of the left hand while the wheel is visible.
                if wheel_info and selected_letter:
                    for hand_id, state in pinch_states.items():
                        if hand_id.lower().startswith("left") and state and not prev_pinch_states.get(hand_id, False):
                            typed_text += selected_letter
                            break

                # When the wheel-hand pinch is released, type the buffered text at the cursor.
                if last_wheel_hand_id and prev_pinch_states.get(last_wheel_hand_id, False) and not pinch_states.get(last_wheel_hand_id, False):
                    if typed_text:
                        if pyautogui is not None:
                            try:
                                pyautogui.typewrite(typed_text)
                                typed_text = ""
                            except Exception as exc:  # noqa: BLE001
                                print(f"[warn] Failed to type text: {exc}")
                        else:
                            print("[warn] pyautogui not available; cannot type text.")
                    wheel_base_angle = None
                    wheel_angle_smooth = None
                    wheel_center_smooth = None

                last_wheel_info = wheel_info
                last_wheel_hand_id = wheel_hand_id

                if wheel_info is None:
                    wheel_angle_smooth = None
                    wheel_center_smooth = None

                prev_pinch_states = pinch_states.copy()
                prev_pinch_middle_states = pinch_middle_states.copy()
                last_observations = observations
                last_cursor_norms = cursor_norms
            else:
                observations = last_observations
                cursor_norms = last_cursor_norms
                wheel_info = last_wheel_info

            if cursor_controller is not None and observations and should_process and not wheel_info:
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
                wheel_info,
                typed_text,
            )
            cv2.imshow(window_name, frame)

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
