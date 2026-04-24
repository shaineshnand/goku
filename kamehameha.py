import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import winsound
except ImportError:
    winsound = None


WINDOW_NAME = "Dragon Ball - Kamehameha"
GLOW_COLOR = (255, 220, 90)  # BGR
CORE_COLOR = (255, 255, 255)
TEXT_COLOR = (50, 255, 50)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
SFX_DIR = os.path.join(os.path.dirname(__file__), "assets", "sfx")
CHARGE_WAV = os.path.join(SFX_DIR, "charge.wav")
FIRE_WAV = os.path.join(SFX_DIR, "fire.wav")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def blend_additive(base, glow, strength: float = 1.0):
    return cv2.addWeighted(base, 1.0, glow, strength, 0)


def draw_vignette(frame):
    h, w = frame.shape[:2]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    radius = np.sqrt(xv * xv + yv * yv)
    mask = np.clip(1.0 - (radius - 0.25) * 0.9, 0.55, 1.0)
    frame[:] = (frame.astype(np.float32) * mask[..., None]).astype(np.uint8)


def apply_blue_white_grading(frame, intensity: float):
    intensity = clamp(intensity, 0.0, 1.0)
    if intensity <= 0.0:
        return
    frame_f = frame.astype(np.float32)
    # Shift scene toward cool blue-white tones during charge/fire.
    frame_f[..., 0] = frame_f[..., 0] * (1.05 + 0.25 * intensity) + 20.0 * intensity
    frame_f[..., 1] = frame_f[..., 1] * (0.95 + 0.15 * intensity) + 10.0 * intensity
    frame_f[..., 2] = frame_f[..., 2] * (0.82 + 0.10 * intensity)
    # Add subtle bloom-like lift.
    frame_f += np.array([12.0, 10.0, 6.0], dtype=np.float32) * intensity
    np.clip(frame_f, 0, 255, out=frame_f)
    frame[:] = frame_f.astype(np.uint8)


def apply_chromatic_aberration(frame, amount: int):
    amount = max(0, amount)
    if amount == 0:
        return frame
    h, w = frame.shape[:2]
    m_left = np.float32([[1, 0, -amount], [0, 1, 0]])
    m_right = np.float32([[1, 0, amount], [0, 1, 0]])
    b, g, r = cv2.split(frame)
    b_s = cv2.warpAffine(b, m_left, (w, h), borderMode=cv2.BORDER_REFLECT)
    r_s = cv2.warpAffine(r, m_right, (w, h), borderMode=cv2.BORDER_REFLECT)
    return cv2.merge([b_s, g, r_s])


def play_tone(freq: int, duration_ms: int):
    if winsound is None:
        return
    try:
        winsound.Beep(freq, duration_ms)
    except RuntimeError:
        pass


def play_wav(path: str) -> bool:
    if winsound is None or not os.path.exists(path):
        return False
    try:
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        return True
    except RuntimeError:
        return False


def play_charge_sfx():
    if play_wav(CHARGE_WAV):
        return
    if winsound is not None:
        winsound.MessageBeep(winsound.MB_ICONASTERISK)


def play_fire_sfx():
    if play_wav(FIRE_WAV):
        return
    play_tone(280, 70)
    play_tone(420, 70)
    play_tone(640, 110)


def ensure_hand_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand landmark model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def create_hand_tracker():
    ensure_hand_model()
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hand_centers_mediapipe(frame, tracker, timestamp_ms: int):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = tracker.detect_for_video(mp_image, timestamp_ms=timestamp_ms)

    centers = []
    for hand_landmarks in result.hand_landmarks:
        xs = [lm.x for lm in hand_landmarks]
        ys = [lm.y for lm in hand_landmarks]
        cx = int(sum(xs) / len(xs) * frame.shape[1])
        cy = int(sum(ys) / len(ys) * frame.shape[0])
        centers.append((cx, cy))
    return centers


def draw_energy_ball(frame, center, radius: int, phase: float):
    x, y = center
    pulse = 0.12 * math.sin(phase * 1.6)
    rr = int(radius * (1.0 + pulse))

    h, w = frame.shape[:2]
    glow = np.zeros_like(frame, dtype=np.uint8)

    # Outer aura
    for i, col in enumerate([(255, 80, 20), (255, 140, 50), (255, 200, 130)]):
        r = rr + 34 - i * 10
        if r > 1:
            cv2.circle(glow, (x, y), r, col, -1, lineType=cv2.LINE_AA)

    # Plasma veins
    for i in range(8):
        ang = phase * 0.7 + i * (2.0 * math.pi / 8.0)
        px = int(x + math.cos(ang) * (rr + 7))
        py = int(y + math.sin(ang) * (rr + 7))
        cv2.line(glow, (x, y), (px, py), (255, 240, 190), 1, lineType=cv2.LINE_AA)

    frame[:] = blend_additive(frame, glow, 0.45)

    # Core with radial falloff
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    core = np.clip(1.0 - dist / max(rr, 1), 0.0, 1.0)
    shell = np.clip(1.0 - dist / max(rr + 16, 1), 0.0, 1.0)

    frame_f = frame.astype(np.float32)
    frame_f[..., 0] += 180.0 * shell
    frame_f[..., 1] += 120.0 * shell
    frame_f[..., 2] += 80.0 * shell
    frame_f += (core[..., None] * np.array([255.0, 255.0, 255.0], dtype=np.float32))
    np.clip(frame_f, 0, 255, out=frame_f)
    frame[:] = frame_f.astype(np.uint8)

    cv2.circle(frame, (x, y), rr + 2, (255, 250, 200), 1, lineType=cv2.LINE_AA)


def draw_beam(frame, start, end, phase: float):
    sx, sy = start
    ex, ey = end

    glow = np.zeros_like(frame, dtype=np.uint8)
    cv2.line(glow, (sx, sy), (ex, ey), (255, 150, 50), 34, lineType=cv2.LINE_AA)
    cv2.line(glow, (sx, sy), (ex, ey), (255, 210, 150), 22, lineType=cv2.LINE_AA)
    cv2.line(glow, (sx, sy), (ex, ey), (255, 250, 230), 8, lineType=cv2.LINE_AA)

    # Directional bloom: strongest near origin, fades toward beam tip.
    steps = 9
    for i in range(steps):
        t = i / max(steps - 1, 1)
        px = int(sx + (ex - sx) * t)
        py = int(sy + (ey - sy) * t)
        radius = int(22 - 12 * t)
        if radius > 0:
            color = (255, int(150 + 80 * (1.0 - t)), int(100 + 120 * (1.0 - t)))
            cv2.circle(glow, (px, py), radius, color, -1, lineType=cv2.LINE_AA)
    frame[:] = blend_additive(frame, glow, 0.35)

    segments = 18
    for i in range(segments + 1):
        t = i / segments
        x = int(sx + (ex - sx) * t)
        y = int(sy + (ey - sy) * t + 7 * math.sin(phase * 2.2 + i * 0.8))
        r = 2 + (i % 3 == 0)
        cv2.circle(frame, (x, y), r, CORE_COLOR, -1, lineType=cv2.LINE_AA)


def draw_particles(frame, center, phase: float, strength: float):
    x, y = center
    for i in range(26):
        a = phase * 0.6 + i * 0.43
        speed = 12 + (i % 7) * 2
        px = int(x + math.cos(a) * speed * strength)
        py = int(y + math.sin(a * 1.3) * speed * strength)
        color = (255, 180 + (i * 3) % 70, 120 + (i * 5) % 110)
        cv2.circle(frame, (px, py), 1 + (i % 2), color, -1, lineType=cv2.LINE_AA)


def draw_hud(frame, charged: bool, shooting: bool):
    cv2.putText(
        frame,
        "Move your hands!",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    if shooting:
        msg = "KAMEHAMEHAAAA!"
    elif charged:
        msg = "Hands together... CHARGED!"
    else:
        msg = "Bring both hands closer"
    cv2.putText(
        frame,
        msg,
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "GOKU",
        (25, frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 180, 255),
        3,
        cv2.LINE_AA,
    )


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera permissions and device.")

    tracker = create_hand_tracker()

    charged = False
    last_charge_time = 0.0
    charge_timeout = 3.0
    charge_close_threshold = 0.22
    fire_open_threshold = 0.26
    hard_fire_threshold = 0.34
    spread_delta_threshold = 0.02
    charge_hold_seconds = 0.18
    shooting_until = 0.0
    can_fire = False
    charge_start_time = 0.0
    prev_norm = None
    smooth_centers = []
    prev_charged = False
    prev_shooting = False
    fire_flash_end = 0.0
    shake_end = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        timestamp_ms = int(time.time() * 1000)
        hand_centers = detect_hand_centers_mediapipe(frame, tracker, timestamp_ms)
        hand_centers = sorted(hand_centers, key=lambda p: p[0])

        if len(hand_centers) == 2:
            if len(smooth_centers) != 2:
                smooth_centers = hand_centers
            else:
                alpha = 0.65
                smooth_centers = [
                    (
                        int(alpha * smooth_centers[i][0] + (1 - alpha) * hand_centers[i][0]),
                        int(alpha * smooth_centers[i][1] + (1 - alpha) * hand_centers[i][1]),
                    )
                    for i in range(2)
                ]
            hand_centers = smooth_centers
        else:
            smooth_centers = []

        now = time.time()
        phase = now * 10.0

        shooting = now < shooting_until
        if len(hand_centers) == 2:
            (x1, y1), (x2, y2) = hand_centers
            distance = math.hypot(x2 - x1, y2 - y1)
            norm = distance / max(w, h)

            if norm < charge_close_threshold:
                if charge_start_time <= 0.0:
                    charge_start_time = now
                if not charged and (now - charge_start_time) >= charge_hold_seconds:
                    charged = True
                    can_fire = True
                    last_charge_time = now
            else:
                charge_start_time = 0.0

            spread_delta = 0.0 if prev_norm is None else (norm - prev_norm)
            if charged and can_fire:
                if (norm > fire_open_threshold and spread_delta > spread_delta_threshold) or norm > hard_fire_threshold:
                    shooting_until = now + 0.35
                    shooting = True
                    can_fire = False
                    last_charge_time = now

            cv2.circle(frame, (x1, y1), 10, (60, 160, 255), -1)
            cv2.circle(frame, (x2, y2), 10, (60, 160, 255), -1)

            if charged:
                ball_x = int((x1 + x2) * 0.5)
                ball_y = int((y1 + y2) * 0.5)
                radius = int(clamp(45 - (norm * 120), 16, 40))
                draw_vignette(frame)
                draw_energy_ball(frame, (ball_x, ball_y), radius, phase)
                draw_particles(frame, (ball_x, ball_y), phase, strength=1.6)

                if shooting:
                    beam_len = int(w * 0.35)
                    draw_beam(
                        frame,
                        (ball_x, ball_y),
                        (ball_x + beam_len, ball_y - 15),
                        phase,
                    )
                    draw_particles(frame, (ball_x + beam_len, ball_y - 15), phase, strength=1.1)

            if charged and not can_fire and now > shooting_until and norm > fire_open_threshold:
                charged = False

            if charged and (now - last_charge_time) > charge_timeout:
                charged = False
                can_fire = False

            prev_norm = norm

        else:
            charge_start_time = 0.0
            prev_norm = None
            if charged and (now - last_charge_time) > charge_timeout:
                charged = False
                can_fire = False

        if charged and not prev_charged:
            play_charge_sfx()

        if shooting and not prev_shooting:
            play_fire_sfx()
            fire_flash_end = now + 0.16
            shake_end = now + 0.22

        # Blue-white color grade ramps up when charged, stronger on fire.
        grade_strength = 0.0
        if charged:
            grade_strength = 0.42
        if shooting:
            grade_strength = 0.85
        apply_blue_white_grading(frame, grade_strength)

        # White flash burst right when firing.
        if now < fire_flash_end:
            flash = np.full_like(frame, 255, dtype=np.uint8)
            alpha = clamp((fire_flash_end - now) / 0.16, 0.0, 1.0) * 0.55
            frame[:] = cv2.addWeighted(frame, 1.0 - alpha, flash, alpha, 0)

        # Camera shake while the shot starts.
        if now < shake_end:
            strength = int(14 * clamp((shake_end - now) / 0.22, 0.0, 1.0))
            dx = int(strength * math.sin(phase * 3.7))
            dy = int(strength * math.cos(phase * 4.9))
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(
                frame,
                m,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

        # Subtle RGB split during fire burst for anime-style impact.
        if shooting or now < fire_flash_end:
            aberration = 2 if shooting else 1
            frame = apply_chromatic_aberration(frame, aberration)

        prev_charged = charged
        prev_shooting = shooting
        draw_hud(frame, charged, shooting)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
