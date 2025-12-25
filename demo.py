import os
import sys
import time
import math
import collections
import traceback

import cv2
import numpy as np
from PIL import Image, ImageSequence
from dotenv import load_dotenv

load_dotenv()
from config import Config

# ----------------- Utilities / GIF loader / Emote player -----------------
def pil_to_cv2_rgba(pil_img):
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    arr = np.array(pil_img)
    arr = arr[:, :, [2, 1, 0, 3]]  # RGBA -> BGRA
    return arr

def load_animated_emote(path, size=None, max_frames=None):
    frames = []
    durations = []
    if not os.path.exists(path):
        return frames, durations
    try:
        im = Image.open(path)
    except Exception:
        return frames, durations
    i = 0
    for frame in ImageSequence.Iterator(im):
        if max_frames and i >= max_frames:
            break
        f = frame.convert("RGBA")
        if size:
            f = f.resize(size, Image.BILINEAR)
        frames.append(pil_to_cv2_rgba(f))
        dur_ms = frame.info.get("duration", im.info.get("duration", 100))
        durations.append(max(1, dur_ms) / 1000.0)
        i += 1
    return frames, durations

class EmotePlayer:
    def __init__(self, frames, durations, loop=True):
        self.frames = frames or []
        self.durations = durations or []
        self.loop = loop
        self.reset()

    def reset(self):
        self.idx = 0
        self.last_t = time.perf_counter()
        self.finished = False

    def update(self):
        if not self.frames or self.finished:
            return
        now = time.perf_counter()
        dur = self.durations[self.idx]
        if now - self.last_t >= dur:
            step = int((now - self.last_t) / dur)
            self.idx += max(1, step)
            self.last_t = now
            if self.idx >= len(self.frames):
                if self.loop:
                    self.idx %= len(self.frames)
                else:
                    self.idx = len(self.frames) - 1
                    self.finished = True

    def current_frame(self):
        return self.frames[self.idx] if self.frames else None

def overlay_rgba(bg, fg_rgba, x, y, scale=1.0):
    if fg_rgba is None:
        return bg
    h, w = fg_rgba.shape[:2]
    fh, fw = int(h * scale), int(w * scale)
    if fh <= 0 or fw <= 0:
        return bg
    fg = cv2.resize(fg_rgba, (fw, fh), interpolation=cv2.INTER_AREA)
    alpha = fg[:, :, 3] / 255.0
    y1, y2 = max(0, y), min(bg.shape[0], y + fh)
    x1, x2 = max(0, x), min(bg.shape[1], x + fw)
    if y1 >= y2 or x1 >= x2:
        return bg
    bg_region = bg[y1:y2, x1:x2].astype(float)
    fg_region = fg[(y1 - y):(y2 - y), (x1 - x):(x2 - x)].astype(float)
    a = alpha[(y1 - y):(y2 - y), (x1 - x):(x2 - x)]
    for c in range(3):
        bg[y1:y2, x1:x2, c] = (1.0 - a) * bg_region[:, :, c] + a * fg_region[:, :, c]
    return bg

# ----------------- MediaPipe setup -----------------
try:
    import mediapipe as mp
except Exception as e:
    print("ERROR: mediapipe import failed:", e)
    raise

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detector = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ----------------- Load emote assets -----------------
EMOTE_SIZE = (220, 220)
asset_dir = getattr(Config, "EMOTE_ASSETS", "src/assets/emotes")

emote_files = {
    "wizard": os.path.join(asset_dir, "wizard.gif"),
    "golden_ratio": os.path.join(asset_dir, "golden_ratio.gif"),
    "thumbs_up": os.path.join(asset_dir, "thumbs_up.gif"),
    "goblin_crying": os.path.join(asset_dir, "goblin_crying.gif"),
}

EMOTE_PLAYERS = {}
for k, p in emote_files.items():
    frames, durations = load_animated_emote(p, size=EMOTE_SIZE)
    if frames:
        EMOTE_PLAYERS[k] = EmotePlayer(frames, durations, loop=True)
        EMOTE_PLAYERS[k].reset()
    else:
        EMOTE_PLAYERS[k] = None
        print(f"[WARN] Emote missing or failed to load: {p}")

# ----------------- Gesture detection helpers & state -----------------
BUFFER_LEN = 12
left_wx = collections.deque(maxlen=BUFFER_LEN)
right_wx = collections.deque(maxlen=BUFFER_LEN)
gesture_history = collections.deque(maxlen=9)

TH_NEAR_FACE = 0.10
TH_JUGGLE_VEL = 0.012
TH_GOBLIN_FACEFRAC = 0.35
TH_FIST_FACTOR = 0.8

FACE_CHIN = 152
FACE_LEFT_EYE = 33
FACE_RIGHT_EYE = 263

IDX = {
    "wrist": 0, "thumb_tip": 4, "index_mcp": 5, "index_tip": 8,
    "middle_mcp": 9, "middle_tip": 12, "ring_mcp": 13, "ring_tip": 16,
    "pinky_mcp": 17, "pinky_tip": 20
}

def norm_dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def hand_size(hand_lm):
    try:
        w = hand_lm[IDX["wrist"]]
        m = hand_lm[IDX["middle_mcp"]]
        return norm_dist((w.x, w.y), (m.x, m.y))
    except Exception:
        return 0.1

def finger_extended(hand_lm, tip_idx, mcp_idx):
    try:
        wrist = (hand_lm[IDX["wrist"]].x, hand_lm[IDX["wrist"]].y)
        tip = (hand_lm[tip_idx].x, hand_lm[tip_idx].y)
        mcp = (hand_lm[mcp_idx].x, hand_lm[mcp_idx].y)
        d_tip = norm_dist(tip, wrist)
        d_mcp = norm_dist(mcp, wrist)
        return d_tip > d_mcp * 1.15
    except Exception:
        return False

def thumb_up_pose(hand_lm):
    try:
        is_thumb_ext = finger_extended(hand_lm, IDX["thumb_tip"], IDX["index_mcp"])
        idx_ext = finger_extended(hand_lm, IDX["index_tip"], IDX["index_mcp"])
        mid_ext = finger_extended(hand_lm, IDX["middle_tip"], IDX["middle_mcp"])
        ring_ext = finger_extended(hand_lm, IDX["ring_tip"], IDX["ring_mcp"])
        pinky_ext = finger_extended(hand_lm, IDX["pinky_tip"], IDX["pinky_mcp"])
        others_folded = (not idx_ext) and (not mid_ext) and (not ring_ext) and (not pinky_ext)
        return is_thumb_ext and others_folded
    except Exception:
        return False

# Initialize last_goblin_metrics safely
last_goblin_metrics = {"d_left": None, "d_right": None, "lf": None, "rf": None, "face_w": None, "prox_thr": None}

def detect_goblin_by_fists(face_landmarks, hand_landmarks_list):
    """
    Return (detected_bool, metrics_dict)
    metrics_dict contains keys: d_left, d_right, lf, rf, face_w, prox_thr
    """
    metrics = {"d_left": None, "d_right": None, "lf": None, "rf": None, "face_w": None, "prox_thr": None}
    if face_landmarks is None or len(hand_landmarks_list) < 2:
        return False, metrics

    try:
        left_eye = face_landmarks.landmark[FACE_LEFT_EYE]
        right_eye = face_landmarks.landmark[FACE_RIGHT_EYE]
        left_eye_xy = (left_eye.x, left_eye.y)
        right_eye_xy = (right_eye.x, right_eye.y)
    except Exception:
        return False, metrics

    try:
        xs = [p.x for p in face_landmarks.landmark]
        face_width = max(xs) - min(xs) if xs else 0.2
        face_width = max(face_width, 0.12)
    except Exception:
        face_width = 0.2
    metrics["face_w"] = face_width

    hands = []
    for hl in hand_landmarks_list:
        lm = hl.landmark
        wrist = (lm[IDX["wrist"]].x, lm[IDX["wrist"]].y)
        try:
            mcp_idxs = [IDX["index_mcp"], IDX["middle_mcp"], IDX["ring_mcp"]]
            px = float(np.mean([lm[i].x for i in mcp_idxs]))
            py = float(np.mean([lm[i].y for i in mcp_idxs]))
            palm_center = (px, py)
        except Exception:
            palm_center = wrist

        tip_idxs = [IDX["thumb_tip"], IDX["index_tip"], IDX["middle_tip"], IDX["ring_tip"], IDX["pinky_tip"]]
        tip_dists = []
        for ti in tip_idxs:
            try:
                tip = (lm[ti].x, lm[ti].y)
                tip_dists.append(norm_dist(tip, wrist))
            except Exception:
                tip_dists.append(1.0)
        avg_tip_to_wrist = float(np.mean(tip_dists))
        hsize = hand_size(lm)
        fistness = avg_tip_to_wrist / max(hsize, 1e-6)

        hands.append({
            "lm": lm,
            "wrist": wrist,
            "palm": palm_center,
            "avg_tip_to_wrist": avg_tip_to_wrist,
            "hsize": hsize,
            "fistness": fistness
        })

    if len(hands) < 2:
        return False, metrics

    sorted_h = sorted(hands, key=lambda x: x["wrist"][0])
    left_hand = sorted_h[0]
    right_hand = sorted_h[1]

    d_left_palm = norm_dist(left_hand["palm"], left_eye_xy)
    d_right_palm = norm_dist(right_hand["palm"], right_eye_xy)
    d_left_wrist = norm_dist(left_hand["wrist"], left_eye_xy)
    d_right_wrist = norm_dist(right_hand["wrist"], right_eye_xy)

    metrics["d_left"] = float((d_left_palm + d_left_wrist) / 2.0)
    metrics["d_right"] = float((d_right_palm + d_right_wrist) / 2.0)
    metrics["lf"] = float(left_hand["fistness"])
    metrics["rf"] = float(right_hand["fistness"])

    prox_thr = max(TH_GOBLIN_FACEFRAC * face_width, 0.04)
    metrics["prox_thr"] = prox_thr

    left_prox_ok = (d_left_palm < prox_thr) or (d_left_wrist < prox_thr * 0.9)
    right_prox_ok = (d_right_palm < prox_thr) or (d_right_wrist < prox_thr * 0.9)

    left_fist_ok = (left_hand["fistness"] < TH_FIST_FACTOR)
    right_fist_ok = (right_hand["fistness"] < TH_FIST_FACTOR)

    detected = left_prox_ok and right_prox_ok and left_fist_ok and right_fist_ok
    if detected:
        print(f"[DEBUG] Goblin detected: dL={metrics['d_left']:.3f}, dR={metrics['d_right']:.3f}, lf={metrics['lf']:.2f}, rf={metrics['rf']:.2f}")
    return bool(detected), metrics

def single_frame_guess(face_landmarks, hand_landmarks_list):
    global last_goblin_metrics
    hands = hand_landmarks_list or []
    if not hands:
        return "neutral"

    # thumbs-up check
    for h in hands:
        try:
            if thumb_up_pose(h.landmark):
                if h.landmark[IDX["thumb_tip"]].y < h.landmark[IDX["wrist"]].y - 0.01:
                    return "thumbs_up"
        except Exception:
            pass

    # golden_ratio: index near chin
    if face_landmarks:
        try:
            chin = face_landmarks.landmark[FACE_CHIN]
            chin_xy = (chin.x, chin.y)
            xs = [p.x for p in face_landmarks.landmark]
            face_w = max(xs) - min(xs) if xs else 0.2
            for h in hands:
                idx = (h.landmark[IDX["index_tip"]].x, h.landmark[IDX["index_tip"]].y)
                if norm_dist(idx, chin_xy) < max(TH_NEAR_FACE * 0.9, face_w * 0.28):
                    return "golden_ratio"
        except Exception:
            pass

    # goblin fists detection
    try:
        ok, metrics = detect_goblin_by_fists(face_landmarks, hands)
        last_goblin_metrics = metrics
        if ok:
            return "goblin_crying"
    except Exception:
        # keep prior metrics but avoid crash
        traceback.print_exc()

    # wizard juggling detection (alternating wrists)
    if len(hands) >= 2:
        try:
            xs = [h.landmark[IDX["wrist"]].x for h in hands[:2]]
            left_wx.append(xs[0])
            right_wx.append(xs[1])
            if len(left_wx) >= 4 and len(right_wx) >= 4:
                lv = np.diff(np.array(left_wx))
                rv = np.diff(np.array(right_wx))
                if (np.max(np.abs(lv)) > TH_JUGGLE_VEL and np.max(np.abs(rv)) > TH_JUGGLE_VEL):
                    if np.mean(lv[-3:] * rv[-3:]) < 0:
                        return "wizard"
        except Exception:
            pass

    return "neutral"

def stable_gesture(candidate):
    gesture_history.append(candidate)
    if len(gesture_history) < 5:
        return None
    counts = collections.Counter(gesture_history)
    most, cnt = counts.most_common(1)[0]
    if cnt / len(gesture_history) >= 0.6 and most != "neutral":
        return most
    if counts.get("neutral", 0) / len(gesture_history) >= 0.8:
        return "neutral"
    return None

# ----------------- Robust main loop with camera retry -----------------
def open_camera_with_retry(idx, retries=3, delay=0.5):
    for attempt in range(1, retries + 1):
        print(f"[camera] Opening index {idx} (attempt {attempt})")
        cap = cv2.VideoCapture(idx)
        time.sleep(0.2)
        if cap.isOpened():
            print("[camera] Opened camera.")
            return cap
        try:
            cap.release()
        except Exception:
            pass
        time.sleep(delay)
    return None

def main_loop():
    cam_idx = int(os.environ.get("CAMERA_INDEX", getattr(Config, "CAMERA_INDEX", 0)))
    cap = open_camera_with_retry(cam_idx, retries=3, delay=0.5)
    if cap is None:
        print(f"[ERROR] Could not open camera index {cam_idx}. Run cam_test.py or change CAMERA_INDEX.", flush=True)
        input("Press Enter to exit...")
        sys.exit(1)

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, getattr(Config, "FRAME_WIDTH", 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, getattr(Config, "FRAME_HEIGHT", 480))
    except Exception:
        pass

    MAIN_WIN = "EmoteCam - Camera"
    EMOTE_WIN = "EmotePanel"
    cv2.namedWindow(MAIN_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(MAIN_WIN, 900, 640)
    cv2.namedWindow(EMOTE_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(EMOTE_WIN, 300, 300)

    current_emote = "neutral"
    print("[info] Starting. Press ESC or 'q' to quit.")

    try:
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("[camera] read failed; attempting re-open...")
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = open_camera_with_retry(cam_idx, retries=2, delay=0.5)
                    if cap is None:
                        print("[camera] Cannot re-open camera; exiting loop.")
                        break
                    continue

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                hands_res = hands_detector.process(rgb)
                face_res = face_detector.process(rgb)

                hand_landmarks = hands_res.multi_hand_landmarks or []
                face_lms = face_res.multi_face_landmarks[0] if (face_res and face_res.multi_face_landmarks) else None

                candidate = single_frame_guess(face_lms, hand_landmarks)
                stable = stable_gesture(candidate)
                if stable is not None and stable != current_emote:
                    current_emote = stable
                    p = EMOTE_PLAYERS.get(current_emote)
                    if p:
                        p.reset()

                # draw face/hand landmarks for debugging
                if face_lms:
                    mp_drawing.draw_landmarks(
                        frame,
                        face_lms,
                        mp_face.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                for hl in hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing_styles.get_default_hand_landmarks_style(),
                                              mp_drawing_styles.get_default_hand_connections_style())

                # safe debug box
                dbg = [f"Gesture: {current_emote}", f"Candidate: {candidate}"]
                d_left = last_goblin_metrics.get("d_left")
                d_right = last_goblin_metrics.get("d_right")
                lf = last_goblin_metrics.get("lf")
                rf = last_goblin_metrics.get("rf")
                fw = last_goblin_metrics.get("face_w")
                thr = last_goblin_metrics.get("prox_thr")

                dbg.append(f"G.dL={d_left:.3f}" if (d_left is not None) else "G.dL=-")
                dbg.append(f"G.dR={d_right:.3f}" if (d_right is not None) else "G.dR=-")
                dbg.append(f"G.fL={lf:.2f}" if (lf is not None) else "G.fL=-")
                dbg.append(f"G.fR={rf:.2f}" if (rf is not None) else "G.fR=-")
                dbg.append(f"G.w={fw:.3f}" if (fw is not None) else "G.w=-")
                dbg.append(f"G.thr={thr:.3f}" if (thr is not None) else "G.thr=-")

                overlay = frame.copy()
                cv2.rectangle(overlay, (8, 8), (460, 12 + 20 * len(dbg)), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                for i, text in enumerate(dbg):
                    cv2.putText(frame, text, (12, 30 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                # emote panel update
                emote_panel_img = np.zeros((EMOTE_SIZE[1], EMOTE_SIZE[0], 3), dtype=np.uint8) + 30
                p = EMOTE_PLAYERS.get(current_emote)
                if p:
                    p.update()
                    ef = p.current_frame()
                    if ef is not None:
                        ph, pw = emote_panel_img.shape[:2]
                        ef_h, ef_w = ef.shape[:2]
                        scale = min(ph / ef_h, pw / ef_w, 1.0)
                        pos_x = int((pw - int(ef_w * scale)) / 2)
                        pos_y = int((ph - int(ef_h * scale)) / 2)
                        emote_panel_img = overlay_rgba(emote_panel_img, ef, pos_x, pos_y, scale=scale)
                else:
                    cv2.putText(emote_panel_img, "No emote file", (10, EMOTE_SIZE[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

                cv2.imshow(MAIN_WIN, frame)
                cv2.imshow(EMOTE_WIN, emote_panel_img)

                k = cv2.waitKey(1) & 0xFF
                if k == 27 or k == ord('q'):
                    print("[info] Exit key pressed.")
                    break

                if cv2.getWindowProperty(MAIN_WIN, cv2.WND_PROP_VISIBLE) < 1 or cv2.getWindowProperty(EMOTE_WIN, cv2.WND_PROP_VISIBLE) < 1:
                    print("[info] One of windows closed by user, exiting.")
                    break

            except Exception as ex:
                print("[ERROR] Exception during loop:", ex)
                traceback.print_exc()
                print("Pausing briefly then continuing...")
                time.sleep(1.0)
                continue

    finally:
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            hands_detector.close()
            face_detector.close()
        except Exception:
            pass
        print("[info] Clean exit.")

if __name__ == "__main__":
    main_loop()
