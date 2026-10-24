import cv2
import base64
import requests
import json
import os
import time
import threading
from datetime import datetime
from flask import Flask, Response, render_template, render_template_string, send_from_directory, jsonify, request
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import numpy as np
import glob
import socket
import subprocess
import re

# =====================================================================
# AI VEHICLE RECOGNITION — HOW THIS FILE IS ORGANISED
# =====================================================================
# Reading top to bottom, this file has these major parts:
#   1. SETUP           — imports, config constants (video/model paths,
#                        Ollama settings, MAVLink port/baud)
#   2. LOGGING          — one session log file (logs/session_<time>.txt)
#                        that every part of the app writes into
#   3. VEHICLE TRACKING — keeps each detected vehicle's identity stable
#                        across frames (so "Car01" stays "Car01")
#   4. AI DESCRIPTION   — sends cropped vehicle images to Ollama (a local
#                        vision-language model) to get a text description
#                        and a more accurate colour guess than YOLO alone
#   5. MAVLINK / FLIGHT — talks to the Pixhawk: reads live GPS/heading/
#                        flight mode, and sends GOTO / LAND / SET_MODE
#                        commands
#   6. VIDEO LOOP       — the background thread that continuously reads
#                        frames (video file or live camera), runs YOLO,
#                        updates tracking, and hands the frame off to be
#                        streamed to the browser
#   7. FLASK ROUTES     — the web server: serves the page, streams video,
#                        and exposes the /api/... endpoints every button
#                        in the browser calls
# =====================================================================

# ─────────────────────────────────────────────
# Optional MAVLink
# ─────────────────────────────────────────────
# Tries to import pymavlink (the library used to talk to the Pixhawk).
# If it's not installed, the whole app still runs — every MAVLink-related
# function below falls back to "simulation mode" instead of crashing.
try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False
    print("[WARN] pymavlink not installed - running in simulation mode.")

# ─────────────────────────────────────────────
# Configuration — the constants you're most likely to need to change
# ─────────────────────────────────────────────
VIDEO_DIR    = "videos"
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_URL_CHAT = "http://localhost:11434/api/chat"
OLLAMA_MODEL    = "minicpm-v"   # Change to switch: "llava-phi3", "llava:13b"

# TEMPORARY DIAGNOSTIC: set very high to effectively disable Ollama calls,
# to test whether they're the cause of the Jetson lag (they run on the same
# shared GPU/RAM as YOLO inference, every OLLAMA_INTERVAL_SEC seconds).
# Once confirmed, put back to a normal value like 15-30 (not 5) so it
# still runs periodically, just far less often, to reduce GPU/RAM contention.
OLLAMA_INTERVAL_SEC = 25
STATUS_PRINT_INTERVAL_SEC = 10   # how often the [STATUS] FPS/GPU/CPU line prints — was every 1s, now every 10s to cut terminal clutter
MAVLINK_PORT = "/dev/ttyTHS1"   # Jetson GPIO UART. Use "/dev/ttyACM0" if Pixhawk is on USB instead.
MAVLINK_BAUD = 921600            # MUST match the Pixhawk's SERIALx_BAUD param for this port
# Used only when there's no live GPS telemetry yet (sim mode / no fix) so the
# GPS-based goto pipeline still has somewhere to project distance/bearing from.
SIM_HOME_GPS = {"lat": 3.139003, "lon": 101.686855}
IMAGE_DIR   = "image"
SUMMARY_DIR = "Summary"
LOG_DIR     = "logs"

import torch
DEVICE   = 0 if torch.cuda.is_available() else "cpu"   # GPU index for CUDA, else CPU
print(f"[Device Check] torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"[Device Check] MODEL will run on: {'GPU (CUDA device ' + str(DEVICE) + ')' if DEVICE != 'cpu' else 'CPU (no GPU detected!)'}")
print(f"[Init] Using device: {DEVICE}")

MODEL    = YOLO("yolov8n.engine", task="detect")   # switched from yolov8m for real-time FPS on Jetson
MIN_CONF = 0.15

# ── CLAHE for distant-vehicle contrast enhancement ──
# Applied before every YOLO call to help detect low-contrast aerial vehicles
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

def enhance_frame(frame):
    """CLAHE on L channel only — filter2D removed (saved ~8ms/call)."""
    lab        = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b    = cv2.split(lab)
    l_enhanced = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Confidence thresholds — TWO tiers based on box size:
#   NEAR = large box (close vehicle)  → strict, avoids road/structure FP
#   FAR  = small box (distant vehicle) → lenient, allows low-conf small detections
CLASS_CONF_NEAR = {2: 0.40, 3: 0.45, 5: 0.65, 7: 0.55}
CLASS_CONF_FAR  = {2: 0.22, 3: 0.28, 5: 0.55, 7: 0.30}
# A box is "far" if it occupies less than 1.2% of the frame area
FAR_VEHICLE_PCT = 0.012

def _passes_conf(cls_id, conf, pct):
    """Use lenient threshold for small/distant vehicles."""
    if pct < FAR_VEHICLE_PCT:
        return conf >= CLASS_CONF_FAR.get(cls_id, 0.22)
    return conf >= CLASS_CONF_NEAR.get(cls_id, 0.40)

_frame_lock       = threading.Lock()   # protects current_frame across threads
_latest_annotated = None               # latest drawn frame for stream thread
_latest_lock      = threading.Lock()   # protects _latest_annotated
_inf_thread       = None               # inference background thread
_inf_stop         = threading.Event()  # signals inference thread to stop

BOX_COLORS = {
    "car":        (0, 200, 255),
    "motorcycle": (255, 100,   0),
    "bus":        (0, 255, 100),
    "truck":      (60,  60, 255),
    "van":        (200,   0, 200),
}

os.makedirs(IMAGE_DIR,   exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(LOG_DIR,     exist_ok=True)

# ─────────────────────────────────────────────
# Session logging — ONE file per session, logs/session_<ts>.txt
# Every event (detections, commands, MAVLink connection, mode changes,
# errors) is written as one line, tagged by subsystem and marked
# OK/FAIL/INFO, so you can open the file and immediately see what
# happened and whether each action actually succeeded.
# Format:  [timestamp] [TAG] STATUS | detail
# ─────────────────────────────────────────────
_session_start = datetime.now()
_session_ts    = _session_start.strftime("%Y%m%d_%H%M%S")
_log_path      = os.path.join(LOG_DIR, f"session_{_session_ts}.txt")
_log_lock      = threading.Lock()   # prevents concurrent writes corrupting the file

with open(_log_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("AI Vehicle Recognition — Session Log\n")
    f.write(f"Session started : {_session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("Format: [timestamp] [TAG] STATUS | detail\n")
    f.write("=" * 70 + "\n\n")

def _write_log(tag, status, detail=""):
    """The one function everything else funnels through to write a line."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{ts}] [{tag:<10}] {status:<7} | {detail}\n"
    with _log_lock:
        with open(_log_path, 'a', encoding='utf-8') as f:
            f.write(line)

def print_result(tag, action_text, success, detail=""):
    """
    The one function every button-triggered action calls to report what
    happened, in plain English, both to the terminal AND the session log —
    e.g. "Pixhawk is connected", "Flight mode changed to STABILIZE",
    "Go to vehicle Car01". Always makes success/failure immediately obvious.
    """
    icon = "✅" if success else "❌"
    full_text = action_text + (f" — {detail}" if detail else "")
    print(f"[{tag}] {icon} {full_text}")
    _write_log(tag, "OK" if success else "FAIL", full_text)
    return full_text

def log_detection(video_source, frame_number, tracked_vehicles_list):
    """One line summarizing all currently-visible vehicles for this frame check-in."""
    if not tracked_vehicles_list:
        return
    summary = ", ".join(
        f"{t.get('id','?')}[{t.get('type','?')}/{t.get('colour','unknown')} "
        f"conf={t.get('conf',0):.2f}]"
        for t in tracked_vehicles_list
    )
    _write_log("DETECTION", "OK",
               f"frame={frame_number} src={os.path.basename(video_source or 'unknown')} "
               f"count={len(tracked_vehicles_list)} vehicles=[{summary}]")

def log_command(command_type, vehicle_id="", north=0.0, east=0.0,
                altitude=0.0, mode="SIM", detail="", success=True):
    """
    Log a drone command attempt. success=True/False makes it immediately
    obvious in the log whether the command to the Pixhawk actually went
    through, or failed — pass success=False from the caller's except block.
    """
    status = "OK" if success else "FAIL"
    _write_log(command_type, status,
               f"vehicle={vehicle_id or '-'}  N={north:.2f}m  E={east:.2f}m  "
               f"ALT={altitude:.1f}m  mode={mode}  {detail}")

def log_session_event(event_text, status="INFO"):
    """Free-text event — video source changes, MAVLink connect/disconnect, etc."""
    _write_log("SESSION", status, event_text)

def close_session_log():
    """Write session footer when app shuts down."""
    end_time = datetime.now()
    duration = end_time - _session_start
    with _log_lock:
        with open(_log_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"Session ended   : {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total duration  : {str(duration).split('.')[0]}\n")
            f.write("=" * 70 + "\n")

import atexit
atexit.register(close_session_log)

print(f"[Log] Session log : {_log_path}")


# ─────────────────────────────────────────────
# Size-based vehicle type classifier
# ─────────────────────────────────────────────
def classify_by_size(cls_id, box_area, frame_area, bbox=None):
    """
    Refine YOLO vehicle type using bounding box size and shape.

    Key rules:
    - Bus ONLY from YOLO cls 5 at high conf — never by size alone.
      A large bounding box from above is almost always a truck or van,
      not a bus. Road dividers are wide but not bus-shaped.
    - Motorcycle ONLY from YOLO cls 3 — small size alone is not enough.
    - Truck/van promoted from car (cls 2) only if box is clearly large
      AND roughly rectangular (width > height from aerial view).
    - cls 7 (truck) downgrades to car if box is small (distant vehicle).
    """
    pct = (box_area / frame_area) * 100

    # ── Aspect ratio hint (aerial: vehicles wider than tall) ──────────
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        aspect = bw / bh   # > 1 = wider than tall (typical vehicle from above)
    else:
        aspect = 1.5       # assume reasonable default

    # Motorcycle — ONLY trust YOLO direct classification
    if cls_id == 3:
        return "motorcycle"

    # Bus — ONLY trust YOLO direct classification at high confidence
    # (confidence already enforced by CLASS_CONF; never promote by size)
    if cls_id == 5:
        return "bus"

    # Car (cls 2) — size + aspect ratio to distinguish car / van / truck
    #
    # IMPORTANT LIMITATION: box size as a % of the frame conflates two
    # different things — "this vehicle is physically larger" and "this
    # vehicle is just closer to the camera/flying lower." A perfectly
    # normal car close to a low-flying drone will occupy a large % of
    # the frame for the same reason an actual truck would, so these
    # thresholds were previously too easy to trigger on a close car.
    # Raised (and made the van aspect-ratio check stricter) so it now
    # takes a noticeably larger AND more elongated box before a car
    # gets reclassified — reduces false van/truck promotions, at the
    # cost of needing a genuinely bigger box to correctly promote a
    # real van/truck. If this still misfires at very low altitudes,
    # the more robust fix is normalizing box size using the drone's
    # actual altitude (available via get_drone_state()) rather than
    # raw frame percentage — let me know if you want that added.
    if cls_id == 2:
        if pct < 4.0:
            return "car"          # small/medium → regular car
        elif pct < 9.0 and aspect > 1.4:
            return "van"          # large AND clearly elongated → van
        elif pct < 16.0:
            return "truck"        # very large → lorry/truck
        else:
            return "truck"        # implausibly large still truck not bus

    # Truck (cls 7) — small detections are likely distant cars
    if cls_id == 7:
        if pct < 1.5:
            return "car"     # small distant detection → car
        elif pct < 4.0:
            return "van"
        elif pct < 8.0:
            return "truck"   # large → confirmed truck/lorry
        else:
            return "car"     # implausibly large → likely a road/structure false positive

    return "car"

# ─────────────────────────────────────────────
# Startup cleanup — runs once when the app launches, to clear out old
# summary files and (in a background thread) periodically delete old
# saved vehicle-crop images every hour, so the disk doesn't slowly fill
# up with images from past sessions.
# ─────────────────────────────────────────────
def clear_directories():
    for directory in [SUMMARY_DIR]:
        for file_path in glob.glob(os.path.join(directory, "*")):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def clear_image_folder():
    while True:
        try:
            for file_path in glob.glob(os.path.join(IMAGE_DIR, "*.jpg")):
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Permanently deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting files in {IMAGE_DIR}: {e}")
        time.sleep(3600)

threading.Thread(target=clear_image_folder, daemon=True).start()
clear_directories()

# ─────────────────────────────────────────────
# Flask / SocketIO — the web server itself. Flask serves the page and
# the /api/... endpoints (see the FLASK ROUTES section near the bottom
# of this file); SocketIO is the live "push" channel used for things
# like detection updates and flight-mode changes that need to reach the
# browser immediately, without it having to ask/refresh.
# ─────────────────────────────────────────────
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'vehicle-ai-secret'
socketio = SocketIO(app)

# ─────────────────────────────────────────────
# Global state — shared values the video loop writes to and the Flask
# routes read from (e.g. so the browser can ask "how many vehicles right
# now?" at any time without waiting for the next video frame).
# ─────────────────────────────────────────────
vehicle_count         = 0
last_vehicle_count    = 0
description_generated = False
video_source_changed  = False
VIDEO_SOURCE          = None

type_counts = {
    "car": 0, "van": 0,
    "motorcycle": 0, "lorry": 0, "bus": 0,
}

# Store latest detections + frame for Ollama crop analysis
current_detections = []   # list of {type, bbox, conf}
current_frame      = None # latest annotated frame

# ─────────────────────────────────────────────
# Persistent vehicle ID tracker
# IDs are per type: Car01, Car02, Van01, etc.
# WHY this exists: YOLO alone only tells you "there's a car at this box,
# in this single frame" — it has no concept of it being the SAME car as
# the one it saw a moment ago. This tracker matches each new detection
# to an existing tracked vehicle (by position/overlap between frames) so
# a car keeps the same ID ("Car01") across the whole time it's visible,
# rather than getting a new random identity every frame. That stable ID
# is what the "Go to vehicle" button and the vehicle list depend on.
# ─────────────────────────────────────────────
tracked_vehicles  = {}   # key -> track dict
type_id_counters  = {}   # vtype -> next number (e.g. "car" -> 3)
TRACK_IOU_THRESH  = 0.05
TRACK_MAX_LOST    = 90
CONF_WINDOW       = 15
CONF_SHOW_THRESH  = 0.15  # reasonable threshold — tracker smoothing handles noise
CONF_HIDE_THRESH  = 0.07

# ── Engagement (continuous follow) ────────────
# Works for both static and moving vehicles: instead of sending one GOTO
# command and forgetting about it, this loop keeps re-reading the target's
# LIVE tracked position and keeps re-sending updated GOTO commands.
# A static vehicle just gets (almost) the same point resent each cycle —
# harmless. A moving vehicle gets a fresh, up-to-date point each cycle.
ENGAGE_INTERVAL_SEC = 1.5   # how often to recompute + resend the target GPS
ENGAGE_LOST_TIMEOUT = 5.0   # seconds of no tracker update before we give up

# HARD SAFETY FLOOR — the drone will NEVER be commanded below this altitude,
# no matter what altitude value a request asks for. Commanding a drone all
# the way to 0m (ground/vehicle-roof height) at the exact GPS point of a
# vehicle — especially a MOVING one that the loop keeps re-aiming at — is a
# direct collision course. This floor keeps "descend toward the vehicle"
# from ever becoming "fly into the vehicle." Tune this to your airframe's
# realistic safe hover clearance, but do not remove it.
MIN_APPROACH_ALT_M = 2.0

engagement_state = {
    "active":  False,
    "vid":     None,
    "thread":  None,
    "stop":    None,   # threading.Event
}
engagement_lock = threading.Lock()

# ── Confidence smoothing ──────────────────────
# Rolling window size: how many frames to average confidence over

# Per-track confidence history: key -> deque of recent raw confs
conf_history      = {}

# Sol 3: UI freeze
UI_STABLE_FRAMES  = 12
ui_visible_ids    = set()
ui_last_push      = -999

# Sol 4: Minimum lifespan before appearing in UI
TRACK_MIN_LIFE    = 2     # appear after just 2 matched frames

def iou_box(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter   = max(0,ix2-ix1) * max(0,iy2-iy1)
    if inter == 0: return 0.0
    ua = (ax2-ax1)*(ay2-ay1)
    ub = (bx2-bx1)*(by2-by1)
    return inter / (ua+ub-inter)

def centre_dist(a, b):
    """Euclidean distance between centres of two boxes."""
    ax = (a[0] + a[2]) / 2; ay = (a[1] + a[3]) / 2
    bx = (b[0] + b[2]) / 2; by = (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

def make_vid(vtype, num):
    """Format ID as e.g. CAR01, VAN02, MOTO03."""
    prefix = {
        "car":        "CAR",
        "van":        "VAN",
        "motorcycle": "MOTO",
        "truck":      "TRK",
        "bus":        "BUS",
    }.get(vtype, vtype[:3].upper())
    return f"{prefix}{num:02d}"

def smooth_conf(key, raw_conf):
    """
    Push raw_conf into a rolling window for this track key,
    return the smoothed (averaged) confidence.
    """
    from collections import deque
    if key not in conf_history:
        conf_history[key] = deque(maxlen=CONF_WINDOW)
    conf_history[key].append(raw_conf)
    return sum(conf_history[key]) / len(conf_history[key])

def update_tracker(new_dets, frame_id):
    """
    Match detections to existing tracks using IoU + centre-distance fallback.
    Applies rolling-average confidence smoothing per track.
    Assigns per-type IDs: CAR01, CAR02, VAN01, etc.
    """
    matched_keys = set()
    new_tracked  = {}

    for det in new_dets:
        bbox   = det["bbox"]
        vtype  = det["type"]
        best_key  = None
        best_iou  = TRACK_IOU_THRESH
        CENTRE_DIST_MAX = 120  # wider search radius for fast-moving vehicles
        best_dist = CENTRE_DIST_MAX

        # Pass 1 — IoU match (same type)
        for key, trk in tracked_vehicles.items():
            if key in matched_keys:
                continue
            if trk["type"] != vtype:
                continue
            score = iou_box(bbox, trk["last_bbox"])
            if score > best_iou:
                best_iou = score
                best_key = key

        # Pass 2 — centre-distance fallback (same type, IoU too low)
        if best_key is None:
            for key, trk in tracked_vehicles.items():
                if key in matched_keys:
                    continue
                if trk["type"] != vtype:
                    continue
                d = centre_dist(bbox, trk["last_bbox"])
                if d < best_dist:
                    best_dist = d
                    best_key  = key

        if best_key is not None:
            matched_keys.add(best_key)
            prev     = tracked_vehicles[best_key]
            smoothed = smooth_conf(best_key, det["conf"])
            life     = prev.get("life", 0) + 1
            new_tracked[best_key] = {
                "id":          prev["id"],          "num":    prev["num"],
                "type":        vtype,               "colour": prev.get("colour", "unknown"),
                "last_bbox":   bbox,                "last_seen": frame_id,
                "cx":          det["cx"],           "cy":     det["cy"],
                "conf":        det["conf"],         "smooth_conf": smoothed,
                "life":        life,                "visible": prev.get("visible", False),
            }
        else:
            num      = type_id_counters.get(vtype, 0) + 1
            type_id_counters[vtype] = num
            vid      = make_vid(vtype, num)
            smoothed = smooth_conf(vid, det["conf"])
            new_tracked[vid] = {
                "id": vid, "num": num, "type": vtype, "colour": "unknown",
                "last_bbox": bbox, "last_seen": frame_id,
                "cx": det["cx"], "cy": det["cy"],
                "conf": det["conf"], "smooth_conf": smoothed,
                "life": 1, "visible": False,
            }

    # Keep lost vehicles; push 0.0 into conf window so confidence decays
    for key, trk in tracked_vehicles.items():
        if key not in new_tracked:
            if frame_id - trk["last_seen"] <= TRACK_MAX_LOST:
                trk["smooth_conf"] = smooth_conf(key, 0.0)
                new_tracked[key]   = trk

    tracked_vehicles.clear()
    tracked_vehicles.update(new_tracked)

    # Hysteresis + minimum lifespan gate
    for key, trk in tracked_vehicles.items():
        sc   = trk.get("smooth_conf", trk["conf"])
        life = trk.get("life", 0)
        if not trk.get("visible", False):
            if life >= TRACK_MIN_LIFE and sc >= CONF_SHOW_THRESH:
                trk["visible"] = True
        else:
            if sc < CONF_HIDE_THRESH:
                trk["visible"] = False

    # Return ALL visible tracks — held bbox drawn during missed frames
    return [v for v in tracked_vehicles.values() if v.get("visible", False)]

# Option B — image-quality technique: collect several recent frames in a
# buffer, then later pick the least-blurry one to send to Ollama, instead
# of whatever frame happened to be current (motion blur makes colour/type
# detection worse — this trades a little delay for a clearer image).
sharp_frames_buffer = []
SHARP_BUFFER_SIZE   = 15   # collect 15 frames, pick sharpest

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def get_video_files():
    exts  = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4"]
    files = []
    for e in exts:
        files += glob.glob(os.path.join(VIDEO_DIR, e))
    return sorted(files)

def encode_image(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

# ─────────────────────────────────────────────
# Option B (continued) — image enhancement before colour analysis.
# Boosts contrast and sharpens the crop so both the HSV colour math (below)
# and Ollama get a clearer image to work from than the raw, often
# washed-out or slightly blurry vehicle crop.
# ─────────────────────────────────────────────
def enhance_image(img):
    """Boost contrast (CLAHE) and sharpen an image crop before colour
    detection — makes washed-out or slightly blurry vehicle crops easier
    to read accurately, for both the HSV method and Ollama."""
    # Convert to LAB for better contrast enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # Slight sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    return enhanced

# ─────────────────────────────────────────────
# Option C — fast, non-AI colour guess (HSV colour math).
# This runs instantly on every detection with no network call, giving a
# quick colour guess using plain image-processing math (hue/saturation/
# brightness), BEFORE Ollama is ever asked. It's used two ways below:
# as an immediate colour shown in the UI right away, and as a "hint"
# passed to Ollama so the AI's answer can be cross-checked against it.
# ─────────────────────────────────────────────
def hsv_colour(crop_bgr):
    """
    Detect vehicle colour using HSV analysis.
    Returns (colour_name, confidence_pct).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", 0

    # Resize to small patch and enhance
    patch = cv2.resize(crop_bgr, (48, 48))
    patch = enhance_image(patch)
    hsv   = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    mean_s = float(np.mean(s_ch))
    mean_v = float(np.mean(v_ch))

    # ── Achromatic: check brightness/saturation first ──
    if mean_s < 35:
        if mean_v > 200: return "white",  90
        if mean_v > 150: return "silver", 80
        if mean_v > 90:  return "grey",   75
        return "black", 85

    # ── Chromatic colours ──
    COLOR_RANGES = {
        "red":       [(0, 8),   (168, 179)],
        "orange":    [(9, 22)],
        "yellow":    [(23, 35)],
        "green":     [(36, 85)],
        "blue":      [(86, 130)],
        "dark blue": [(100, 125)],
        "purple":    [(126, 159)],
        "maroon":    [(0, 10)],
        "brown":     [(10, 20)],
    }

    best, best_pct = "unknown", 0.0
    total_px = h_ch.size

    for colour, ranges in COLOR_RANGES.items():
        mask = np.zeros_like(h_ch)
        for lo, hi in ranges:
            mask |= cv2.inRange(h_ch, lo, hi)
        sat_mask = cv2.inRange(s_ch, 40, 255)
        combined = cv2.bitwise_and(mask, sat_mask)
        pct = cv2.countNonZero(combined) / total_px
        if pct > best_pct:
            best_pct = pct
            best = colour

    # Confidence based on how dominant the colour was
    confidence = int(min(best_pct * 300, 95))

    if best_pct < 0.12:
        # Not enough colour pixels — fallback to brightness
        if mean_v > 160: return "white",  70
        if mean_v > 100: return "silver", 65
        return "dark",   60

    # Refine: dark blue vs blue based on value channel
    if best == "blue" and mean_v < 90:
        best = "dark blue"

    return best, confidence

# ─────────────────────────────────────────────
# Option A — crop each vehicle out individually, instead of sending
# Ollama the whole video frame at once. A single frame can have several
# vehicles at very different sizes/positions — cropping each one out
# (with a little padding, upscaled if it's small) lets Ollama focus on
# one vehicle at a time, which gives much more reliable type/colour
# answers than asking it to describe everything in the full frame.
# ─────────────────────────────────────────────
def crop_vehicles(frame, detections, padding=8):
    """
    Crop each detected vehicle from frame.
    Returns list of (crop_b64, vtype, hsv_colour_guess).
    """
    crops = []
    h, w  = frame.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        # Add padding around box
        x1p = max(0, x1 - padding)
        y1p = max(0, y1 - padding)
        x2p = min(w, x2 + padding)
        y2p = min(h, y2 + padding)
        crop = frame[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue
        # Upscale small crops so Ollama can see more detail
        ch, cw = crop.shape[:2]
        if cw < 80 or ch < 60:
            scale  = max(80/cw, 60/ch)
            crop   = cv2.resize(crop, (int(cw*scale), int(ch*scale)),
                                interpolation=cv2.INTER_CUBIC)
        # Enhance before encoding
        crop_enhanced = enhance_image(crop)
        _, buf = cv2.imencode('.jpg', crop_enhanced,
                              [cv2.IMWRITE_JPEG_QUALITY, 95])
        b64   = base64.b64encode(buf.tobytes()).decode('utf-8')
        # Get HSV colour guess
        colour_guess, conf_pct = hsv_colour(crop)
        crops.append({
            "b64":          b64,
            "type":         det["type"],
            "colour_guess": colour_guess,
            "conf_pct":     conf_pct,
            "bbox":         det["bbox"],
        })
    return crops

# ─────────────────────────────────────────────
# Option A+C combined — for one individually-cropped vehicle (Option A),
# ask Ollama to describe/confirm its colour, while also telling Ollama
# what the fast HSV method (Option C) already guessed, as a hint. Ollama
# can agree with the hint or override it if it sees something different —
# this cross-checking is what makes the final colour more reliable than
# either method alone.
# ─────────────────────────────────────────────
def describe_vehicle_crop(crop_info, vehicle_num, det):
    """
    Ask Ollama to confirm/correct colour for a single vehicle crop.
    Passes HSV guess as a hint (Option C).
    """
    size_hint = crop_info.get("type", "car")

    prompt = (
        "This is a close-up aerial TOP-DOWN view of a vehicle.\n"
        "Size-based analysis classifies this as: " + str(size_hint) + "\n"
        "OpenCV colour analysis suggests the colour is '"
        + str(crop_info["colour_guess"]) + "' "
        "(confidence: " + str(crop_info["conf_pct"]) + "%).\n\n"
        "Look at this image carefully and respond ONLY in this exact format:\n"
        "  Type     : (choose ONLY from: car, van, motorcycle, lorry, bus)\n"
        "  Colour   : (choose ONLY from: black, white, grey, silver, red, "
        "dark red, blue, dark blue, green, dark green, "
        "yellow, orange, brown, maroon, beige)\n"
        "  Size     : (small/medium/large)\n"
        "  Movement : (moving/stationary)\n\n"
        "TYPE RULES — follow strictly:\n"
        "- Trust the size-based classification above unless clearly wrong\n"
        "- Small bounding box = car\n"
        "- Medium box-shaped vehicle = van\n"
        "- Long rectangular shape = lorry\n"
        "- Large wide rectangle = bus\n"
        "- Very small with 2 wheels = motorcycle\n"
        "- When unsure between car and van, use size-based result\n\n"
        "COLOUR RULES:\n"
        "- NEVER write Unknown. Always pick the closest colour.\n"
        "- Dark vehicles = black. Shiny = silver or white.\n"
    )
    # Try /api/generate first (minicpm-v), fallback to /api/chat (llava)
    endpoints = [
        (OLLAMA_URL, {
            "model":  OLLAMA_MODEL,
            "prompt": prompt,
            "images": [crop_info["b64"]],
            "stream": True,
        }),
        (OLLAMA_URL_CHAT, {
            "model":    OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt,
                          "images": [crop_info["b64"]]}],
            "stream": True,
        }),
    ]
    for url, payload in endpoints:
        try:
            resp = requests.post(url, json=payload, stream=True, timeout=60)
            if resp.status_code != 200:
                continue
            result = ""
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    if "message" in data and "content" in data["message"]:
                        result += data["message"]["content"]
                    elif "response" in data:
                        result += data["response"]
                    if data.get("done"):
                        break
            if result.strip():
                return result.strip()
        except Exception as e:
            print(f"[Ollama crop] {url} error: {e}")
            continue
    return ""

# ─────────────────────────────────────────────
# Ollama image description — this whole section builds and sends the
# request that asks Ollama's vision model to describe/count/colour the
# vehicles in view, then parses its text answer back into structured data.
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Option B — sharpest-frame picker (the actual "which frame do we use"
# half of Option B — enhance_image() above was the "make it clearer"
# half). Laplacian variance is a standard blur-detection measure: a sharp,
# in-focus image has more high-frequency detail (higher variance) than a
# blurry one. Given several recent frames, this picks the least blurry.
# ─────────────────────────────────────────────
def is_sharp(frame, threshold=80.0):
    """Return True if frame is sharp enough using Laplacian variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= threshold

def get_sharpest_frame(frames):
    """Return the sharpest frame from a list using Laplacian variance."""
    best_frame, best_score = frames[0], 0
    for f in frames:
        gray  = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if score > best_score:
            best_score = score
            best_frame = f
    print(f"[Sharp] Best frame score: {best_score:.1f}")
    return best_frame

# ─────────────────────────────────────────────
# Ollama HTTP request helper — this itself is just ONE request/response.
# Option C's actual "ask multiple times, pick the best" logic lives in
# pick_best_response() below and the loop that calls this 3x in
# get_image_description() further down — this function just handles the
# mechanics of one single call (and which of Ollama's two API endpoints
# to use, depending on the model).
# ─────────────────────────────────────────────
def ask_ollama_once(base64_image, prompt):
    """
    Single Ollama request. Tries /api/chat first (llava models),
    falls back to /api/generate (minicpm-v).
    """
    # Try /api/chat first
    payload_chat = {
        "model":    OLLAMA_MODEL,
        "messages": [{
            "role":    "user",
            "content": prompt,
            "images":  [base64_image],
        }],
        "stream": True
    }
    try:
        print(f"[Ollama] Trying /api/chat with {OLLAMA_MODEL}...")
        response = requests.post(OLLAMA_URL_CHAT, json=payload_chat,
                                 stream=True, timeout=120)
        if response.status_code == 200:
            description = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    if "message" in data and "content" in data["message"]:
                        description += data["message"]["content"]
                    if data.get("done"):
                        break
            if description.strip():
                print(f"[Ollama] /api/chat success ({len(description)} chars)")
                return description.strip()
        print(f"[Ollama] /api/chat status {response.status_code} — trying /api/generate...")
    except Exception as e:
        print(f"[Ollama] /api/chat error: {e} — trying /api/generate...")

    # Fallback to /api/generate (minicpm-v uses this)
    payload_gen = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "images": [base64_image],
        "stream": True
    }
    try:
        print(f"[Ollama] Trying /api/generate with {OLLAMA_MODEL}...")
        response = requests.post(OLLAMA_URL, json=payload_gen,
                                 stream=True, timeout=120)
        response.raise_for_status()
        description = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    description += data["response"]
                if data.get("done"):
                    break
        if description.strip():
            print(f"[Ollama] /api/generate success ({len(description)} chars)")
        else:
            print("[Ollama] WARNING: Empty response from /api/generate")
        return description.strip()
    except requests.exceptions.ConnectionError:
        print("[Ollama] ERROR: Cannot connect. Is Ollama running?")
        print("[Ollama] Run: ollama run minicpm-v")
        return ""
    except Exception as e:
        print(f"[Ollama] /api/generate error: {e}")
        return ""

# ─────────────────────────────────────────────
# Option C (continued) — vision-language models can be inconsistent
# between identical requests (this is normal for AI models — some
# randomness in each answer). Asking 3 times and picking whichever
# response found the most vehicles (or, if tied, the longest/most
# detailed one) is a cheap way to reduce how often a single unlucky
# answer misses a vehicle or gives a vague description.
# ─────────────────────────────────────────────
def pick_best_response(responses):
    """
    From multiple Ollama responses, pick the one with
    the highest total vehicle count (most confident detection).
    Falls back to longest non-empty response.
    """
    best_resp  = ""
    best_count = -1
    for r in responses:
        if not r:
            continue
        # Parse total count from response
        total = 0
        for line in r.splitlines():
            if "total vehicle" in line.lower():
                digits = ''.join(filter(str.isdigit, line))
                if digits:
                    total = int(digits)
                    break
        if total > best_count:
            best_count = total
            best_resp  = r
        elif total == best_count and len(r) > len(best_resp):
            best_resp = r
    print(f"[Ollama] Best response: total={best_count}")
    return best_resp

# ─────────────────────────────────────────────
# Option D — a carefully structured prompt, PLUS telling Ollama how many
# vehicles YOLO already found (the "count hint"). Without this hint,
# Ollama sometimes reports a different vehicle count than YOLO detected;
# telling it the expected number upfront reduces that mismatch. Note:
# this build_prompt() function is only used by the whole-frame fallback
# path below (get_image_description) — the per-vehicle-crop path
# (describe_vehicle_crop, Option A+C above) builds its own separate,
# simpler prompt for a single cropped vehicle.
# ─────────────────────────────────────────────
def build_prompt(yolo_count=None):
    """
    Build an improved prompt.
    If yolo_count provided, tell Ollama exactly how many vehicles
    YOLO detected so it doesn't miss or invent any.
    """
    yolo_hint = ""
    if yolo_count and yolo_count > 0:
        yolo_hint = (
            f"IMPORTANT: The computer vision system has already detected "
            f"{yolo_count} vehicle(s) in this image using object detection. "
            f"Use this as a guide — try to find and describe all {yolo_count} vehicle(s).\n\n"
        )

    return (
        "You are an AI Vehicle Recognition System mounted on a surveillance drone.\n"
        "The camera is pointing DIRECTLY DOWNWARD — this is a TOP-DOWN aerial view.\n"
        "Vehicles will appear as small rectangular or oval shapes from above.\n\n"
        + yolo_hint +
        "INSTRUCTIONS:\n"
        "- Use ONLY these vehicle types: car, van, motorcycle, lorry, bus.\n"
        "- Use ONLY these sizes: small, medium, large.\n"
        "- Use ONLY these movements: moving, stationary.\n"
        "- Use ONLY these positions: top-left, top-center, top-right, "
        "center-left, center, center-right, bottom-left, bottom-center, bottom-right.\n\n"
        "COLOUR RULES — follow strictly:\n"
        "- NEVER write Unknown for colour. You MUST always pick the closest colour.\n"
        "- Dark vehicles that absorb light = black.\n"
        "- Very dark blue vehicles = dark blue.\n"
        "- Very dark green vehicles = dark green.\n"
        "- Shiny bright vehicles = silver or white.\n"
        "- Use ONLY these colours: black, white, grey, silver, red, blue, "
        "dark blue, green, dark green, yellow, orange, brown, maroon, beige.\n\n"
        "Respond STRICTLY in this format, nothing else:\n\n"
        "Total Vehicle Count : (number)\n"
        "Car Count : (number)\n"
        "Van Count : (number)\n"
        "Motorcycle Count : (number)\n"
        "Lorry / Truck Count : (number)\n"
        "Bus Count : (number)\n\n"
        "Vehicle Details :\n\n"
        "Vehicle 1 :\n"
        "  Type     : (type)\n"
        "  Colour   : (colour)\n"
        "  Size     : (size)\n"
        "  Position : (position)\n"
        "  Movement : (movement)\n"
        "---\n"
        "(repeat block for each vehicle)\n\n"
        "If no vehicles visible:\n"
        "Total Vehicle Count : 0\n"
        "No vehicles detected.\n"
    )

# ─────────────────────────────────────────────
# This function is the WHOLE-FRAME description path (as opposed to the
# per-vehicle-crop path, describe_vehicle_crop, above) — it takes one
# full saved frame image and asks Ollama to describe every vehicle in
# it at once. It combines three of the techniques above:
#   Option B — the frame passed in was already picked as the sharpest
#              of several recent frames (by get_sharpest_frame, elsewhere)
#   Option C — asks Ollama up to 3 times and keeps the best answer
#   Option D — the prompt includes YOLO's vehicle count as a hint
# ─────────────────────────────────────────────
def get_image_description(image_path, yolo_count=None):
    base64_image = encode_image(image_path)
    if not base64_image:
        return ""

    prompt = build_prompt(yolo_count)

    print(f"[Ollama] Sending to {OLLAMA_MODEL} (yolo_count={yolo_count})...")

    # Ask up to 3 times (Option C) — stop early if a response already
    # looks complete, to avoid unnecessary extra Ollama calls.
    responses = []
    for attempt in range(3):
        print(f"[Ollama] Attempt {attempt + 1}/3...")
        resp = ask_ollama_once(base64_image, prompt)
        if resp:
            responses.append(resp)
            # If first response looks good (has vehicle details), use it
            if "Vehicle Details" in resp and "Type" in resp:
                print(f"[Ollama] Good response on attempt {attempt + 1}, skipping remaining")
                break

    if not responses:
        print("[Ollama] WARNING: All attempts returned empty response")
        return ""

    best = pick_best_response(responses)
    print(f"[Ollama] Final response ({len(best)} chars):\n{best}")
    return best



# ─────────────────────────────────────────────
# Ollama's answer is plain text, not structured data — this function reads
# through it line-by-line looking for lines like "Car Count : 3" and pulls
# the numbers out, so the vehicle-count numbers shown in the browser
# (and sent via the vehicle_count_update socket event) come from what
# Ollama actually said, not just YOLO's raw detection count.
# ─────────────────────────────────────────────
def parse_counts(description):
    global vehicle_count, type_counts
    counts = {"car": 0, "van": 0, "motorcycle": 0, "lorry": 0, "bus": 0}
    total  = 0

    for line in description.splitlines():
        ll = line.lower()
        try:
            digits = ''.join(filter(str.isdigit, line))
            if not digits:
                continue
            num = int(digits)
            if "total vehicle" in ll:
                total = num
            elif "car" in ll and "count" in ll:
                counts["car"] = num
            elif "van" in ll and "count" in ll:
                counts["van"] = num
            elif "motorcycle" in ll and "count" in ll:
                counts["motorcycle"] = num
            elif ("lorry" in ll or "truck" in ll) and "count" in ll:
                counts["lorry"] = num
            elif "bus" in ll and "count" in ll:
                counts["bus"] = num
        except:
            continue

    if total == 0:
        total = sum(counts.values())

    vehicle_count = total
    type_counts   = counts

    socketio.emit('vehicle_count_update', {
        'vehicle_count': total,
        'car':           counts["car"],
        'van':           counts["van"],
        'motorcycle':    counts["motorcycle"],
        'lorry':         counts["lorry"],
        'bus':           counts["bus"],
    })
    print(f"[Count] Total={total}  {counts}")

# ─────────────────────────────────────────────
# Ollama's exact wording/formatting varies between calls even with the
# same prompt (normal for AI text output) — this function normalizes
# whatever it said into one consistent, predictable text layout, so the
# summary panel in the browser always looks the same regardless of how
# minicpm-v happened to phrase that particular answer.
# ─────────────────────────────────────────────
def reformat_description(description, live_counts=None, dets=None):
    """
    Parse raw Ollama response and rebuild it into
    one strict fixed template regardless of how
    minicpm-v decides to format its output.
    """
    lines = description.splitlines()

    # ── Parse counts ─────────────────────────────
    counts = {
        "total": 0, "car": 0, "van": 0,
        "motorcycle": 0, "lorry": 0, "bus": 0
    }
    for line in lines:
        ll = line.lower()
        digits = ''.join(filter(str.isdigit, line))
        if not digits:
            continue
        num = int(digits)
        if "total vehicle" in ll:        counts["total"]      = num
        elif "car" in ll and "count" in ll:  counts["car"]    = num
        elif "van" in ll and "count" in ll:  counts["van"]    = num
        elif "motorcycle" in ll and "count" in ll: counts["motorcycle"] = num
        elif ("lorry" in ll or "truck" in ll) and "count" in ll: counts["lorry"] = num
        elif "bus" in ll and "count" in ll:  counts["bus"]    = num

    if counts["total"] == 0:
        counts["total"] = sum(v for k,v in counts.items() if k != "total")

    # Override with live YOLO counts if available (no lag)
    if live_counts:
        counts["car"]        = live_counts.get("car", counts["car"])
        counts["van"]        = live_counts.get("van", counts["van"])
        counts["motorcycle"] = live_counts.get("motorcycle", counts["motorcycle"])
        counts["lorry"]      = live_counts.get("truck", live_counts.get("lorry", counts["lorry"]))
        counts["bus"]        = live_counts.get("bus", counts["bus"])
        counts["total"]      = sum(v for k,v in counts.items() if k != "total")


    # ── Parse individual vehicle detail blocks ────
    # Extract: type, colour, size, position, movement
    vehicles = []
    current = {}

    for line in lines:
        ll  = line.lower().strip()
        val = line.split(":", 1)[-1].strip() if ":" in line else ""

        # Clean up val — remove parentheses, extra notes
        # Keep only the first meaningful word/phrase before any bracket
        if "(" in val:
            val = val.split("(")[0].strip()
        # Trim long descriptive text after common delimiters
        for delim in [",", ";", " - ", " based", " relative", " moving", " along", " with"]:
            if delim in val.lower() and len(val) > 30:
                val = val[:val.lower().index(delim)].strip()

        if any(x in ll for x in ["type", "vehicle type"]) and "count" not in ll and "total" not in ll and "detail" not in ll:
            current["type"] = val.capitalize() if val else "Unidentified"
        elif any(x in ll for x in ["colour", "color"]):
            # Replace Unknown with best guess label
            colour_val = val.capitalize() if val else ""
            if not colour_val or colour_val.lower() in ["unknown", "unidentified", "unclear", "n/a", "-"]:
                colour_val = "Dark"
            current["colour"] = colour_val
        elif "size" in ll and "count" not in ll:
            current["size"] = val.capitalize() if val else "Medium"
        elif "position" in ll and "count" not in ll:
            current["position"] = val.strip() if val else "Unidentified"
        elif "movement" in ll and "count" not in ll:
            current["movement"] = val.capitalize() if val else "Unidentified"

        # When we have all 5 fields, save vehicle and reset
        if all(k in current for k in ["type","colour","size","position","movement"]):
            vehicles.append(dict(current))
            current = {}

    # If leftover partial block, try to save it
    if current and "type" in current:
        current.setdefault("colour",   "Unknown")
        current.setdefault("size",     "Unknown")
        current.setdefault("movement", "Unknown")
        vehicles.append(dict(current))

    # ── Build fixed output ────────────────────────
    lines_out = []

    if not vehicles:
        lines_out.append("  No vehicle details available.")
    else:
        # Use real tracker IDs from dets if available, else generate
        type_nums = {}
        for i, v in enumerate(vehicles):
            # Prefer real tracker ID from dets
            if dets and i < len(dets):
                vid = dets[i].get("id", None)
            else:
                vid = None
            if not vid:
                vtype  = v["type"].lower()
                prefix = {"car":"CAR","van":"VAN","motorcycle":"MOTO",
                          "truck":"TRK","lorry":"TRK","bus":"BUS"}.get(vtype, vtype[:3].upper())
                type_nums[vtype] = type_nums.get(vtype, 0) + 1
                vid = f"{prefix}{type_nums[vtype]:02d}"
            lines_out.append(f"  {vid} :")
            lines_out.append(f"    Type     : {v['type']}")
            lines_out.append(f"    Colour   : {v['colour']}")
            lines_out.append(f"    Size     : {v['size']}")
            lines_out.append(f"    Movement : {v['movement']}")
            if i < len(vehicles) - 1:
                lines_out.append("")

    return "\n".join(lines_out)


def strip_summary_header(text):
    """
    Remove the count-summary header lines so only per-vehicle
    detail blocks are shown in the UI.
    Strips: "Total Vehicle Count : N", "Car Count : N",
            "Van Count : N", etc., and the "Vehicle Details :" label.
    """
    skip_prefixes = (
        "total vehicle count",
        "car count",
        "van count",
        "motorcycle count",
        "lorry",
        "bus count",
        "vehicle details",
    )
    out = []
    for line in text.splitlines():
        ll = line.lower().strip()
        if any(ll.startswith(p) for p in skip_prefixes):
            continue
        out.append(line)
    # Drop leading/trailing blank lines
    result = "\n".join(out).strip()
    return result

def save_summary(description, output_path):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(strip_summary_header(description))
        print(f"Summary saved: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving summary: {e}")
        return False

def generate_description_and_audio(image_path, yolo_count=None, dets_snapshot=None):
    """
    Full pipeline: crop each vehicle, enhance, get HSV guess,
    ask Ollama per-crop, then build final description.
    """
    global description_generated

    # Snapshot counts at call time so they don't change during Ollama processing
    snap_counts = dict(type_counts)
    snap_total  = yolo_count if yolo_count else sum(snap_counts.values())

    # Use snapshot of detections taken at trigger time (IDs match the saved frame)
    dets  = dets_snapshot if dets_snapshot is not None else list(current_detections)
    with _frame_lock:
        frame = current_frame.copy() if current_frame is not None else None

    lines_out = []

    # ── Counts from live YOLO snapshot (no lag) ──
    lines_out.append(f"Total Vehicle Count : {snap_total}")
    if snap_counts.get("car",        0) > 0:
        lines_out.append(f"Car Count           : {snap_counts['car']}")
    if snap_counts.get("van",        0) > 0:
        lines_out.append(f"Van Count           : {snap_counts['van']}")
    if snap_counts.get("motorcycle", 0) > 0:
        lines_out.append(f"Motorcycle Count    : {snap_counts['motorcycle']}")
    if snap_counts.get("lorry",      0) > 0:
        lines_out.append(f"Lorry / Truck Count : {snap_counts['lorry']}")
    if snap_counts.get("bus",        0) > 0:
        lines_out.append(f"Bus Count           : {snap_counts['bus']}")
    lines_out.append("")
    lines_out.append("Vehicle Details :")
    lines_out.append("")

    if not dets or frame is None:
        # No YOLO detections available — fall back to asking Ollama about
        # the WHOLE frame instead (get_image_description, Options B+C+D).
        print("[Ollama] No detections for crop mode — using full frame")
        raw = get_image_description(image_path, yolo_count=yolo_count)
        if raw:
            desc = reformat_description(raw, live_counts=type_counts, dets=dets)
        else:
            desc = "\n".join(lines_out) + "\n  No vehicle details available."
    else:
        # Normal path: YOLO found vehicles, so describe each one
        # individually instead of the whole frame — crops each vehicle
        # out (Option A), from an already-enhanced image (Option B), and
        # asks Ollama once per vehicle with the HSV colour guess included
        # as a cross-check hint (Option C's hint, not its "ask 3 times"
        # part — that's only used in the whole-frame fallback above).
        crops = crop_vehicles(frame, dets)
        vehicle_details = []

        for i, (crop_info, det) in enumerate(zip(crops, dets), 1):
            # Get persistent ID for this detection
            det_vid = det.get("id", f"VEH{i:02d}")
            print(f"[Ollama] Analysing {det_vid} "
                  f"({det['type']}, HSV={crop_info['colour_guess']} "
                  f"{crop_info['conf_pct']}%)")

            result = describe_vehicle_crop(crop_info, i, det)

            VALID_TYPES = {"car", "van", "motorcycle", "lorry", "bus"}

            if result:
                # Start with YOLO type as default
                vdet = {
                    "type":     det["type"].capitalize(),
                    "colour":   crop_info["colour_guess"].capitalize(),
                    "size":     "Medium",
                    "movement": "Unknown",
                }
                for line in result.splitlines():
                    ll  = line.lower().strip()
                    val = line.split(":", 1)[-1].strip() if ":" in line else ""
                    val_clean = val.lower().strip()

                    # Type — let Ollama correct YOLO if it gives a valid type
                    if ll.startswith("type") and val_clean in VALID_TYPES:
                        vdet["type"] = val.capitalize()

                    # Colour — prefer HSV when confident, else trust Ollama
                    elif "colour" in ll or "color" in ll:
                        if val_clean and val_clean not in [
                            "unknown", "unclear", "n/a", "-", ""
                        ]:
                            if crop_info["conf_pct"] < 70:
                                vdet["colour"] = val.capitalize()

                    elif ll.startswith("size") and val_clean in [
                        "small", "medium", "large"
                    ]:
                        vdet["size"] = val.capitalize()

                    elif ll.startswith("movement") and val_clean in [
                        "moving", "stationary"
                    ]:
                        vdet["movement"] = val.capitalize()

                print(f"[Ollama] {det_vid}: "
                      f"YOLO={det['type']} → Ollama={vdet['type']}, "
                      f"Colour={vdet['colour']}")
            else:
                # Ollama failed — use YOLO type + HSV colour
                vdet = {
                    "type":     det["type"].capitalize(),
                    "colour":   crop_info["colour_guess"].capitalize(),
                    "size":     "Medium",
                    "movement": "Unknown",
                }

            vehicle_details.append(vdet)

        # Update tracked_vehicles with Ollama-corrected colour
        # Use det ID directly — avoids the frame_count scope issue
        for i, v in enumerate(vehicle_details):
            if i < len(dets):
                vid = dets[i].get("id")
                if vid and vid in tracked_vehicles:
                    tracked_vehicles[vid]["colour"] = v.get("colour","unknown").lower()
        # Sync colour to current_detections
        for det in current_detections:
            vid = det.get("id")
            if vid and vid in tracked_vehicles:
                det["colour"] = tracked_vehicles[vid].get("colour","unknown")

        # Recount using Ollama-corrected types
        corrected_counts = {"car":0,"van":0,"motorcycle":0,"lorry":0,"bus":0}
        for v in vehicle_details:
            t = v["type"].lower()
            if t in corrected_counts:
                corrected_counts[t] += 1

        corrected_total = sum(corrected_counts.values())

        # Rebuild count lines with corrected types
        lines_out = []
        lines_out.append(f"Total Vehicle Count : {corrected_total}")
        if corrected_counts["car"]        > 0:
            lines_out.append(f"Car Count           : {corrected_counts['car']}")
        if corrected_counts["van"]        > 0:
            lines_out.append(f"Van Count           : {corrected_counts['van']}")
        if corrected_counts["motorcycle"] > 0:
            lines_out.append(f"Motorcycle Count    : {corrected_counts['motorcycle']}")
        if corrected_counts["lorry"]      > 0:
            lines_out.append(f"Lorry / Truck Count : {corrected_counts['lorry']}")
        if corrected_counts["bus"]        > 0:
            lines_out.append(f"Bus Count           : {corrected_counts['bus']}")
        lines_out.append("")
        lines_out.append("Vehicle Details :")
        lines_out.append("")

        # Build vehicle detail lines — use the exact tracker ID from dets
        for i, (v, det) in enumerate(zip(vehicle_details, dets)):
            vid = det.get("id", f"VEH{i+1:02d}")   # e.g. CAR07 from tracker
            lines_out.append(f"  {vid} :")
            lines_out.append(f"    Type     : {v['type']}")
            lines_out.append(f"    Colour   : {v['colour']}")
            lines_out.append(f"    Size     : {v['size']}")
            lines_out.append(f"    Movement : {v['movement']}")
            if i < len(vehicle_details) - 1:
                lines_out.append("")
            print(f"[Description] {vid} → {v['type']} {v['colour']}")

        desc = "\n".join(lines_out)

    summary_path = os.path.join(SUMMARY_DIR, "summary_current.txt")
    save_summary(desc, summary_path)
    socketio.emit('summary_update', {'summary': strip_summary_header(desc)})
    print(f"[Socket] Emitted summary_update ({len(desc)} chars)")
    socketio.emit('detections_update', {'detections': list(current_detections)})
    description_generated = True

# ─────────────────────────────────────────────
# MAVLink global state — everything below this point until the video
# loop is about talking to the Pixhawk flight controller: connecting,
# reading its live telemetry (GPS, heading, flight mode), and sending it
# commands (GOTO, LAND, SET_MODE). mav_conn is the actual connection
# object; mavlink_sim=True means no real Pixhawk is connected, so
# commands get logged/simulated instead of actually sent.
# ─────────────────────────────────────────────
import math

def get_lan_ip():
    """
    Best-effort real LAN/WiFi IP — the address a SEPARATE device on the
    same WiFi network would use to reach this Jetson.

    Deliberately does NOT use the "open a UDP socket to 8.8.8.8 and see
    what local IP gets used" trick — that only works if the VPN is NOT
    configured to redirect all traffic through the tunnel. Many client
    VPNs (including this one) DO redirect all traffic, which makes that
    trick return the VPN's own IP instead of the real WiFi IP, since
    that genuinely becomes the OS's default route once connected.

    Instead, this reads network interfaces directly and picks the
    WiFi/Ethernet one BY NAME, explicitly skipping loopback, VPN (tun*/
    tap*), and Docker (docker0/veth*/br-*) interfaces — so it stays
    correct regardless of what the VPN does to routing.
    """
    try:
        out = subprocess.check_output(
            ["ip", "-4", "-o", "addr", "show"],
            stderr=subprocess.DEVNULL, timeout=2
        ).decode()
    except Exception:
        return None

    skip_prefixes = ("lo", "tun", "tap", "docker", "veth", "br-")
    candidates = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        iface = parts[1]
        if iface.startswith(skip_prefixes):
            continue
        m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", line)
        if m:
            candidates[iface] = m.group(1)

    # Prefer the actual WiFi/Ethernet interface names if present
    for preferred in ("wlan0", "eth0", "enp", "wlp"):
        for iface, ip in candidates.items():
            if iface.startswith(preferred):
                return ip

    # Fall back to whatever non-excluded interface was found
    return next(iter(candidates.values()), None)

def get_vpn_ip(interface="tun0"):
    """
    Best-effort VPN tunnel IP — the address assigned to this machine's
    OpenVPN interface (tun0 by default), only reachable by OTHER devices
    also connected to that same VPN. Returns None if no VPN is currently
    connected (interface doesn't exist).
    """
    try:
        out = subprocess.check_output(
            ["ip", "-4", "addr", "show", interface],
            stderr=subprocess.DEVNULL, timeout=2
        ).decode()
        m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None

mav_conn       = None
mavlink_status = "disconnected"
mavlink_sim    = True
drone_cmd_log  = []

# ── Live drone GPS + heading, read from Pixhawk telemetry ──
# Populated continuously by _telemetry_loop() from GLOBAL_POSITION_INT
# (lat/lon/alt) and ATTITUDE (yaw) MAVLink messages. This is what lets the
# goto pipeline convert a camera pixel into a real GPS point: it needs to
# know where the drone itself currently is and which way it's facing.
drone_gps        = {"lat": None, "lon": None, "alt": None}
drone_heading    = 0.0
drone_flight_mode = "UNKNOWN"   # e.g. "STABILIZE", "RTL", "GUIDED" — updated from HEARTBEAT
_telemetry_lock  = threading.Lock()
_telemetry_thread = None
_telemetry_stop  = threading.Event()

def connect_mavlink():
    global mav_conn, mavlink_status, mavlink_sim
    if not MAVLINK_AVAILABLE:
        mavlink_status = "simulated"
        mavlink_sim    = True
        socketio.emit("mavlink_status", {"status": "simulated"})
        print_result("Pixhawk", "Pixhawk is NOT connected (pymavlink not installed — simulation mode)", False)
        return True
    try:
        mav_conn = mavutil.mavlink_connection(MAVLINK_PORT, baud=MAVLINK_BAUD)
        hb = mav_conn.wait_heartbeat(timeout=5)
        if hb is None:
            # wait_heartbeat() does NOT raise on timeout — it just returns
            # None. Without this explicit check, a successfully-opened but
            # unanswered port (e.g. the device file exists but nothing is
            # actually wired to it) was being silently treated as "connected."
            raise TimeoutError(f"No HEARTBEAT received on {MAVLINK_PORT} within 5s — nothing appears to be connected")
        mavlink_status = "connected"
        mavlink_sim    = False
        socketio.emit("mavlink_status", {"status": "connected"})
        print_result("Pixhawk", "Pixhawk is connected", True, detail=f"{MAVLINK_PORT} @ {MAVLINK_BAUD} baud")
        request_data_streams()
        start_telemetry_thread()
        return True
    except Exception as e:
        mavlink_status = "simulated"
        mavlink_sim    = True
        socketio.emit("mavlink_status", {"status": "simulated"})
        print_result("Pixhawk", "Pixhawk is NOT connected (running in simulation mode)", False, detail=str(e))
        return True

def request_data_streams():
    """Ask the flight controller to stream position + attitude data."""
    if mavlink_sim or mav_conn is None:
        return
    try:
        mav_conn.mav.request_data_stream_send(
            mav_conn.target_system, mav_conn.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1  # 4 Hz
        )
    except Exception as e:
        print(f"[MAVLink] request_data_stream failed: {e}")

def _decode_flight_mode(msg):
    """
    HEARTBEAT.custom_mode is just a number whose meaning depends on the
    autopilot type — pymavlink's mode_mapping() (populated after the first
    heartbeat) gives the name<->number mapping for whatever's connected.
    """
    try:
        mapping = mav_conn.mode_mapping()   # {"STABILIZE": 0, "RTL": 6, ...}
        if mapping:
            for name, number in mapping.items():
                if number == msg.custom_mode:
                    return name
    except Exception:
        pass
    return f"MODE_{msg.custom_mode}"

def _telemetry_loop():
    """
    Background loop: continuously reads GLOBAL_POSITION_INT (lat/lon/alt),
    ATTITUDE (yaw), and HEARTBEAT (flight mode) from the Pixhawk and stores
    the latest values in drone_gps / drone_heading / drone_flight_mode.
    Runs at whatever rate the flight controller streams at (~4 Hz requested
    above); HEARTBEAT itself always streams at ~1 Hz regardless.
    """
    global drone_gps, drone_heading, drone_flight_mode
    _last_known_mode = None
    while not _telemetry_stop.is_set():
        if mavlink_sim or mav_conn is None:
            time.sleep(0.5)
            continue
        try:
            msg = mav_conn.recv_match(
                type=["GLOBAL_POSITION_INT", "ATTITUDE", "HEARTBEAT"], blocking=True, timeout=1.0
            )
            if msg is None:
                continue
            mtype = msg.get_type()
            if mtype == "GLOBAL_POSITION_INT":
                with _telemetry_lock:
                    drone_gps["lat"] = msg.lat / 1e7          # degE7 -> degrees
                    drone_gps["lon"] = msg.lon / 1e7
                    drone_gps["alt"] = msg.relative_alt / 1000.0  # mm -> m
            elif mtype == "ATTITUDE":
                with _telemetry_lock:
                    drone_heading = (math.degrees(msg.yaw) + 360) % 360
            elif mtype == "HEARTBEAT":
                mode_name = _decode_flight_mode(msg)
                with _telemetry_lock:
                    drone_flight_mode = mode_name
                if mode_name != _last_known_mode:
                    socketio.emit("flight_mode_update", {"mode": mode_name})
                    if _last_known_mode is not None:   # skip the very first read on connect
                        log_session_event(f"Flight mode changed: {_last_known_mode} -> {mode_name}", status="INFO")
                    _last_known_mode = mode_name
        except Exception as e:
            print(f"[MAVLink] Telemetry error: {e}")
            time.sleep(0.5)

def start_telemetry_thread():
    global _telemetry_thread
    if _telemetry_thread is None or not _telemetry_thread.is_alive():
        _telemetry_stop.clear()
        _telemetry_thread = threading.Thread(target=_telemetry_loop, daemon=True)
        _telemetry_thread.start()
        print("[MAVLink] Telemetry thread started (GPS + heading)")

def get_drone_state():
    """Returns the drone's best-known (lat, lon, heading, altitude) right now.

    Falls back to SIM_HOME_GPS / heading 0 / DEFAULT_SIM_ALT when there's no
    live fix yet (simulation mode, or real hardware that hasn't reported a
    position), so the rest of the pipeline always has something to compute
    from.
    """
    with _telemetry_lock:
        lat = drone_gps["lat"]
        lon = drone_gps["lon"]
        alt = drone_gps.get("alt")
        heading = drone_heading
    if lat is None or lon is None:
        lat = SIM_HOME_GPS["lat"]
        lon = SIM_HOME_GPS["lon"]
    if alt is None:
        alt = 10.0
    return lat, lon, heading, alt

def get_flight_mode():
    """Returns the drone's best-known current flight mode name right now."""
    with _telemetry_lock:
        return drone_flight_mode

def send_set_mode(mode_name):
    """
    Send MAV_CMD_DO_SET_MODE (via pymavlink's set_mode() helper, which
    picks the right underlying message for ArduPilot) to switch flight
    mode. In simulation mode (no real Pixhawk), just updates the tracked
    mode directly so the UI/logic can still be tested end-to-end.
    Returns True/False for whether the command was actually sent.
    """
    global drone_flight_mode
    success = True
    detail  = ""
    if not mavlink_sim and mav_conn is not None:
        try:
            mav_conn.set_mode(mode_name)
        except Exception as e:
            success = False
            detail  = str(e)
    else:
        # No real Pixhawk — simulate the mode change immediately.
        with _telemetry_lock:
            drone_flight_mode = mode_name
        socketio.emit("flight_mode_update", {"mode": mode_name})

    if success:
        print_result("Flight Mode", f"Flight mode changed to {mode_name}", True,
                     detail="(SIM)" if mavlink_sim else "(command sent — Pixhawk confirms via HEARTBEAT)")
    else:
        print_result("Flight Mode", f"Failed to change flight mode to {mode_name}", False, detail=detail)
    return success

# ─────────────────────────────────────────────
# The pixel -> real-world GPS pipeline. This is the core of how clicking
# a vehicle on screen becomes a place the drone can fly to. Three steps,
# each handled by one function below, always used in this order:
#   1. pixel_to_ned()          — vehicle's screen position -> metres
#                                offset from the camera (still relative
#                                to the CAMERA's own view, not compass
#                                directions yet)
#   2. ned_to_distance_bearing() — rotates that offset by the drone's
#                                actual compass heading, turning it into
#                                a true distance + compass bearing
#   3. destination_point()      — combines that distance/bearing with the
#                                drone's own current GPS position to work
#                                out the vehicle's actual GPS coordinate
# ─────────────────────────────────────────────
def pixel_to_ned(cx, cy, fw=1280, fh=720, alt=10.0, hfov=62.2):
    """
    Camera-frame offset: 'north' = up the image, 'east' = right of image.
    Only lines up with true compass north/east if the drone happens to be
    facing north (yaw = 0) — see ned_to_distance_bearing() for the fix.
    """
    fx    = fw / (2 * math.tan(math.radians(hfov / 2)))
    gsd   = alt / fx
    north = round(-(cy - fh / 2) * gsd, 2)
    east  = round( (cx - fw / 2) * gsd, 2)
    return north, east

EARTH_RADIUS_METERS = 6371000

def ned_to_distance_bearing(north, east, drone_heading_deg=0.0):
    """
    Rotate a camera-frame (north, east) offset into world-frame by the
    drone's current yaw, then convert to a great-circle distance (m) and
    compass bearing (0-360 deg). This is the step the AIVL vessel model
    doesn't need (it derives bearing straight from pixel angle + heading),
    because here the offset is already resolved into metres via altitude —
    it just needs de-rotating into true north/east first.
    """
    yaw = math.radians(drone_heading_deg)
    world_north = north * math.cos(yaw) - east * math.sin(yaw)
    world_east  = north * math.sin(yaw) + east * math.cos(yaw)
    distance = math.hypot(world_north, world_east)
    bearing  = (math.degrees(math.atan2(world_east, world_north)) + 360) % 360
    return distance, bearing

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_METERS * c

def destination_point(lat, lon, distance_m, bearing_deg):
    """
    Great-circle 'destination point' formula: given a start point, a
    distance, and a bearing, returns the (lat, lon) you land on. Ported
    directly from the AIVL vessel locator's GPSCalculator — same spherical
    trig, just fed the drone's own GPS instead of a boat's.
    """
    br     = math.radians(bearing_deg)
    lat_r  = math.radians(lat)
    lon_r  = math.radians(lon)
    ang    = distance_m / EARTH_RADIUS_METERS

    lat2_r = math.asin(
        math.sin(lat_r) * math.cos(ang) +
        math.cos(lat_r) * math.sin(ang) * math.cos(br)
    )
    lon2_r = lon_r + math.atan2(
        math.sin(br) * math.sin(ang) * math.cos(lat_r),
        math.cos(ang) - math.sin(lat_r) * math.sin(lat2_r)
    )
    return math.degrees(lat2_r), math.degrees(lon2_r)

def send_goto(north, east, down=-10.0):
    global drone_cmd_log
    ts = datetime.now().strftime("%H:%M:%S")
    success = True
    if not mavlink_sim and mav_conn is not None:
        try:
            mav_conn.mav.set_position_target_local_ned_send(
                0,
                mav_conn.target_system,
                mav_conn.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111111000,
                north, east, down,
                0, 0, 0, 0, 0, 0, 0, 0
            )
            entry = f"[{ts}] REAL  GOTO  N={north}m  E={east}m  ALT={abs(down)}m"
        except Exception as e:
            success = False
            entry = f"[{ts}] ERROR: {e}"
    else:
        entry = f"[{ts}] [SIM] GOTO  N={north}m  E={east}m  ALT={abs(down)}m"
    drone_cmd_log.append(entry)
    drone_cmd_log = drone_cmd_log[-20:]
    socketio.emit("drone_cmd", {"log": entry, "all": drone_cmd_log})
    log_command("GOTO", north=north, east=east, altitude=abs(down),
                mode="REAL" if not mavlink_sim else "SIM", detail=entry, success=success)
    print(f"[MAVLink] {entry}")
    return success

def send_goto_global(lat, lon, alt=10.0):
    """
    Send SET_POSITION_TARGET_GLOBAL_INT — an absolute GPS waypoint, instead
    of send_goto()'s local-frame offset. MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
    means altitude is relative to home/takeoff point, lat/lon are real GPS
    coordinates. This avoids MAV_FRAME_LOCAL_NED's ambiguity of being
    relative to the EKF origin rather than the drone's current position.
    """
    global drone_cmd_log
    ts = datetime.now().strftime("%H:%M:%S")
    success = True
    if not mavlink_sim and mav_conn is not None:
        try:
            mav_conn.mav.set_position_target_global_int_send(
                0,
                mav_conn.target_system,
                mav_conn.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                0b0000111111111000,          # only lat/lon/alt fields active
                int(lat * 1e7), int(lon * 1e7), alt,
                0, 0, 0, 0, 0, 0, 0, 0
            )
            entry = f"[{ts}] REAL  GOTO_GPS  lat={lat:.6f}  lon={lon:.6f}  ALT={alt}m"
        except Exception as e:
            success = False
            entry = f"[{ts}] ERROR: {e}"
    else:
        entry = f"[{ts}] [SIM] GOTO_GPS  lat={lat:.6f}  lon={lon:.6f}  ALT={alt}m"
    drone_cmd_log.append(entry)
    drone_cmd_log = drone_cmd_log[-20:]
    socketio.emit("drone_cmd", {"log": entry, "all": drone_cmd_log})
    log_command("GOTO_GPS", altitude=alt,
                mode="REAL" if not mavlink_sim else "SIM",
                detail=f"{entry} (lat={lat:.6f}, lon={lon:.6f})", success=success)
    print(f"[MAVLink] {entry}")
    return success

def send_land_command():
    """Send MAV_CMD_NAV_LAND — vehicle performs its own final landing flare."""
    global drone_cmd_log
    ts = datetime.now().strftime("%H:%M:%S")
    success = True
    if not mavlink_sim and mav_conn is not None:
        try:
            mav_conn.mav.command_long_send(
                mav_conn.target_system, mav_conn.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND,
                0, 0, 0, 0, 0, 0, 0, 0
            )
            entry = f"[{ts}] REAL  LAND command sent"
        except Exception as e:
            success = False
            entry = f"[{ts}] ERROR: {e}"
    else:
        entry = f"[{ts}] [SIM] LAND command sent"
    drone_cmd_log.append(entry)
    drone_cmd_log = drone_cmd_log[-20:]
    socketio.emit("drone_cmd", {"log": entry, "all": drone_cmd_log})
    log_command("LAND", mode="REAL" if not mavlink_sim else "SIM", detail=entry, success=success)
    print(f"[MAVLink] {entry}")
    return success

# ─────────────────────────────────────────────
# ArUco / "H" marker landing — this is what the "Landing" button in the
# browser triggers. Rather than landing blindly, the drone looks for a
# printed ArUco marker (a small black-and-white square pattern, like a
# simplified QR code) on the ground, and uses ITS position in the camera
# frame the same way pixel_to_ned() uses a vehicle's position — to know
# exactly where to descend to, correcting its position as it gets lower.
# Uses OpenCV's built-in ArUco module — no extra dependency beyond
# opencv-contrib (already bundled with most opencv-python installs).
# ─────────────────────────────────────────────
try:
    _ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    _ARUCO_PARAMS = cv2.aruco.DetectorParameters()
    _aruco_detector = cv2.aruco.ArucoDetector(_ARUCO_DICT, _ARUCO_PARAMS)
    ARUCO_AVAILABLE = True
except Exception as e:
    ARUCO_AVAILABLE = False
    print(f"[WARN] cv2.aruco not available ({e}) — landing marker detection disabled")

# Landing state machine states:
#   idle      — not landing
#   searching — descending slightly while scanning for marker
#   centering — marker found, correcting horizontal position
#   final     — centered and low altitude, executing final descent
#   landed    — MAV_CMD_NAV_LAND sent, sequence complete
#   lost      — marker lost during sequence, aborted back to idle
landing_state   = "idle"
landing_target  = None     # last known marker pixel centre (cx, cy)
landing_alt     = 10.0     # current simulated/tracked altitude (m)
landing_log     = []

# Tunables
LANDING_CENTER_TOL_PX = 25     # marker must be within this many px of frame centre
LANDING_DESCEND_RATE  = 0.4    # metres descended per inference cycle while centering
LANDING_FINAL_ALT     = 1.2    # altitude (m) at which MAV_CMD_NAV_LAND is triggered
LANDING_MIN_ALT       = 0.0
LANDING_LOST_FRAMES   = 30     # consecutive missed-marker frames before aborting

def detect_landing_marker(frame):
    """
    Detect ArUco markers (any ID in DICT_4X4_50) on the given BGR frame.
    Returns (cx, cy, corners) of the largest detected marker, or None.
    A real "H"-pad detector would use template/shape matching; ArUco is
    used here because it's robust, fast, and built into OpenCV already —
    most DIY landing pads use a printed ArUco tag for this reason.
    """
    if not ARUCO_AVAILABLE:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _aruco_detector.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        return None
    # Pick the largest marker (closest / most reliable) if multiple seen
    best_idx  = 0
    best_area = 0
    for i, c in enumerate(corners):
        pts  = c.reshape(4, 2)
        area = cv2.contourArea(pts)
        if area > best_area:
            best_area = area
            best_idx  = i
    pts = corners[best_idx].reshape(4, 2)
    cx  = int(pts[:, 0].mean())
    cy  = int(pts[:, 1].mean())
    return (cx, cy, pts)

def draw_landing_marker(frame, marker):
    """Overlay a crosshair + box on the detected marker for operator feedback."""
    if marker is None:
        return
    cx, cy, pts = marker
    cv2.polylines(frame, [pts.astype(int)], True, (0, 255, 0), 2)
    cv2.drawMarker(frame, (cx, cy), (0, 255, 0),
                    markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.putText(frame, "LANDING MARKER", (cx - 60, cy - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def _landing_log_entry(msg):
    global landing_log
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    landing_log.append(entry)
    landing_log = landing_log[-20:]
    socketio.emit("landing_status", {
        "state": landing_state, "log": entry, "all": landing_log,
        "altitude": round(landing_alt, 2),
    })
    log_command("LANDING_EVENT", altitude=landing_alt,
                mode="REAL" if not mavlink_sim else "SIM", detail=msg)
    log_session_event(f"[Landing] {msg}")
    print(f"[Landing] {entry}")

def start_landing_sequence():
    """Called by the API route when the user presses the Landing button."""
    global landing_state, landing_alt, landing_log
    landing_state = "searching"
    landing_log   = []
    _landing_log_entry("Landing sequence initiated — searching for marker")
    return True

def abort_landing_sequence(reason="Aborted by user"):
    global landing_state
    landing_state = "idle"
    _landing_log_entry(f"Landing sequence aborted: {reason}")

def step_landing_sequence(frame, fw, fh):
    """
    Called once per inference frame while landing_state != 'idle'.
    Runs the search -> center -> final -> landed state machine and
    draws the marker overlay onto `frame` in place.
    """
    global landing_state, landing_target, landing_alt

    if landing_state == "idle":
        return

    marker = detect_landing_marker(frame)
    draw_landing_marker(frame, marker)

    if marker is None:
        landing_target = (landing_target or (0, 0))  # keep last known for HUD
        step_landing_sequence._lost_count = getattr(step_landing_sequence, "_lost_count", 0) + 1
        if step_landing_sequence._lost_count >= LANDING_LOST_FRAMES:
            abort_landing_sequence("Marker lost — search timed out")
            step_landing_sequence._lost_count = 0
        return
    step_landing_sequence._lost_count = 0

    cx, cy, _ = marker
    landing_target = (cx, cy)
    north, east = pixel_to_ned(cx, cy, fw=fw, fh=fh, alt=max(landing_alt, 0.5))
    dist_px = ((cx - fw/2)**2 + (cy - fh/2)**2) ** 0.5

    if landing_state == "searching":
        landing_state = "centering"
        _landing_log_entry(f"Marker acquired — centering (alt={landing_alt:.1f}m)")

    elif landing_state == "centering":
        if dist_px <= LANDING_CENTER_TOL_PX:
            # Centered — begin descent
            landing_alt = max(LANDING_MIN_ALT, landing_alt - LANDING_DESCEND_RATE)
            send_goto(north, east, -landing_alt)
            if landing_alt <= LANDING_FINAL_ALT:
                landing_state = "final"
                _landing_log_entry(f"Centered + low altitude — entering final descent")
        else:
            # Off-centre — correct horizontal position, hold altitude
            send_goto(north, east, -landing_alt)

    elif landing_state == "final":
        if dist_px <= LANDING_CENTER_TOL_PX * 1.5:
            landing_alt = max(LANDING_MIN_ALT, landing_alt - LANDING_DESCEND_RATE * 1.5)
            send_goto(north, east, -landing_alt)
            if landing_alt <= LANDING_MIN_ALT + 0.05:
                send_land_command()
                landing_state = "landed"
                _landing_log_entry("Touchdown — LAND command sent")
        else:
            # Drifted during final approach — re-center first
            landing_state = "centering"
            _landing_log_entry("Drifted during final approach — re-centering")

# ─────────────────────────────────────────────
# Two-thread video architecture
#
# _inference_loop : reads frames, runs YOLO, draws boxes.
#   Runs in background as fast as GPU allows — no sleep, no I/O blocking.
#   Publishes annotated frame to _latest_annotated.
#
# process_video   : stream generator in Flask response thread.
#   Only job = grab _latest_annotated, encode JPEG, yield at 20 FPS.
#   NEVER blocked by YOLO inference time.
#
# Everything from here down to _inference_loop() itself is supporting
# infrastructure for that loop: a small IOU (box-overlap) helper used by
# the tracker, GPU/CPU load readers for the terminal status line, and a
# background camera-reading thread (_LatestFrameGrabber) that keeps live
# webcam/RTSP capture from blocking on the camera's own speed.
# ─────────────────────────────────────────────

def _iou(a, b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter==0: return 0.0
    return inter/((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter)


def _read_gpu_load_percent():
    """
    Best-effort Jetson GPU load reading, straight from the same sysfs
    interface tegrastats itself reads. Path varies by Jetson chip
    generation, so try known candidates in order. Returns None if none
    are readable (e.g. running this code on a non-Jetson machine).
    """
    candidates = [
        "/sys/devices/platform/bus@0/17000000.gpu/load",   # confirmed correct for this Orin Nano
        "/sys/devices/gpu.0/load",
        "/sys/devices/57000000.gpu/load",
        "/sys/devices/17000000.gpu/load",
        "/sys/devices/platform/host1x/57000000.gpu/load",
        "/sys/devices/platform/host1x/17000000.gpu/load",
    ]
    for path in candidates:
        try:
            with open(path) as f:
                val = int(f.read().strip())
            return val / 10.0   # sysfs reports per-mille — value/10 = percent
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    return None


_prev_cpu_times = None
def _read_cpu_load_percent():
    """
    Self-contained overall CPU% via /proc/stat deltas between calls —
    no psutil dependency needed. Returns None on the very first call
    (needs two samples to compute a delta) or if /proc/stat is unreadable.
    """
    global _prev_cpu_times
    try:
        with open("/proc/stat") as f:
            parts = [int(x) for x in f.readline().split()[1:]]
        idle, total = parts[3] + parts[4], sum(parts)
        if _prev_cpu_times is None:
            _prev_cpu_times = (idle, total)
            return None
        prev_idle, prev_total = _prev_cpu_times
        d_idle, d_total = idle - prev_idle, total - prev_total
        _prev_cpu_times = (idle, total)
        return 100.0 * (1.0 - d_idle / d_total) if d_total > 0 else None
    except Exception:
        return None


STREAM_STALL_TIMEOUT_SEC = 10   # if no successful frame for this long, consider the stream dead

class _LatestFrameGrabber:
    """
    Continuously reads frames from a live camera in its own background
    thread, always holding onto only the most recent one. The inference
    loop then just grabs whatever's freshest instead of blocking on
    cap.read() itself — this stops camera latency and YOLO inference
    time from stacking sequentially, so overall throughput can approach
    min(camera's real fps, inference's real fps) instead of being their sum.
    Only used for live sources (webcam/RTSP) — video files stay on the
    original sequential path, since frame order matters there.

    Also detects a STALLED stream (e.g. an RTSP feed over a flaky VPN
    link that goes quiet without OpenCV raising a clean error) — sets
    self.stalled=True after STREAM_STALL_TIMEOUT_SEC with no successful
    frame, which _inference_loop checks to trigger an automatic
    reconnect, instead of the app silently freezing on a dead stream.
    """
    def __init__(self, cap):
        self.cap        = cap
        self.lock        = threading.Lock()
        self.frame       = None
        self.ret         = False
        self.last_fps    = 0.0   # updated ~once/sec, read by main loop's unified status line
        self.stalled     = False
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        _cam_frames = 0
        _cam_t0 = time.time()
        _last_success_ts = time.time()
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

            if ret:
                _last_success_ts = time.time()
                # ── Camera-only FPS — true capture rate, independent of ──
                # ── whatever the inference loop is doing with each frame ──
                _cam_frames += 1
                _elapsed = time.time() - _cam_t0
                if _elapsed >= 1.0:
                    self.last_fps = _cam_frames / _elapsed
                    _cam_frames = 0
                    _cam_t0 = time.time()
            else:
                # No frame this cycle — check whether it's been long
                # enough with NO successful frames that we should
                # consider the stream dead rather than just briefly slow.
                if time.time() - _last_success_ts > STREAM_STALL_TIMEOUT_SEC:
                    self.stalled = True

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)


def _inference_loop():
    """
    Background inference thread — this is the heart of the whole app,
    running continuously for as long as the app is up. Each pass through
    its main loop, in order:
      1. Open/reopen the video source if it changed (video file, webcam,
         or RTSP stream) — see _needs_reopen below.
      2. Read the next frame (via the background camera grabber for live
         sources, or directly for video files).
      3. Run YOLO detection on that frame.
      4. Match detections to existing tracked vehicles (update_tracker),
         keeping each vehicle's ID stable across frames.
      5. Draw bounding boxes/labels onto the frame.
      6. Periodically (not every frame): trigger an Ollama description,
         log detections, and run the ArUco landing state machine if
         landing is active.
      7. Publish the finished frame to _latest_annotated, which the
         separate video-streaming thread (process_video) picks up and
         sends to the browser — this loop never waits on that part.
    All slow/blocking work (socketio emits, saving images to disk) is
    handed off to background threads/tasks so it can't stall this loop.
    """
    global vehicle_count, description_generated
    global VIDEO_SOURCE, video_source_changed
    global ui_visible_ids, ui_last_push
    global current_frame, _latest_annotated

    cap              = None
    cam_grabber      = None   # background reader, live sources only
    frame_count      = 0
    prev_source      = None
    last_ollama_time = 0.0
    ollama_running   = False
    _cached_videos   = []
    _last_scan       = 0.0
    _fps_frames      = 0     # frames processed since last FPS print
    _fps_t0          = time.time()
    _t_decode_sum    = 0.0   # accumulated time in cap.grab()/cap.read()
    _t_infer_sum     = 0.0   # accumulated time in MODEL() calls
    _t_post_sum      = 0.0   # accumulated time in postprocess/tracker/draw/emit

    while not _inf_stop.is_set():
        # ── Video list — rescan disk at most once per 5 s ─────────
        now = time.time()
        if now - _last_scan > 5.0:
            _cached_videos = get_video_files()
            _last_scan     = now
        if not _cached_videos:
            time.sleep(0.5); continue
        if VIDEO_SOURCE is None:
            VIDEO_SOURCE = _cached_videos[0]

        # ── Open / switch video source ─────────────────────────────
        _is_live_source = isinstance(VIDEO_SOURCE, int) or \
                           (isinstance(VIDEO_SOURCE, str) and
                            (VIDEO_SOURCE.startswith('rtsp://') or
                             VIDEO_SOURCE.startswith('http://')))

        _needs_reopen = (
            cap is None
            or video_source_changed
            or prev_source != VIDEO_SOURCE
            or (_is_live_source and cam_grabber is None)   # only live sources need a grabber
            or (cam_grabber is not None and cam_grabber.stalled)   # stream died — auto-reconnect
        )
        if _needs_reopen:
            if cam_grabber and cam_grabber.stalled:
                print_result("Stream", f"RTSP/camera stream stalled — reconnecting", False,
                            detail=f"no frame for over {STREAM_STALL_TIMEOUT_SEC}s")
            if cam_grabber: cam_grabber.stop(); cam_grabber = None
            if cap: cap.release()
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            if not cap.isOpened():
                print(f"[Inference] Failed to open: {VIDEO_SOURCE}")
                time.sleep(1); continue

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            is_live = _is_live_source

            if is_live:
                # Force MJPEG — many UVC webcams default to heavy raw YUYV
                # unless explicitly told otherwise, which is far slower to
                # pull over USB and decode than compressed MJPEG.
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  720)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 405)
                src_fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
                _frame_skip = 1   # process every frame for live camera

                # Cameras often silently ignore the requested size/format —
                # print what we actually got so we can tell immediately.
                actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
                fourcc_str = "".join([chr((fourcc_int >> 8*i) & 0xFF) for i in range(4)])
                print(f"[Inference] Live camera opened  requested fps={src_fps:.0f}  "
                      f"actual={actual_w}x{actual_h}  fourcc={fourcc_str}")
                cam_grabber = _LatestFrameGrabber(cap)
            else:
                # Video file — decode at 720×405, skip frames to match original speed
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  720)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 405)
                src_fps     = cap.get(cv2.CAP_PROP_FPS) or 24.0
                _frame_skip = max(1, round(src_fps / 12))   # was /20 — lower target fps = less GPU load on Jetson
                print(f"[Inference] Opened: {VIDEO_SOURCE}  "
                      f"src={src_fps:.0f}fps  skip={_frame_skip}")

            vehicle_count=0; description_generated=False
            video_source_changed=False; prev_source=VIDEO_SOURCE; frame_count=0
            tracked_vehicles.clear(); type_id_counters.clear(); conf_history.clear()
            ui_visible_ids=set(); ui_last_push=-999
            if landing_state != "idle":
                abort_landing_sequence("Video source changed")

        # Live camera: pull whatever the background grabber has most
        # recently captured (non-blocking). Video file: original
        # sequential grab-skip + read (frame order matters for playback).
        _t0 = time.perf_counter()
        if is_live and cam_grabber:
            ret, frame = cam_grabber.read()
        else:
            for _ in range(_frame_skip - 1):
                cap.grab()
            ret, frame = cap.read()
        _t_decode_sum += time.perf_counter() - _t0
        if not ret:
            vehicle_count=0; description_generated=False
            socketio.start_background_task(socketio.emit, 'vehicle_count_update',
                {'vehicle_count':0,'car':0,'van':0,'motorcycle':0,'lorry':0,'bus':0})
            if not is_live:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video file
            else:
                time.sleep(0.1)   # brief wait for live camera reconnect
            continue

        frame_count += 1
        if frame.shape[:2] != (405, 720):
            frame = cv2.resize(frame, (720, 405))
        h, w   = frame.shape[:2]
        f_area = w * h

        # Single YOLO pass — no CLAHE, CUDA accelerated
        _t0 = time.perf_counter()
        res = MODEL(frame, imgsz=640, verbose=False, device=DEVICE)[0]   # must match engine's fixed 640x640 input
        _t_infer_sum += time.perf_counter() - _t0
        raw = [(res.boxes, 0, 0, 1.0, 1.0)]

        # ── Unified status line — terminal only, prints every STATUS_PRINT_INTERVAL_SEC ───
        _fps_frames += 1
        _fps_elapsed = time.time() - _fps_t0
        if _fps_elapsed >= STATUS_PRINT_INTERVAL_SEC:
            _cam_fps_str = f"{cam_grabber.last_fps:.1f}" if (is_live and cam_grabber) else "N/A (video file)"
            _gpu_pct = _read_gpu_load_percent()
            _cpu_pct = _read_cpu_load_percent()
            _gpu_str = f"{_gpu_pct:.0f}%" if _gpu_pct is not None else "N/A"
            _cpu_str = f"{_cpu_pct:.0f}%" if _cpu_pct is not None else "N/A"
            print(f"[STATUS] Camera FPS: {_cam_fps_str}  "
                  f"| Exec FPS: {_fps_frames / _fps_elapsed:.1f}  "
                  f"| GPU: {_gpu_str}  | CPU: {_cpu_str}  "
                  f"(decode {1000*_t_decode_sum/_fps_frames:.1f}ms, "
                  f"infer {1000*_t_infer_sum/_fps_frames:.1f}ms, "
                  f"post {1000*_t_post_sum/_fps_frames:.1f}ms avg/frame)")
            _fps_frames = 0
            _fps_t0 = time.time()
            _t_decode_sum = 0.0
            _t_infer_sum  = 0.0
            _t_post_sum   = 0.0
        _t_post_start = time.perf_counter()   # postprocess timer starts here

        # SAHI tile pass every 60 frames only
        if frame_count % 300 == 0:   # was %60 — 5x less often, this pass does 4 extra full inferences
            TW, TH = int(w*0.6), int(h*0.6)
            for tx, ty in [(0,0),(w-TW,0),(0,h-TH),(w-TW,h-TH)]:
                tc = cv2.resize(frame[ty:ty+TH, tx:tx+TW], (640,640))
                rt = MODEL(tc, imgsz=640, verbose=False, device=DEVICE)[0]
                raw.append((rt.boxes, tx, ty, TW/640, TH/640))

        # ── Collect + dedup + shape filter ────────────────────────
        all_dets = []
        for boxes, ox, oy, sx, sy in raw:
            for box in boxes:
                cls_id=int(box.cls.item()); conf=float(box.conf.item())
                if cls_id not in VEHICLE_CLASSES: continue
                bx1,by1,bx2,by2=map(int,box.xyxy[0])
                x1=int(ox+bx1*sx); y1=int(oy+by1*sy)
                x2=int(ox+bx2*sx); y2=int(oy+by2*sy)
                bw=max(1,x2-x1); bh=max(1,y2-y1)
                pct=(bw*bh)/f_area
                if not _passes_conf(cls_id, conf, pct): continue
                all_dets.append((cls_id,conf,x1,y1,x2,y2))

        all_dets.sort(key=lambda d: d[1], reverse=True)
        kept, raw_dets, yolo_count, yolo_types = [], [], 0, {}

        for cls_id, conf, x1, y1, x2, y2 in all_dets:
            x1,y1=max(0,x1),max(0,y1); x2,y2=min(w,x2),min(h,y2)
            if x2<=x1 or y2<=y1: continue
            bw,bh=x2-x1,y2-y1; area=bw*bh
            pct = area / f_area

            # ── Size gates ────────────────────────────────────────
            if area < 150: continue          # absolute minimum — real noise
            if pct > 0.12: continue          # >12% of frame → road/structure

            # ── Aspect ratio — reject elongated non-vehicle shapes ─
            ratio = max(bw,bh) / max(min(bw,bh),1)
            if ratio > 5.5: continue         # road markings / dividers

            # ── Size-scaled confidence gate (large boxes only) ─────
            # Small/far boxes: already handled by CLASS_CONF_FAR above
            # Large boxes: need extra confidence to avoid road FP
            if pct > 0.03:
                extra   = (pct - 0.03) * 4.0
                min_req = CLASS_CONF_NEAR.get(cls_id, 0.40) + extra
                if conf < min_req: continue

            if any(_iou((x1,y1,x2,y2),k)>0.35 for k in kept): continue
            kept.append((x1,y1,x2,y2))
            vtype=classify_by_size(cls_id,area,f_area,bbox=(x1,y1,x2,y2))
            yolo_count+=1; yolo_types[vtype]=yolo_types.get(vtype,0)+1
            raw_dets.append({"type":vtype,"bbox":[x1,y1,x2,y2],
                "cx":(x1+x2)//2,"cy":(y1+y2)//2,
                "conf":round(conf,2),"colour":"unknown"})

        # ── Tracker ───────────────────────────────────────────────
        visible = update_tracker(raw_dets, frame_count)
        current_detections.clear()
        for trk in visible:
            current_detections.append({
                "id":trk["id"],"type":trk["type"],
                "colour":trk.get("colour","unknown"),
                "bbox":trk["last_bbox"],"cx":trk["cx"],"cy":trk["cy"],
                "conf":trk["conf"],"last_seen":trk["last_seen"],
            })

        # ── Draw boxes ────────────────────────────────────────────
        for trk in visible:
            bbox     = trk.get("predicted_bbox") or trk["last_bbox"]
            x1,y1,x2,y2 = bbox
            x1,y1 = max(0,x1),max(0,y1); x2,y2 = min(w,x2),min(h,y2)
            color = BOX_COLORS.get(trk["type"],(255,255,255))
            sc    = trk.get("smooth_conf", trk["conf"])
            label = f"{trk['id']} {int(sc*100)}%"
            thick = 1 if trk["last_seen"] < frame_count else 2
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,thick)
            (lw,lh),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.48,1)
            cv2.rectangle(frame,(x1,y1-lh-6),(x1+lw+4,y1),color,-1)
            cv2.putText(frame,label,(x1+2,y1-3),
                        cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,0,0),1)
        cv2.putText(frame,f"Vehicles: {yolo_count}",
                    (10,h-12),cv2.FONT_HERSHEY_SIMPLEX,0.62,(0,255,0),2)

        # ── Counts ────────────────────────────────────────────────
        trk_types={}
        for trk in visible: trk_types[trk["type"]]=trk_types.get(trk["type"],0)+1
        type_counts.update({
            "car":trk_types.get("car",0),"van":trk_types.get("van",0),
            "motorcycle":trk_types.get("motorcycle",0),
            "lorry":trk_types.get("lorry",0)+trk_types.get("truck",0),
            "bus":trk_types.get("bus",0),
        })
        vehicle_count=len(visible)

        # ── Socket emits — offloaded, never block inference thread ─
        if frame_count % 12 == 0:
            _vc=dict(type_counts); _yc=vehicle_count
            socketio.start_background_task(socketio.emit,'vehicle_count_update',{
                'vehicle_count':_yc,'car':_vc["car"],'van':_vc["van"],
                'motorcycle':_vc["motorcycle"],'lorry':_vc["lorry"],'bus':_vc["bus"],
            })
        cur_ids=set(d["id"] for d in current_detections)
        if cur_ids!=ui_visible_ids:
            ui_visible_ids=set(cur_ids)
            _snap=list(current_detections)
            socketio.start_background_task(
                socketio.emit,'detections_update',{'detections':_snap})

        # ── Ollama trigger — imwrite moved off inference thread ───
        now=time.time()
        if now-last_ollama_time>OLLAMA_INTERVAL_SEC and not ollama_running:
            last_ollama_time=now; ollama_running=True
            ts=datetime.now().strftime("%Y%m%d_%H%M%S")
            ipath=os.path.join(IMAGE_DIR,f"vehicle_{ts}.jpg")
            snap=list(current_detections); cnt=yolo_count or None
            _fc=frame.copy()
            def _run(p,c,s,img):
                nonlocal ollama_running
                try:
                    cv2.imwrite(p,img)   # disk write happens here, not above
                    generate_description_and_audio(p,yolo_count=c,dets_snapshot=s)
                finally: ollama_running=False
            threading.Thread(target=_run,args=(ipath,cnt,snap,_fc),daemon=True).start()

        # ── Landing sequence — runs only while active, draws overlay ──
        if landing_state != "idle":
            step_landing_sequence(frame, w, h)

        # ── Log detections every 15 frames (~1s at 15fps) ──────────
        # Writing every single frame would create huge CSV files;
        # every 15 frames gives a useful ~1-second-resolution record.
        if frame_count % 15 == 0 and visible:
            log_detection(VIDEO_SOURCE, frame_count, visible)

        # Frame already 720×405 — publish directly
        with _latest_lock:
            _latest_annotated = frame
        with _frame_lock:
            current_frame = frame
        _t_post_sum += time.perf_counter() - _t_post_start

    if cam_grabber: cam_grabber.stop()
    if cap: cap.release()


def process_video():
    """
    Stream generator — runs in the Flask response thread.
    Only encodes + yields the latest annotated frame at 20 FPS.
    Never runs YOLO — completely decoupled from inference speed.
    """
    global _inf_thread
    if _inf_thread is None or not _inf_thread.is_alive():
        _inf_stop.clear()
        _inf_thread = threading.Thread(target=_inference_loop,
                                       daemon=True, name="InferenceLoop")
        _inf_thread.start()
        print("[Stream] Inference thread started")

    STREAM_FPS  = 20
    frame_delay = 1.0 / STREAM_FPS
    blank = np.zeros((405,720,3), dtype=np.uint8)
    cv2.putText(blank,"Waiting for video...",(160,200),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(80,80,80),2)

    while True:
        deadline = time.time() + frame_delay
        with _latest_lock:
            frame = _latest_annotated
        out    = frame if frame is not None else blank
        ok,buf = cv2.imencode('.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,72])
        if ok:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')
        wait = deadline - time.time()
        if wait > 0: time.sleep(wait)

# ─────────────────────────────────────────────
# ⚠️ HTML_STREAM_A — THIS IS UNUSED / DEAD CODE.
# This variable is defined here (a ~350-line inline HTML template) but is
# never referenced anywhere else in this file — nothing renders it. The
# actual page the browser sees comes from templates/index.html, served
# via render_template() elsewhere. This block appears to be left over
# from earlier development. It's harmless to leave in place (Python just
# stores the string and moves on), but if you're trying to find where a
# particular piece of UI lives, this is NOT it — check templates/index.html
# instead. Safe to delete entirely if you want to clean up the file.
# ─────────────────────────────────────────────
HTML_STREAM_A = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Vehicle Recognition</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.1/socket.io.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg:#07090d; --surface:#0e1318; --panel:#131a22;
            --border:#1e2d3d; --accent:#00e5ff; --accent2:#39ff7e;
            --warn:#ffc107; --danger:#ff3d3d; --text:#cdd9e5; --dim:#3d5a70;
        }
        *,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
        html,body { height:100%; background:var(--bg); color:var(--text);
                    font-family:"Barlow Condensed",sans-serif; overflow:hidden; }
        body { display:flex; flex-direction:column; padding:8px; gap:8px; }

        /* Header */
        header { display:flex; align-items:center; justify-content:space-between;
                 background:var(--surface); border:1px solid var(--border);
                 border-radius:8px; padding:8px 18px; flex-shrink:0; }
        .logo { font-size:1.15rem; font-weight:700; letter-spacing:4px; color:var(--accent); }
        .logo span { color:var(--text); font-weight:400; }
        .header-right { display:flex; align-items:center; gap:10px; }
        .pill { display:flex; align-items:center; gap:6px; background:var(--panel);
                border:1px solid var(--border); padding:4px 12px; border-radius:20px;
                font-family:"Share Tech Mono",monospace; font-size:0.72rem; letter-spacing:1px; }
        .dot { width:7px; height:7px; border-radius:50%; background:var(--dim); }
        .dot.on { background:var(--accent2); box-shadow:0 0 8px var(--accent2);
                  animation:pulse 2s infinite; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
        #clock { color:var(--accent2); font-family:"Share Tech Mono",monospace;
                 font-size:0.78rem; }

        /* Layout */
        .main { display:flex; gap:8px; flex:1; min-height:0; }

        /* Left panel */
        .left { width:270px; flex-shrink:0; display:flex; flex-direction:column;
                gap:8px; overflow-y:auto; }
        .card { background:var(--surface); border:1px solid var(--border);
                border-radius:8px; padding:12px 14px; }
        .card-title { font-size:0.62rem; letter-spacing:3px; text-transform:uppercase;
                      color:var(--dim); font-family:"Share Tech Mono",monospace;
                      margin-bottom:10px; padding-bottom:6px;
                      border-bottom:1px solid var(--border); }

        /* Count grid */
        .count-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
        .count-box { background:var(--panel); border:1px solid var(--border);
                     border-radius:6px; padding:10px;
                     display:flex; align-items:center; gap:8px; }
        .count-icon { font-size:1.3rem; width:28px; text-align:center; }
        .count-val { font-size:1.7rem; font-weight:700; line-height:1; color:var(--accent); }
        .count-lbl { font-size:0.62rem; color:var(--dim); letter-spacing:1px;
                     text-transform:uppercase; }
        .total-row { display:flex; justify-content:space-between; align-items:baseline;
                     margin-top:8px; padding:8px 12px; background:var(--panel);
                     border-radius:6px; border:1px solid var(--border); }
        .total-lbl { font-size:0.68rem; letter-spacing:2px; color:var(--dim);
                     text-transform:uppercase; }
        .total-num { font-size:2.2rem; font-weight:700; color:var(--accent2); }

        /* Inputs */
        select, input { background:var(--panel); color:var(--text);
                        border:1px solid var(--border); border-radius:4px;
                        padding:6px 8px; font-size:0.85rem;
                        font-family:"Barlow Condensed",sans-serif;
                        outline:none; width:100%; }
        select:focus, input:focus { border-color:var(--accent); }
        .btn { background:var(--panel); border:1px solid var(--border); color:var(--text);
               border-radius:4px; padding:6px 14px; cursor:pointer;
               font-family:"Barlow Condensed",sans-serif; font-size:0.85rem;
               font-weight:600; letter-spacing:1px; transition:all .15s;
               width:100%; margin-top:6px; }
        .btn:hover { border-color:var(--accent); color:var(--accent); }
        .btn.primary { border-color:var(--accent); color:var(--accent); }
        .rtsp-row { display:flex; gap:6px; }
        .rtsp-row input { flex:1; }
        .rtsp-row .btn { width:auto; margin-top:0; padding:6px 12px; }

        /* Center */
        .center { flex:1; display:flex; flex-direction:column; gap:8px; min-width:0; }
        .video-wrap { flex:1; position:relative; background:#000;
                      border:1px solid var(--border); border-radius:8px; overflow:hidden; }
        .video-wrap img { width:100%; height:100%; object-fit:contain; display:block; }
        .corner { position:absolute; width:20px; height:20px;
                  border-color:var(--accent); border-style:solid;
                  opacity:.5; pointer-events:none; }
        .corner.tl { top:8px; left:8px; border-width:2px 0 0 2px; }
        .corner.tr { top:8px; right:8px; border-width:2px 2px 0 0; }
        .corner.bl { bottom:8px; left:8px; border-width:0 0 2px 2px; }
        .corner.br { bottom:8px; right:8px; border-width:0 2px 2px 0; }

        /* Right panel */
        .right { width:310px; flex-shrink:0; display:flex; flex-direction:column; gap:8px; }
        .summary-box { flex:1; overflow-y:auto; background:var(--panel);
                       border-radius:4px; padding:10px; min-height:0; }
        .summary-box pre { white-space:pre-wrap; font-family:"Share Tech Mono",monospace;
                           font-size:0.73rem; line-height:1.65; color:var(--text); }
        .summary-card { flex:1; display:flex; flex-direction:column; min-height:0; }
        audio { width:100%; border-radius:4px; outline:none; }
        .loader { border:3px solid var(--border); border-top-color:var(--accent);
                  border-radius:50%; width:16px; height:16px;
                  animation:spin 1s linear infinite; display:none; margin-left:8px; }
        .loader.on { display:inline-block; }
        @keyframes spin { to { transform:rotate(360deg); } }

        /* Nav */
        .nav { display:flex; gap:6px; flex-wrap:wrap; }
        .nav a { color:var(--dim); text-decoration:none; font-size:0.7rem;
                 letter-spacing:1px; padding:3px 10px; border:1px solid var(--border);
                 border-radius:3px; font-family:"Share Tech Mono",monospace;
                 transition:all .15s; }
        .nav a:hover, .nav a.active { color:var(--accent); border-color:var(--accent); }

        ::-webkit-scrollbar { width:3px; }
        ::-webkit-scrollbar-track { background:transparent; }
        ::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }
    </style>
    <script>
        const socket = io();
        let isAudioPlaying = false;

        // Clock
        setInterval(() => {
            const el = document.getElementById('clock');
            if (el) el.textContent = new Date().toTimeString().slice(0,8);
        }, 1000);

        // Vehicle count updates from Ollama
        socket.on('vehicle_count_update', d => {
            setText('cnt-total', d.vehicle_count);
            setText('cnt-car',   d.car);
            setText('cnt-van',   d.van);
            setText('cnt-moto',  d.motorcycle);
            setText('cnt-lorry', d.lorry);
            setText('cnt-bus',   d.bus);
        });

        // Summary update from Ollama
        socket.on('summary_update', d => {
            const el = document.querySelector('.summary-box pre');
            if (el) el.innerText = d.summary;
            document.getElementById('ai-loader').classList.remove('on');
        });

        // Audio ready
        socket.on('audio_ready', d => {
            if (isAudioPlaying) return;
            const a = document.getElementById('audio-player');
            if (a) {
                a.src = '/Audio/' + d.audio_file + '?t=' + Date.now();
                a.load();
                a.play().then(() => { isAudioPlaying = true; }).catch(() => {});
            }
        });

        function setText(id, val) {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        }

        function changeVideoSource() {
            const sel = document.getElementById('video-select');
            socket.emit('change_video_source', { source: sel.value });
            document.getElementById('ai-loader').classList.add('on');
            const el = document.querySelector('.summary-box pre');
            if (el) el.innerText = 'Analysing new source...';
        }

        function setRtsp() {
            const url = document.getElementById('rtsp-input').value.trim();
            if (!url) return;
            socket.emit('change_video_source', { source: url });
            document.getElementById('ai-loader').classList.add('on');
        }

        // Poll summary every 5s
        function updateSummary() {
            fetch('/get_summary').then(r => r.text()).then(t => {
                const el = document.querySelector('.summary-box pre');
                if (el && t) el.innerText = t;
            }).catch(() => {});
        }
        setInterval(updateSummary, 5000);
        updateSummary();

        window.onload = function () {
            const a   = document.getElementById('audio-player');
            const btn = document.getElementById('start-audio');
            if (a) {
                a.play().then(() => { isAudioPlaying = true; }).catch(() => {});
                if (btn) {
                    btn.addEventListener('click', () => {
                        a.play().then(() => {
                            isAudioPlaying = true;
                            btn.style.display = 'none';
                        });
                    });
                }
            }
        };
    </script>
</head>
<body>

<!-- Header -->
<header>
    <div class="logo">AI&nbsp;<span>VEHICLE</span>&nbsp;RECOGNITION</div>
    <div id="clock">00:00:00</div>
    <div class="header-right">
        <div class="pill"><div class="dot on"></div>YOLO LIVE</div>
        <div class="pill"><div class="dot on"></div>MINICPM-V</div>
    </div>
    <div class="nav">
        <a href="/" class="active">FULL</a>
        <a href="/stream_b">VIDEO</a>
        <a href="/stream_c">AUDIO</a>
        <a href="/stream_d">SUMMARY</a>
        <a href="/stream_e">COMPACT</a>
    </div>
</header>

<!-- Main -->
<div class="main">

    <!-- Left panel -->
    <div class="left">

        <!-- Counts -->
        <div class="card">
            <div class="card-title">Vehicle Count (Ollama)</div>
            <div class="count-grid">
                <div class="count-box">
                    <div class="count-icon">🚗</div>
                    <div>
                        <div class="count-val" id="cnt-car">0</div>
                        <div class="count-lbl">Car</div>
                    </div>
                </div>
                <div class="count-box">
                    <div class="count-icon">🚐</div>
                    <div>
                        <div class="count-val" id="cnt-van">0</div>
                        <div class="count-lbl">Van</div>
                    </div>
                </div>
                <div class="count-box">
                    <div class="count-icon">🏍️</div>
                    <div>
                        <div class="count-val" id="cnt-moto">0</div>
                        <div class="count-lbl">Motorcycle</div>
                    </div>
                </div>
                <div class="count-box">
                    <div class="count-icon">🚛</div>
                    <div>
                        <div class="count-val" id="cnt-lorry">0</div>
                        <div class="count-lbl">Lorry</div>
                    </div>
                </div>
                <div class="count-box" style="grid-column:span 2;">
                    <div class="count-icon">🚌</div>
                    <div>
                        <div class="count-val" id="cnt-bus">0</div>
                        <div class="count-lbl">Bus</div>
                    </div>
                </div>
            </div>
            <div class="total-row">
                <span class="total-lbl">Total Detected</span>
                <span class="total-num" id="cnt-total">0</span>
            </div>
        </div>

        <!-- Video source -->
        <div class="card">
            <div class="card-title">Video Source</div>
            <select id="video-select">
                {% for video in video_files %}
                <option value="videos/{{ video }}"
                    {% if video == current_video %}selected{% endif %}>
                    {{ video.lower().replace('.mp4','').replace('.avi','').replace('.mov','') }}
                </option>
                {% endfor %}
            </select>
            <button class="btn primary" onclick="changeVideoSource()">LOAD VIDEO</button>
        </div>

        <!-- RTSP -->
        <div class="card">
            <div class="card-title">RTSP Camera (Live)</div>
            <div class="rtsp-row">
                <input id="rtsp-input" placeholder="rtsp://user:pass@ip/stream">
                <button class="btn" style="width:auto;" onclick="setRtsp()">GO</button>
            </div>
        </div>

    </div>

    <!-- Center video -->
    <div class="center">
        <div class="video-wrap">
            <img src="{{ url_for('video_feed') }}" alt="Drone Feed">
            <div class="corner tl"></div>
            <div class="corner tr"></div>
            <div class="corner bl"></div>
            <div class="corner br"></div>
        </div>
    </div>

    <!-- Right panel -->
    <div class="right">

        <!-- AI Summary -->
        <div class="card summary-card">
            <div class="card-title" style="display:flex;align-items:center;">
                Ollama AI Analysis
                <div id="ai-loader" class="loader on"></div>
            </div>
            <div class="summary-box">
                <pre>{{ summary_content if summary_content else
'Waiting for minicpm-v analysis...\n\nMake sure Ollama is running:\n  ollama run minicpm-v\n\nBounding boxes = YOLO (real-time)\nCounts + details = Ollama (every 30 frames)' }}</pre>
            </div>
        </div>

        <!-- Audio -->
        <div class="card">
            <div class="card-title">Audio Report</div>
            {% if audio_exists %}
            <audio id="audio-player" controls>
                <source src="/Audio/{{ audio_file }}" type="audio/mpeg">
            </audio>
            <button id="start-audio" class="btn" style="margin-top:6px;">
                ▶ PLAY AUDIO
            </button>
            {% else %}
            <div style="font-size:0.75rem;color:var(--dim);
                        font-family:'Share Tech Mono',monospace;line-height:1.5;">
                Audio will appear after<br>first Ollama analysis...
            </div>
            {% endif %}
        </div>

    </div>
</div>

</body>
</html>
'''

# ─────────────────────────────────────────────
# Route helpers — shared setup used by more than one page route below,
# so each route doesn't have to repeat this same file-reading logic.
# ─────────────────────────────────────────────
def get_common_vars():
    summary_content = ""
    summary_path    = os.path.join(SUMMARY_DIR, "summary_current.txt")
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_content = f.read()
    video_files   = [os.path.basename(f) for f in get_video_files()]
    current_video = os.path.basename(VIDEO_SOURCE) if VIDEO_SOURCE else ""
    return summary_content, video_files, current_video

# ─────────────────────────────────────────────
# Routes — everything the browser can actually request. Two kinds:
#
# PAGE routes (return a full HTML page):
#   /  and /stream_a — the main dashboard (templates/index.html) — this
#                       is the one you actually use day-to-day
#   /stream_b .. /stream_e — lightweight alternate views of the SAME
#                       live data (just the video, just audio, just the
#                       text summary, etc.) — small standalone pages
#                       defined inline here, useful e.g. for a second
#                       screen showing only the video feed
#
# API routes (/api/..., return JSON, called by buttons via JavaScript):
#   /video_feed              — the actual MJPEG video stream (the <img>
#                               tag in index.html points here)
#   /get_summary              — plain-text AI description, polled by the
#                               summary panel
#   /api/mavlink_connect      — "Connect Pixhawk" button
#   /api/flight_mode          — current flight mode (fetched on page load)
#   /api/toggle_flight_mode   — the flight-mode button (STABILIZE<->RTL)
#   /api/track_vehicle        — the "Go to" button (starts continuous
#                               engagement — see _engagement_loop)
#   /api/cancel_engagement    — stops an active "Go to" engagement
#   /api/goto_vehicle         — older one-shot hover-above-vehicle command
#   /api/start_landing, /api/abort_landing — the "Landing" button
# ─────────────────────────────────────────────
@app.route('/')
@app.route('/stream_a')
def stream_a():
    summary_content, video_files, current_video = get_common_vars()
    return render_template('index.html',
        summary_content=summary_content,
        video_files=video_files,
        current_video=current_video,
        vehicle_count=vehicle_count)

@app.route('/stream_b')
def stream_b():
    return render_template_string('''
    <!DOCTYPE html><html><head><meta charset="UTF-8">
    <title>Vehicle Recognition - Video</title>
    <style>body{margin:0;background:#000;display:flex;flex-direction:column;
    align-items:center;justify-content:center;height:100vh;}
    img{max-width:100%;max-height:92vh;}
    .nav{position:fixed;bottom:10px;display:flex;gap:10px;}
    .nav a{color:#00e5ff;text-decoration:none;font-size:0.8rem;font-family:monospace;
           padding:3px 10px;border:1px solid #00e5ff;border-radius:3px;}
    </style></head><body>
    <img src="{{ url_for('video_feed') }}" alt="Video Feed">
    <div class="nav">
        <a href="/">Full</a><a href="/stream_c">Audio</a>
        <a href="/stream_d">Summary</a><a href="/stream_e">Compact</a>
    </div>
    </body></html>''')

@app.route('/stream_c')
def stream_c():
    _, audio_file, audio_exists, _, _ = get_common_vars()
    return render_template_string('''
    <!DOCTYPE html><html><head><meta charset="UTF-8">
    <title>Vehicle Recognition - Audio</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.1/socket.io.js"></script>
    <style>body{margin:0;background:#07090d;color:#cdd9e5;font-family:sans-serif;
    display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;}
    h2{color:#00e5ff;letter-spacing:3px;margin-bottom:24px;}
    audio{width:360px;}
    .nav{position:fixed;bottom:10px;display:flex;gap:10px;}
    .nav a{color:#00e5ff;text-decoration:none;font-size:0.8rem;font-family:monospace;
           padding:3px 10px;border:1px solid #1e2d3d;border-radius:3px;}
    </style>
    <script>
    const socket = io();
    socket.on('audio_ready', d => {
        const a = document.getElementById('ap');
        if (a) {
            a.src = '/Audio/' + d.audio_file + '?t=' + Date.now();
            a.load();
            a.play().catch(() => {});
        }
    });
    </script>
    </head><body>
    <h2>AUDIO REPORT</h2>
    {% if audio_exists %}
    <audio id="ap" controls>
        <source src="/Audio/{{ audio_file }}" type="audio/mpeg">
    </audio>
    {% else %}
    <p style="color:#3d5a70;">Waiting for first Ollama analysis...</p>
    {% endif %}
    <div class="nav">
        <a href="/">Full</a><a href="/stream_b">Video</a>
        <a href="/stream_d">Summary</a><a href="/stream_e">Compact</a>
    </div>
    </body></html>''', audio_file=audio_file, audio_exists=audio_exists)

@app.route('/stream_d')
def stream_d():
    summary_content, _, _, _, _ = get_common_vars()
    return render_template_string('''
    <!DOCTYPE html><html><head><meta charset="UTF-8">
    <title>Vehicle Recognition - Summary</title>
    <style>body{margin:0;background:#07090d;color:#cdd9e5;font-family:monospace;
    padding:30px;overflow-y:auto;}
    h2{color:#00e5ff;letter-spacing:3px;margin-bottom:20px;}
    pre{white-space:pre-wrap;font-size:0.88rem;line-height:1.7;
        background:#0e1318;padding:20px;border-radius:8px;border:1px solid #1e2d3d;}
    .nav{margin-top:20px;display:flex;gap:10px;}
    .nav a{color:#00e5ff;text-decoration:none;font-size:0.8rem;
           padding:3px 10px;border:1px solid #1e2d3d;border-radius:3px;}
    </style>
    <script>
    setInterval(() => {
        fetch('/get_summary').then(r => r.text()).then(t => {
            if (t) document.querySelector('pre').innerText = t;
        });
    }, 5000);
    </script>
    </head><body>
    <h2>AI VEHICLE ANALYSIS</h2>
    <pre>{{ summary_content if summary_content else 'Waiting for Ollama analysis...' }}</pre>
    <div class="nav">
        <a href="/">Full</a><a href="/stream_b">Video</a>
        <a href="/stream_c">Audio</a><a href="/stream_e">Compact</a>
    </div>
    </body></html>''', summary_content=summary_content)

@app.route('/stream_e')
def stream_e():
    summary_content, video_files, current_video = get_common_vars()
    return render_template_string('''
    <!DOCTYPE html><html><head><meta charset="UTF-8">
    <title>Vehicle Recognition - Compact</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.1/socket.io.js"></script>
    <style>
    body{margin:0;background:#07090d;color:#cdd9e5;font-family:sans-serif;
         height:100vh;display:flex;flex-direction:column;padding:8px;gap:8px;}
    .tabs{display:flex;gap:4px;}
    .tab{padding:6px 16px;background:#0e1318;border:1px solid #1e2d3d;
         border-radius:6px 6px 0 0;cursor:pointer;font-size:0.82rem;letter-spacing:1px;}
    .tab.active{background:#00e5ff;color:#000;border-color:#00e5ff;font-weight:700;}
    .panel{display:none;flex:1;flex-direction:column;}
    .panel.active{display:flex;}
    .vwrap{flex:1;background:#000;border-radius:0 8px 8px 8px;overflow:hidden;
           border:1px solid #1e2d3d;}
    .vwrap img{width:100%;height:100%;object-fit:contain;}
    pre{flex:1;overflow-y:auto;white-space:pre-wrap;font-size:0.8rem;line-height:1.6;
        background:#0e1318;padding:14px;border-radius:8px;border:1px solid #1e2d3d;
        font-family:monospace;}
    select{background:#0e1318;color:#cdd9e5;border:1px solid #1e2d3d;
           border-radius:4px;padding:5px 8px;font-size:0.85rem;}
    .bar{display:flex;align-items:center;gap:8px;padding:4px 0;font-size:0.8rem;}
    .nav{display:flex;gap:6px;flex-wrap:wrap;}
    .nav a{color:#3d5a70;text-decoration:none;font-size:0.7rem;padding:3px 8px;
           border:1px solid #1e2d3d;border-radius:3px;}
    .nav a:hover{color:#00e5ff;border-color:#00e5ff;}
    </style>
    <script>
    const socket = io();
    socket.on('summary_update', d => {
        const el = document.querySelector('#sum pre');
        if (el) el.innerText = d.summary;
    });
    socket.on('audio_ready', d => {
        const a = document.getElementById('ap');
        if (a) { a.src = '/Audio/' + d.audio_file + '?t=' + Date.now(); a.load(); }
    });
    function switchTab(id) {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        document.querySelector('[data-tab="'+id+'"]').classList.add('active');
        document.getElementById(id).classList.add('active');
    }
    function changeSource() {
        socket.emit('change_video_source',
            { source: document.getElementById('vs').value });
    }
    setInterval(() => {
        fetch('/get_summary').then(r => r.text()).then(t => {
            const el = document.querySelector('#sum pre');
            if (el && t) el.innerText = t;
        });
    }, 5000);
    document.addEventListener('DOMContentLoaded', () => switchTab('feed'));
    </script>
    </head><body>
    <div class="tabs">
        <div class="tab active" data-tab="feed" onclick="switchTab('feed')">AI FEED</div>
        <div class="tab" data-tab="sum"  onclick="switchTab('sum')">SUMMARY</div>
        <div class="tab" data-tab="aud"  onclick="switchTab('aud')">AUDIO</div>
    </div>
    <div id="feed" class="panel active">
        <div class="bar">
            <label style="font-size:0.72rem;color:#3d5a70;">SOURCE</label>
            <select id="vs" onchange="changeSource()">
                {% for v in video_files %}
                <option value="videos/{{ v }}"
                    {% if v == current_video %}selected{% endif %}>
                    {{ v.lower().replace('.mp4','') }}
                </option>
                {% endfor %}
            </select>
        </div>
        <div class="vwrap">
            <img src="{{ url_for('video_feed') }}" alt="Feed">
        </div>
    </div>
    <div id="sum" class="panel">
        <pre>{{ summary_content if summary_content else 'Waiting for analysis...' }}</pre>
    </div>
    <div id="aud" class="panel" style="align-items:center;justify-content:center;">
        {% if audio_exists %}
        <audio id="ap" controls style="width:320px;">
            <source src="/Audio/{{ audio_file }}" type="audio/mpeg">
        </audio>
        {% else %}
        <p style="color:#3d5a70;">No audio yet...</p>
        {% endif %}
    </div>
    <div class="nav">
        <a href="/">Full Dashboard</a>
        <a href="/stream_b">Video Only</a>
        <a href="/stream_c">Audio Only</a>
        <a href="/stream_d">Summary Only</a>
    </div>
    </body></html>
    ''', summary_content=summary_content, audio_file=audio_file,
        audio_exists=audio_exists, video_files=video_files,
        current_video=current_video)

@app.route('/video_feed')
def video_feed():
    return Response(process_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_summary')
def get_summary():
    path = os.path.join(SUMMARY_DIR, "summary_current.txt")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"[Summary] Serving {len(content)} chars to browser")
            return content
    return ""

@app.route('/logo.png')
def serve_logo():
    return send_from_directory('.', 'logo.png')

@app.route('/api/mavlink_connect', methods=['POST'])
def api_mavlink_connect():
    ok = connect_mavlink()
    return jsonify({"ok": ok, "status": mavlink_status})

@app.route('/api/flight_mode')
def api_flight_mode():
    """Lets the frontend fetch the current flight mode on page load, before
    any flight_mode_update socket event has fired yet."""
    return jsonify({"mode": get_flight_mode(), "sim": mavlink_sim})

def _engagement_loop(vid, requested_target_alt, stop_event):
    """
    Runs in a background thread. Every ENGAGE_INTERVAL_SEC seconds:
      1. Re-reads the vehicle's CURRENT tracked pixel position (cx, cy)
         from tracked_vehicles — this is what makes it work for a MOVING
         vehicle, since that position is updated every frame by the
         inference loop.
      2. Reads the drone's LIVE current altitude (not a static number) to
         scale the pixel->metres conversion correctly, since that altitude
         keeps changing as the drone descends toward the target.
      3. Re-runs the distance/bearing -> GPS pipeline and sends an updated
         GOTO command with a target altitude that ratchets down toward
         requested_target_alt — but NEVER below MIN_APPROACH_ALT_M.
    This makes the drone continuously close in horizontally AND descend,
    tracking a moving vehicle the whole way down, while a hard altitude
    floor prevents it from ever being commanded into a collision.
    If the vehicle disappears from the tracker for longer than
    ENGAGE_LOST_TIMEOUT seconds, the engagement stops automatically.
    """
    # Clamp once up front — whatever was requested, we never go below the floor.
    target_alt = max(MIN_APPROACH_ALT_M, requested_target_alt)
    if requested_target_alt < MIN_APPROACH_ALT_M:
        log_command("ENGAGE_ALT_CLAMPED", vehicle_id=vid,
                    detail=(f"requested {requested_target_alt}m is below the "
                            f"{MIN_APPROACH_ALT_M}m safety floor; clamped to "
                            f"{MIN_APPROACH_ALT_M}m"))

    last_seen_ts = time.time()
    last_cx = last_cy = None

    while not stop_event.is_set():
        trk = tracked_vehicles.get(str(vid))

        if trk is not None:
            last_seen_ts = time.time()
            last_cx, last_cy = trk["cx"], trk["cy"]
            vtype = trk["type"]
        elif last_cx is not None and (time.time() - last_seen_ts) <= ENGAGE_LOST_TIMEOUT:
            # Briefly occluded / not detected this frame — hold last known spot
            # instead of immediately aborting the engagement.
            vtype = "unknown"
        else:
            log_command("ENGAGE_STOP", vehicle_id=vid, detail="target lost")
            socketio.emit("engagement_status", {"active": False, "vid": vid, "reason": "lost"})
            break

        drone_lat, drone_lon, drone_heading, drone_alt = get_drone_state()

        # Use the drone's LIVE altitude (not the descend target) to scale
        # pixels->metres correctly — this matters a lot here since the
        # drone's real altitude is actively changing every cycle.
        north, east = pixel_to_ned(last_cx, last_cy, alt=drone_alt)
        distance, bearing = ned_to_distance_bearing(north, east, drone_heading)
        target_lat, target_lon = destination_point(drone_lat, drone_lon, distance, bearing)

        _sent_ok = send_goto_global(target_lat, target_lon, target_alt)
        log_command("ENGAGE_UPDATE", vehicle_id=vid, north=north, east=east,
                    altitude=target_alt, mode="REAL" if not mavlink_sim else "SIM",
                    detail=(f"type={vtype} lat={target_lat:.6f} lon={target_lon:.6f} "
                            f"dist={distance:.1f}m bear={bearing:.1f}deg "
                            f"drone_alt={drone_alt:.1f}m target_alt={target_alt:.1f}m"),
                    success=_sent_ok)
        socketio.emit("engagement_status", {
            "active": True, "vid": vid, "distance": round(distance, 1),
            "bearing": round(bearing, 1), "target_alt": target_alt,
            "target": {"lat": round(target_lat, 6), "lon": round(target_lon, 6)},
        })

        stop_event.wait(ENGAGE_INTERVAL_SEC)

    with engagement_lock:
        if engagement_state["vid"] == vid:
            engagement_state["active"] = False
            engagement_state["vid"] = None
            engagement_state["thread"] = None
            engagement_state["stop"] = None


@app.route('/api/track_vehicle', methods=['POST'])
def api_track_vehicle():
    """
    Start (or retarget) a CONTINUOUS engagement on a vehicle: descends and
    keeps closing in toward the vehicle's live position (works for moving
    vehicles too), down to a minimum safe altitude — see _engagement_loop()
    and MIN_APPROACH_ALT_M.
    """
    data = request.get_json()
    vid  = data.get("vid")
    # Default target is the safety floor itself ("go as close as is safe").
    requested_alt = float(data.get("target_altitude", MIN_APPROACH_ALT_M))

    if vid is None:
        print_result("Command", "Go to vehicle failed", False, detail="no vehicle id provided")
        return jsonify({"ok": False, "error": "vid is required"}), 400
    if str(vid) not in tracked_vehicles:
        print_result("Command", f"Go to vehicle {vid}", False, detail="vehicle not found / not currently visible")
        return jsonify({"ok": False, "error": f"Vehicle {vid} not found"}), 404

    clamped_alt = max(MIN_APPROACH_ALT_M, requested_alt)

    with engagement_lock:
        # Stop any existing engagement first (retargeting to a new vehicle).
        if engagement_state["active"] and engagement_state["thread"]:
            engagement_state["stop"].set()
            engagement_state["thread"].join(timeout=ENGAGE_INTERVAL_SEC + 1)

        stop_event = threading.Event()
        thread = threading.Thread(target=_engagement_loop, args=(str(vid), requested_alt, stop_event), daemon=True)
        engagement_state.update({"active": True, "vid": str(vid), "thread": thread, "stop": stop_event})
        thread.start()

    print_result("Command", f"Go to vehicle {vid}", True, detail=f"target altitude {clamped_alt:.1f}m")
    log_command("ENGAGE_START", vehicle_id=vid, altitude=clamped_alt,
                mode="REAL" if not mavlink_sim else "SIM",
                detail=f"requested_target_alt={requested_alt}m clamped_to={clamped_alt}m")
    return jsonify({
        "ok": True, "vid": vid, "engaged": True,
        "requested_target_altitude": requested_alt,
        "actual_target_altitude": clamped_alt,
        "min_approach_alt": MIN_APPROACH_ALT_M,
    })


@app.route('/api/cancel_engagement', methods=['POST'])
def api_cancel_engagement():
    """Stop the current continuous engagement, if any."""
    with engagement_lock:
        if not engagement_state["active"]:
            print_result("Command", "Stop engagement", False, detail="no active engagement to stop")
            return jsonify({"ok": True, "was_active": False})
        vid = engagement_state["vid"]
        engagement_state["stop"].set()

    print_result("Command", f"Stopped engagement on vehicle {vid}", True)
    log_command("ENGAGE_CANCEL", vehicle_id=vid or "")
    return jsonify({"ok": True, "was_active": True, "vid": vid, "sim": mavlink_sim})


@app.route('/api/toggle_flight_mode', methods=['POST'])
def api_toggle_flight_mode():
    """
    One button, two behaviors:
      - If the drone is NOT currently in STABILIZE -> switch to STABILIZE.
      - If the drone IS currently in STABILIZE      -> switch to RTL (Return to Launch).
    """
    current_mode = get_flight_mode()
    target_mode  = "RTL" if current_mode == "STABILIZE" else "STABILIZE"

    success = send_set_mode(target_mode)

    return jsonify({
        "ok":            success,
        "previous_mode": current_mode,
        "target_mode":   target_mode,
        "sim":           mavlink_sim,
    })


@app.route('/api/goto_vehicle', methods=['POST'])
def api_goto_vehicle():
    data   = request.get_json()
    alt    = float(data.get("altitude", 10.0))

    # Look up by persistent vehicle ID if provided
    vid = data.get("vid")
    if vid is not None:
        trk = tracked_vehicles.get(str(vid))
        if trk:
            cx    = trk["cx"]
            cy    = trk["cy"]
            vtype = trk["type"]
        else:
            return jsonify({"ok": False, "error": f"Vehicle {vid} not found"}), 404
    else:
        # Fallback to cx/cy if no vid
        vtype = data.get("type",   "unknown")
        cx    = int(data.get("cx", 480))
        cy    = int(data.get("cy", 270))

    north, east = pixel_to_ned(cx, cy, alt=alt)

    frame = data.get("frame", "gps")   # "gps" (default) or "local"

    if frame == "local":
        # Original behaviour: local NED offset, relative to the EKF origin.
        _sent_ok = send_goto(north, east, -alt)
        log_command("GOTO_VEHICLE", vehicle_id=vid or "", north=north, east=east,
                    altitude=alt, mode="REAL" if not mavlink_sim else "SIM",
                    detail=f"type={vtype} frame=local", success=_sent_ok)
        return jsonify({
            "ok":    _sent_ok,
            "frame": "local",
            "north": north,
            "east":  east,
            "type":  vtype,
            "vid":   vid,
            "sim":   mavlink_sim,
        })

    # GPS pipeline (default) — same method as the AIVL vessel locator:
    # own GPS + distance + bearing -> destination_point() -> target GPS,
    # then send an absolute-GPS MAVLink command instead of a local offset.
    drone_lat, drone_lon, drone_heading, _drone_alt = get_drone_state()
    distance, bearing = ned_to_distance_bearing(north, east, drone_heading)
    target_lat, target_lon = destination_point(drone_lat, drone_lon, distance, bearing)

    _sent_ok = send_goto_global(target_lat, target_lon, alt)
    log_command("GOTO_VEHICLE", vehicle_id=vid or "", north=north, east=east,
                altitude=alt, mode="REAL" if not mavlink_sim else "SIM",
                detail=(f"type={vtype} frame=gps lat={target_lat:.6f} lon={target_lon:.6f} "
                        f"dist={distance:.1f}m bear={bearing:.1f}deg "
                        f"drone_heading={drone_heading:.1f}deg"),
                success=_sent_ok)

    return jsonify({
        "ok":       _sent_ok,
        "frame":    "gps",
        "north":    north,
        "east":     east,
        "distance": round(distance, 1),
        "bearing":  round(bearing, 1),
        "drone":    {"lat": drone_lat, "lon": drone_lon, "heading": round(drone_heading, 1)},
        "target":   {"lat": round(target_lat, 6), "lon": round(target_lon, 6)},
        "type":     vtype,
        "vid":      vid,
        "sim":      mavlink_sim,
    })

@app.route('/api/mavlink_status')
def api_mavlink_status():
    return jsonify({
        "status":   mavlink_status,
        "sim":      mavlink_sim,
        "cmd_log":  drone_cmd_log,
    })

@app.route('/api/start_landing', methods=['POST'])
def api_start_landing():
    if not ARUCO_AVAILABLE:
        print_result("Landing", "Landing sequence NOT started", False, detail="cv2.aruco not available on this install")
        return jsonify({"ok": False, "error": "cv2.aruco not available on this install"}), 400
    if landing_state != "idle":
        print_result("Landing", "Landing sequence NOT started", False, detail=f"already in progress ({landing_state})")
        return jsonify({"ok": False, "error": f"Landing already in progress ({landing_state})"}), 409
    start_landing_sequence()
    print_result("Landing", "Landing sequence started", True)
    return jsonify({"ok": True, "state": landing_state})

@app.route('/api/abort_landing', methods=['POST'])
def api_abort_landing():
    abort_landing_sequence("Aborted by user")
    print_result("Landing", "Landing sequence aborted", True)
    return jsonify({"ok": True, "state": landing_state})

@app.route('/api/landing_status')
def api_landing_status():
    return jsonify({
        "state":    landing_state,
        "altitude": round(landing_alt, 2),
        "target":   landing_target,
        "log":      landing_log,
        "aruco_available": ARUCO_AVAILABLE,
    })

@socketio.on('change_video_source')
def handle_video_source_change(data):
    global VIDEO_SOURCE, video_source_changed, description_generated
    source = data['source']

    # Resolve special source tokens
    if source == '__cam0__':
        source = 0           # OpenCV camera index 0 = first USB/built-in camera
        label  = "Live Camera (index 0)"
    elif source.startswith('rtsp://') or source.startswith('http://'):
        label  = f"RTSP Stream: {source}"
    else:
        label  = os.path.basename(str(source))

    VIDEO_SOURCE          = source
    video_source_changed  = True
    description_generated = False
    clear_directories()
    log_session_event(f"Video source changed to: {label}")
    print(f"[Source] Changed to: {label}")

# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    video_files = get_video_files()
    if not video_files:
        print(f"No video files in '{VIDEO_DIR}'. Please add .mp4 files.")
        exit(1)
    VIDEO_SOURCE = video_files[0]
    print(f"[INIT] Starting with: {VIDEO_SOURCE}")
    print(f"[INFO] Make sure Ollama is running:  ollama run minicpm-v")
    connect_mavlink()

    print("=" * 60)
    print("[Access] How to open this app from another device:")
    lan_ip = get_lan_ip()
    if lan_ip:
        print(f"  Same WiFi network : http://{lan_ip}:5000")
    else:
        print(f"  Same WiFi network : could not detect LAN IP")
    vpn_ip = get_vpn_ip()
    if vpn_ip:
        print(f"  Same VPN          : http://{vpn_ip}:5000")
    else:
        print(f"  Same VPN          : not connected (no tun0 interface found)")
    print("  (Docker-internal addresses like 172.17.x.x / 192.168.116.x")
    print("   only work from a browser running ON this Jetson itself —")
    print("   not from a separate device.)")
    print("=" * 60)

    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
