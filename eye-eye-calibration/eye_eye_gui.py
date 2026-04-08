#!/usr/bin/env python3
"""
asti_cal_gui.py
Node: asti_cal_gui

Pure visualization GUI for the ASTI eye-eye calibration pipeline.
Zero heavy compute — all PnP lives in asti_board_pose_publisher.py.

Panels
------
  Left   : Leica DeckLink raw feed  + lightweight marker overlays (no PnP)
  Centre : Intel D405 raw feed      + lightweight marker overlays (no PnP)
  Right  : Synthetic world view     — board pose animated from
             /asti/board_pose/world_from_leica  (purple, Leica-derived)
             /asti/board_pose/d405              (gray ghost, direct D405)
           Two sub-views share the panel:
             Top  — D405-frame perspective projection (what D405 "sees" synthetically)
             Bot  — XY top-down orthographic projection

Status bar
----------
  Subscribes to /asti/calibration/status (std_msgs/String JSON) and
  displays: state, n_pairs, in-sample residual, hold-out residual, save path.

Topics subscribed
-----------------
  /decklink/camera/image_raw          sensor_msgs/Image
  /d405/color/image_raw               sensor_msgs/Image
  /asti/board_pose/leica              geometry_msgs/PoseStamped
  /asti/board_pose/d405               geometry_msgs/PoseStamped
  /asti/board_pose/world_from_leica   geometry_msgs/PoseStamped
  /asti/calibration/status            std_msgs/String

Keys: D=detail overlays  B=F_board axes  S=save frames  Q/ESC=quit
"""

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

import numpy as np
import cv2
import cv2.aruco as aruco
import threading
import time
import json
import os
from datetime import datetime
from scipy.spatial.transform import Rotation

from PyQt5 import QtWidgets, QtCore, QtGui

# ── Board geometry (identical to publisher/calibration nodes) ─────────────────
SQUARE_MM    = 4.0
MARKER_RATIO = 0.70
TAG_MM       = 24.0
CELL_MM      = 28.0
BOARD_W_M    = 3.0 * CELL_MM / 1000.0   # 84mm total board width
BOARD_H_M    = 3.0 * CELL_MM / 1000.0

# ── Style ─────────────────────────────────────────────────────────────────────
DARK     = "#141414"
DARK2    = "#1a1a1a"
DARK3    = "#111111"
BORDER   = "#444"
TEXT     = "#ddd"
TEXT_DIM = "#888"
GREEN    = "#4c9"
BLUE     = "#7af"
AMBER    = "#fa4"
PURPLE   = "#b8f"
MONO     = "monospace"
ACCENT   = "#2a6099"
ACCENT_H = "#3a80bb"
ACCENT_P = "#1a4070"

C_APRIL  = (0,   200, 255)   # BGR
C_CHARU  = (0,   255, 120)
C_BOARD  = (180,  80, 255)
C_AX_X   = (255,   0,   0)
C_AX_Y   = (0,   220,   0)
C_AX_Z   = (60,   60, 255)

GLOBAL_STYLE = f"""
    QWidget {{ background:{DARK}; color:{TEXT}; font-family:{MONO}; font-size:11px; }}
    QGroupBox {{
        border:1px solid {BORDER}; border-radius:3px; margin-top:6px;
        padding:6px; color:{TEXT_DIM}; font-size:11px;
    }}
    QGroupBox::title {{ subcontrol-origin:margin; left:8px; }}
    QLabel {{ color:{TEXT_DIM}; border:none; }}
    QPushButton {{
        background:{ACCENT}; color:#fff; border:none; font-size:12px;
        font-family:{MONO}; font-weight:bold; border-radius:4px; padding:0 12px;
    }}
    QPushButton:hover   {{ background:{ACCENT_H}; }}
    QPushButton:pressed {{ background:{ACCENT_P}; }}
    QPushButton:checked {{ background:#1e3a1e; color:{GREEN}; border:1px solid {GREEN}; }}
    QSlider::groove:horizontal {{ height:4px; background:{BORDER}; border-radius:2px; }}
    QSlider::handle:horizontal {{
        background:{BLUE}; width:12px; height:12px; margin:-4px 0; border-radius:6px;
    }}
    QSlider::sub-page:horizontal {{ background:{ACCENT}; border-radius:2px; }}
"""


# ── Geometry helpers ──────────────────────────────────────────────────────────
def _T44_from_pose_msg(pose):
    q = pose.orientation
    p = pose.position
    R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [p.x, p.y, p.z]
    return T


def _rvec_to_euler_deg(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        rx = np.degrees(np.arctan2( R[2, 1], R[2, 2]))
        ry = np.degrees(np.arctan2(-R[2, 0], sy))
        rz = np.degrees(np.arctan2( R[1, 0], R[0, 0]))
    else:
        rx = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        ry = np.degrees(np.arctan2(-R[2, 0], sy))
        rz = 0.0
    return rx, ry, rz


def _draw_axes(img, K, dist, rvec, tvec, length=0.025, thickness=2):
    pts3d = np.float32([[0,0,0],[length,0,0],[0,length,0],[0,0,length]])
    try:
        pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
        pts2d = pts2d.reshape(4, 2).astype(int)
    except Exception:
        return
    o = tuple(pts2d[0])
    h, w = img.shape[:2]
    if not (0 <= o[0] < w and 0 <= o[1] < h):
        return
    for pt, col in [(pts2d[1], C_AX_X), (pts2d[2], C_AX_Y), (pts2d[3], C_AX_Z)]:
        ep = (int(np.clip(pt[0], -9999, 9999)), int(np.clip(pt[1], -9999, 9999)))
        cv2.arrowedLine(img, o, ep, col, thickness, tipLength=0.25, line_type=cv2.LINE_AA)


def _fallback_K(w, h):
    f = max(w, h) * 0.85
    return np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)


# ── Lightweight overlay detector (no PnP, just marker detection) ──────────────
class OverlayDetector:
    """
    Detects AprilTags and ChArUco corners for visual overlay only.
    No board pose solved here — pose comes from /asti/board_pose/* topics.
    """
    def __init__(self):
        april_dict   = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        charuco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.charuco_board = aruco.CharucoBoard(
            (6, 6), SQUARE_MM / 1000.0,
            (SQUARE_MM * MARKER_RATIO) / 1000.0, charuco_dict,
        )
        params = aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin  = 5
        params.adaptiveThreshWinSizeMax  = 23
        params.adaptiveThreshWinSizeStep = 4
        params.minMarkerPerimeterRate    = 0.02
        params.maxMarkerPerimeterRate    = 0.5
        params.cornerRefinementMethod    = aruco.CORNER_REFINE_SUBPIX
        self.april_det   = aruco.ArucoDetector(april_dict, params)
        self.charuco_det = aruco.CharucoDetector(self.charuco_board)

    def draw(self, frame_rgb, show_detail, pose_T44, K, show_board):
        """
        Draw overlays onto a copy of frame_rgb (returned as BGR).
        pose_T44 : 4x4 from the pose topic — used only for axis drawing.
        """
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        out  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        n_april, n_charu, april_ids = 0, 0, []

        # ── AprilTag borders ──────────────────────────────────────────────
        corners, ids, _ = self.april_det.detectMarkers(gray)
        if ids is not None:
            n_april   = len(ids)
            april_ids = ids.flatten().tolist()
            if show_detail:
                aruco.drawDetectedMarkers(out, corners, ids, borderColor=C_APRIL)
                for i, c in enumerate(corners):
                    tag_id = int(ids[i][0])
                    cx_ = int(c[0][:, 0].mean())
                    cy_ = int(c[0][:, 1].mean())
                    cv2.putText(out, f"AT{tag_id}", (cx_ - 14, cy_ - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_APRIL, 1, cv2.LINE_AA)

        # ── ChArUco corner dots ────────────────────────────────────────────
        ch_corners, ch_ids, _, _ = self.charuco_det.detectBoard(gray)
        if ch_ids is not None and len(ch_ids) >= 2:
            n_charu = len(ch_ids)
            if show_detail:
                aruco.drawDetectedCornersCharuco(out, ch_corners, ch_ids,
                                                 cornerColor=C_CHARU)

        # ── F_board axes from topic pose ───────────────────────────────────
        if show_board and pose_T44 is not None:
            R   = pose_T44[:3, :3]
            t   = pose_T44[:3,  3]
            rvec, _ = cv2.Rodrigues(R)
            tvec    = t.reshape(3, 1)
            dist    = np.zeros(5)
            _draw_axes(out, K, dist, rvec, tvec, length=0.030, thickness=3)
            # Label at projected origin
            try:
                pt, _ = cv2.projectPoints(
                    np.float32([[0, 0, 0]]), rvec, tvec, K, dist
                )
                px, py = int(pt[0, 0, 0]), int(pt[0, 0, 1])
                h, w   = out.shape[:2]
                if 0 <= px < w and 0 <= py < h:
                    cv2.putText(out, "F_board", (px + 6, py - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_BOARD, 1, cv2.LINE_AA)
            except Exception:
                pass

        stats = {
            "n_april":   n_april,
            "april_ids": april_ids,
            "n_charu":   n_charu,
            "board_ok":  pose_T44 is not None,
        }
        return out, stats


# ── World view renderer ───────────────────────────────────────────────────────
# Board corner offsets in board frame (4 corners, metric)
_BOARD_CORNERS_3D = np.float32([
    [0,        0,        0],
    [BOARD_W_M, 0,        0],
    [BOARD_W_M, BOARD_H_M, 0],
    [0,        BOARD_H_M, 0],
])
# Interior grid lines at 1/3 and 2/3 of board size
_GRID_U = [BOARD_W_M / 3, 2 * BOARD_W_M / 3]
_GRID_V = [BOARD_H_M / 3, 2 * BOARD_H_M / 3]


def _project_board_to_canvas(T44, K_virt, canvas_w, canvas_h):
    """
    Project board corners + grid into a virtual camera defined by K_virt.
    T44: T_board_in_world (4x4).
    Returns list of 2D point arrays for polygon drawing, or None.
    """
    R   = T44[:3, :3]
    t   = T44[:3,  3]
    rvec, _ = cv2.Rodrigues(R)
    tvec    = t.reshape(3, 1)
    dist    = np.zeros(5)

    # Board outline
    try:
        pts2d, _ = cv2.projectPoints(_BOARD_CORNERS_3D, rvec, tvec, K_virt, dist)
        pts2d = pts2d.reshape(4, 2)
    except Exception:
        return None, None, None

    # Grid lines: horizontal
    h_lines = []
    for v in _GRID_V:
        seg3d = np.float32([[0, v, 0], [BOARD_W_M, v, 0]])
        try:
            p2d, _ = cv2.projectPoints(seg3d, rvec, tvec, K_virt, dist)
            h_lines.append(p2d.reshape(2, 2))
        except Exception:
            pass

    # Grid lines: vertical
    v_lines = []
    for u in _GRID_U:
        seg3d = np.float32([[u, 0, 0], [u, BOARD_H_M, 0]])
        try:
            p2d, _ = cv2.projectPoints(seg3d, rvec, tvec, K_virt, dist)
            v_lines.append(p2d.reshape(2, 2))
        except Exception:
            pass

    # Board origin axes
    axes3d = np.float32([[0,0,0],[0.025,0,0],[0,0.025,0],[0,0,0.025]])
    try:
        axes2d, _ = cv2.projectPoints(axes3d, rvec, tvec, K_virt, dist)
        axes2d = axes2d.reshape(4, 2)
    except Exception:
        axes2d = None

    return pts2d, (h_lines, v_lines), axes2d


def _make_world_canvas(w, h, T_leica, T_d405, view_mode):
    """
    Render the world view canvas (white background).
    view_mode: 'perspective' or 'topdown'
    T_leica, T_d405: T_board_in_world (4x4) or None.
    """
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)  # off-white

    # Grid
    grid_col = (210, 210, 210)
    step_px  = max(h, w) // 8
    for gx in range(0, w, step_px):
        cv2.line(canvas, (gx, 0), (gx, h), grid_col, 1)
    for gy in range(0, h, step_px):
        cv2.line(canvas, (0, gy), (w, gy), grid_col, 1)

    # Virtual camera intrinsics
    if view_mode == "perspective":
        # Looking roughly along +Z in D405 frame from a slight angle
        f = max(w, h) * 1.8
        K_virt = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)
        label  = "D405-frame perspective"
    else:
        # Top-down orthographic approximated by very long focal length
        # Scale: 1mm → ~3px
        scale  = min(w, h) / 0.25   # 250mm maps to panel height
        K_virt = np.array([[scale, 0, w/2], [0, scale, h/2], [0, 0, 1]], dtype=np.float64)
        label  = "XY top-down (ortho)"

    cv2.putText(canvas, label, (8, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    # World origin marker
    cv2.drawMarker(canvas, (w // 2, h // 2), (100, 100, 100),
                   cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
    cv2.putText(canvas, "D405 origin", (w // 2 + 8, h // 2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1, cv2.LINE_AA)

    # ── D405 direct (gray ghost) ──────────────────────────────────────────
    if T_d405 is not None:
        pts, lines, axes = _project_board_to_canvas(T_d405, K_virt, w, h)
        if pts is not None:
            poly = pts.astype(int)
            cv2.polylines(canvas, [poly], True, (160, 160, 160), 1, cv2.LINE_AA)
            h_lines, v_lines = lines
            for seg in h_lines + v_lines:
                p1 = tuple(seg[0].astype(int))
                p2 = tuple(seg[1].astype(int))
                cv2.line(canvas, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)
            # fill lightly
            cv2.fillPoly(canvas, [poly], (235, 235, 232))
            cv2.polylines(canvas, [poly], True, (160, 160, 160), 1, cv2.LINE_AA)
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            cv2.putText(canvas, "D405", (cx - 14, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1, cv2.LINE_AA)

    # ── Leica-derived (purple) ────────────────────────────────────────────
    if T_leica is not None:
        pts, lines, axes = _project_board_to_canvas(T_leica, K_virt, w, h)
        if pts is not None:
            poly      = pts.astype(int)
            purple_cv = (180, 80, 200)   # BGR
            cv2.polylines(canvas, [poly], True, purple_cv, 2, cv2.LINE_AA)
            h_lines, v_lines = lines
            for seg in h_lines + v_lines:
                p1 = tuple(seg[0].astype(int))
                p2 = tuple(seg[1].astype(int))
                cv2.line(canvas, p1, p2, purple_cv, 1, cv2.LINE_AA)
            # Board axes
            if axes is not None:
                o = tuple(axes[0].astype(int))
                for ep_f, col in [
                    (axes[1], C_AX_X),
                    (axes[2], C_AX_Y),
                    (axes[3], C_AX_Z),
                ]:
                    ep = tuple(ep_f.astype(int))
                    cv2.arrowedLine(canvas, o, ep, col, 2,
                                    tipLength=0.3, line_type=cv2.LINE_AA)
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            cv2.putText(canvas, "Leica", (cx - 18, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, purple_cv, 1, cv2.LINE_AA)

    # Legend
    lx, ly = w - 100, h - 36
    cv2.rectangle(canvas, (lx - 4, ly - 14), (w - 4, h - 4), (230, 230, 230), -1)
    cv2.rectangle(canvas, (lx - 4, ly - 14), (w - 4, h - 4), (180, 180, 180), 1)
    cv2.line(canvas, (lx, ly),      (lx + 20, ly),      (160, 160, 160), 2)
    cv2.putText(canvas, "D405 direct",   (lx + 24, ly + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.line(canvas, (lx, ly + 16), (lx + 20, ly + 16), (180, 80, 200), 2)
    cv2.putText(canvas, "Leica->world", (lx + 24, ly + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (140, 60, 160), 1, cv2.LINE_AA)

    return canvas


# ── Image decoder (shared) ────────────────────────────────────────────────────
def _decode_image_msg(msg):
    try:
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        enc = msg.encoding
        h, w = msg.height, msg.width
        if   enc == "rgb8":        return raw.reshape(h, w, 3)
        elif enc == "bgr8":        return raw.reshape(h, w, 3)[:, :, ::-1].copy()
        elif enc == "mono8":       return cv2.cvtColor(raw.reshape(h, w), cv2.COLOR_GRAY2RGB)
        elif enc in ("yuv422", "YUV422"):
            return cv2.cvtColor(raw.reshape(h, w, 2), cv2.COLOR_YUV2RGB_YUYV)
        elif enc == "uyvy":        return cv2.cvtColor(raw.reshape(h, w, 2), cv2.COLOR_YUV2RGB_UYVY)
        elif enc == "mono16":
            r16 = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
            return cv2.cvtColor((r16 >> 8).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif enc == "bayer_rggb8": return cv2.cvtColor(raw.reshape(h, w), cv2.COLOR_BayerBG2RGB)
        else:                      return raw.reshape(h, w, -1)[:, :, :3].copy()
    except Exception as e:
        rospy.logwarn_throttle(5.0, f"[asti_cal_gui] decode ({msg.encoding}): {e}")
        return None


# ── ROS image subscriber ──────────────────────────────────────────────────────
class ROSImageSub:
    def __init__(self, topic, name):
        self.name     = name
        self.frame    = None
        self._lock    = threading.Lock()
        self._t_last  = None
        self._fps_buf = []
        rospy.Subscriber(topic, Image, self._cb, queue_size=1, buff_size=2**24)

    def _cb(self, msg):
        arr = _decode_image_msg(msg)
        if arr is None:
            return
        now = time.time()
        with self._lock:
            self.frame   = arr
            self._t_last = now
        self._fps_buf.append(now)
        if len(self._fps_buf) > 60:
            self._fps_buf.pop(0)

    def get(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    @property
    def fps(self):
        b = self._fps_buf
        return 0.0 if len(b) < 2 else (len(b) - 1) / max(b[-1] - b[0], 1e-6)

    @property
    def streaming(self):
        return self._t_last is not None and (time.time() - self._t_last) < 2.0


# ── ROS pose subscriber ───────────────────────────────────────────────────────
class ROSPoseSub:
    """Subscribes to a PoseStamped topic and stores latest T44."""
    def __init__(self, topic):
        self._T44    = None
        self._lock   = threading.Lock()
        self._t_last = None
        rospy.Subscriber(topic, PoseStamped, self._cb, queue_size=5)

    def _cb(self, msg):
        T = _T44_from_pose_msg(msg.pose)
        now = time.time()
        with self._lock:
            self._T44    = T
            self._t_last = now

    def get(self):
        with self._lock:
            return self._T44.copy() if self._T44 is not None else None

    @property
    def active(self):
        return self._t_last is not None and (time.time() - self._t_last) < 2.0


# ── Pose readout widget ───────────────────────────────────────────────────────
class PoseReadout(QtWidgets.QFrame):
    _N_HIST = 80

    def __init__(self, cam_name, accent=BLUE, parent=None):
        super().__init__(parent)
        self.setFixedHeight(78)
        self.setStyleSheet(
            f"QFrame {{ background:{DARK3}; border-top:1px solid {BORDER}; }}"
        )
        self._accent = accent
        self._t_hist = [[], [], []]

        self._lbl_src = QtWidgets.QLabel("● --")
        self._lbl_src.setStyleSheet(
            f"color:#555; font-family:{MONO}; font-size:10px; border:none;"
        )
        self._lbl_t = QtWidgets.QLabel("t:  --")
        self._lbl_t.setStyleSheet(
            f"color:{accent}; font-family:{MONO}; font-size:11px; border:none;"
        )
        self._lbl_r = QtWidgets.QLabel("R:  --")
        self._lbl_r.setStyleSheet(
            f"color:{PURPLE}; font-family:{MONO}; font-size:11px; border:none;"
        )

        left = QtWidgets.QVBoxLayout()
        left.setSpacing(2)
        left.setContentsMargins(8, 5, 0, 5)
        left.addWidget(self._lbl_src)
        left.addWidget(self._lbl_t)
        left.addWidget(self._lbl_r)

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 6, 0)
        root.addLayout(left, stretch=1)

    def push(self, T44, label="--"):
        if T44 is None:
            self._lbl_src.setText("● no pose")
            self._lbl_src.setStyleSheet(
                f"color:#555; font-family:{MONO}; font-size:10px; border:none;"
            )
            self._lbl_t.setText("t:  --  --  --  mm")
            self._lbl_r.setText("R:  --  --  --  deg")
            return

        t   = T44[:3, 3] * 1000.0
        R   = T44[:3, :3]
        eu  = _rvec_to_euler_deg(R)

        self._lbl_src.setText(f"● F_board  [{label}]")
        self._lbl_src.setStyleSheet(
            f"color:{GREEN}; font-family:{MONO}; font-size:10px; border:none;"
        )
        self._lbl_t.setText(
            f"t:  x{t[0]:+7.1f}  y{t[1]:+7.1f}  z{t[2]:+7.1f}  mm"
        )
        self._lbl_r.setText(
            f"R:  rx{eu[0]:+6.1f}  ry{eu[1]:+6.1f}  rz{eu[2]:+6.1f}  deg"
        )


# ── Camera panel ──────────────────────────────────────────────────────────────
class CameraPanel(QtWidgets.QWidget):
    def __init__(self, title, cam_name, accent=BLUE, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QWidget {{ border:1px solid {BORDER}; background:{DARK2}; border-radius:2px; }}"
        )

        self._title_lbl = QtWidgets.QLabel(title)
        self._title_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self._title_lbl.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:10px; padding:2px; border:none;"
        )

        self._img_lbl = QtWidgets.QLabel()
        self._img_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self._img_lbl.setMinimumSize(480, 300)
        self._img_lbl.setStyleSheet(f"background:{DARK3}; border:none;")

        self._stream_lbl = QtWidgets.QLabel("● no stream")
        self._stream_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self._stream_lbl.setStyleSheet(
            f"color:#555; font-size:10px; padding:2px; border:none;"
        )

        self.pose_readout = PoseReadout(cam_name, accent=accent)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 0)
        lay.setSpacing(2)
        lay.addWidget(self._title_lbl)
        lay.addWidget(self._img_lbl, stretch=1)
        lay.addWidget(self._stream_lbl)
        lay.addWidget(self.pose_readout)

    def push_frame(self, bgr, streaming, fps, stats):
        if bgr is not None:
            rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg   = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
            self._img_lbl.setPixmap(
                QtGui.QPixmap.fromImage(qimg).scaled(
                    self._img_lbl.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
            )
        if streaming:
            n_at  = stats.get("n_april", 0)
            ids_s = str(stats.get("april_ids", []))
            n_ch  = stats.get("n_charu",  0)
            ok    = stats.get("board_ok",  False)
            col   = GREEN if ok else TEXT_DIM
            res   = stats.get("res", "--")
            self._stream_lbl.setText(
                f"● {fps:.1f}Hz  {res}  |  AT:{n_at}{ids_s}  Ch:{n_ch}"
                f"  {'F_board OK' if ok else 'no board'}"
            )
            self._stream_lbl.setStyleSheet(
                f"color:{col}; font-size:10px; padding:2px; border:none;"
            )
        else:
            self._stream_lbl.setText("● no stream")
            self._stream_lbl.setStyleSheet(
                f"color:#555; font-size:10px; padding:2px; border:none;"
            )


# ── World panel ───────────────────────────────────────────────────────────────
class WorldPanel(QtWidgets.QWidget):
    """
    Synthetic world view — two sub-panels stacked vertically:
      Top: D405-frame perspective projection
      Bot: XY top-down orthographic
    Board moves in real time from /asti/board_pose/* topics.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QWidget {{ border:1px solid {BORDER}; background:{DARK2}; border-radius:2px; }}"
        )

        title = QtWidgets.QLabel("World frame  (D405 origin)  —  board from Leica")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:10px; padding:2px; border:none;"
        )

        self._persp_lbl = QtWidgets.QLabel()
        self._persp_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self._persp_lbl.setStyleSheet(f"background:{DARK3}; border:none;")

        self._top_lbl = QtWidgets.QLabel()
        self._top_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self._top_lbl.setStyleSheet(f"background:{DARK3}; border:none;")

        self._stream_lbl = QtWidgets.QLabel("● /asti/board_pose/world_from_leica — waiting")
        self._stream_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self._stream_lbl.setStyleSheet(
            f"color:#555; font-size:10px; padding:2px; border:none;"
        )

        self.pose_readout = PoseReadout("world", accent=PURPLE)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 0)
        lay.setSpacing(2)
        lay.addWidget(title)
        lay.addWidget(self._persp_lbl, stretch=1)
        lay.addWidget(self._top_lbl,   stretch=1)
        lay.addWidget(self._stream_lbl)
        lay.addWidget(self.pose_readout)

    def push(self, T_leica, T_d405, active):
        w = self._persp_lbl.width()  or 400
        h = self._persp_lbl.height() or 200

        for lbl, mode in [(self._persp_lbl, "perspective"), (self._top_lbl, "topdown")]:
            lw = lbl.width()  or 400
            lh = lbl.height() or 200
            canvas = _make_world_canvas(lw, lh, T_leica, T_d405, mode)
            rgb    = canvas   # already BGR for cv2, convert for Qt
            qimg   = QtGui.QImage(canvas.data, lw, lh, lw * 3, QtGui.QImage.Format_RGB888)
            lbl.setPixmap(
                QtGui.QPixmap.fromImage(qimg).scaled(
                    lbl.size(), QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
            )

        col = PURPLE if active else "#555"
        self._stream_lbl.setText(
            "● /asti/board_pose/world_from_leica  active"
            if active else "● /asti/board_pose/world_from_leica  waiting"
        )
        self._stream_lbl.setStyleSheet(
            f"color:{col}; font-size:10px; padding:2px; border:none;"
        )


# ── Calibration status bar ────────────────────────────────────────────────────
class CalStatusBar(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(44)
        self.setStyleSheet(
            f"QFrame {{ background:{DARK3}; border:1px solid {BORDER}; border-radius:2px; }}"
        )

        def _seg(label):
            w = QtWidgets.QWidget()
            w.setStyleSheet("border:none;")
            lay = QtWidgets.QVBoxLayout(w)
            lay.setContentsMargins(10, 3, 10, 3)
            lay.setSpacing(1)
            lbl_top = QtWidgets.QLabel(label.upper())
            lbl_top.setStyleSheet(
                f"color:#555; font-size:9px; font-family:{MONO}; border:none;"
            )
            lbl_val = QtWidgets.QLabel("--")
            lbl_val.setStyleSheet(
                f"color:{TEXT_DIM}; font-size:11px; font-family:{MONO}; border:none;"
            )
            lay.addWidget(lbl_top)
            lay.addWidget(lbl_val)
            return w, lbl_val

        def _sep():
            s = QtWidgets.QFrame()
            s.setFrameShape(QtWidgets.QFrame.VLine)
            s.setStyleSheet(f"color:{BORDER}; border:none; background:{BORDER};")
            s.setFixedWidth(1)
            return s

        self._w_state,  self._v_state  = _seg("state")
        self._w_pairs,  self._v_pairs  = _seg("pairs")
        self._w_insamp, self._v_insamp = _seg("in-sample residual")
        self._w_holdout,self._v_holdout= _seg("hold-out residual")
        self._w_file,   self._v_file   = _seg("saved")

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(4, 0, 4, 0)
        root.setSpacing(0)
        for i, w in enumerate([
            self._w_state, self._w_pairs,
            self._w_insamp, self._w_holdout, self._w_file,
        ]):
            if i > 0:
                root.addWidget(_sep())
            root.addWidget(w)
        root.addStretch()

    def push(self, status_json):
        try:
            d = json.loads(status_json)
        except Exception:
            return

        state = d.get("state", "--")
        col   = GREEN if state == "calibrated" else (AMBER if state == "collecting" else TEXT_DIM)
        self._v_state.setText(f"● {state}")
        self._v_state.setStyleSheet(
            f"color:{col}; font-size:11px; font-family:{MONO}; border:none;"
        )

        n      = d.get("n_pairs", 0)
        target = d.get("n_cal_target", 20)
        self._v_pairs.setText(f"{n} / {target}")
        self._v_pairs.setStyleSheet(
            f"color:{BLUE}; font-size:11px; font-family:{MONO}; border:none;"
        )

        t_mm = d.get("t_mean_mm")
        r_deg = d.get("r_mean_deg")
        if t_mm is not None and state == "calibrated":
            self._v_insamp.setText(f"t {t_mm:.2f}mm  R {r_deg:.2f}°")
            col2 = GREEN if t_mm < 1.0 else AMBER
            self._v_insamp.setStyleSheet(
                f"color:{col2}; font-size:11px; font-family:{MONO}; border:none;"
            )
        else:
            self._v_insamp.setText("--")

        ho_n   = d.get("holdout_n", 0)
        if ho_n > 0 and t_mm is not None:
            self._v_holdout.setText(f"t {t_mm:.2f}mm  R {r_deg:.2f}°  (n={ho_n})")
            self._v_holdout.setStyleSheet(
                f"color:{GREEN}; font-size:11px; font-family:{MONO}; border:none;"
            )
        else:
            self._v_holdout.setText("--")

        path = d.get("save_path") or "--"
        if path != "--":
            fname = os.path.basename(path)
            self._v_file.setText(fname)
            self._v_file.setStyleSheet(
                f"color:{BLUE}; font-size:11px; font-family:{MONO}; border:none;"
            )
        else:
            self._v_file.setText("--")


# ── Main window ───────────────────────────────────────────────────────────────
class ASTICalGUI(QtWidgets.QWidget):
    # Qt signals for thread-safe updates from ROS callbacks
    _sig_status = QtCore.pyqtSignal(str)

    def __init__(self, leica_img, d405_img, pose_leica, pose_d405, pose_world):
        super().__init__()
        self._leica_img   = leica_img
        self._d405_img    = d405_img
        self._pose_leica  = pose_leica
        self._pose_d405   = pose_d405
        self._pose_world  = pose_world

        self._detector    = OverlayDetector()
        self._det_mode    = True
        self._board_mode  = True
        self._save_dir    = os.path.expanduser("~/asti-cal-frames")
        self._save_n      = 0
        self._last_bgr    = {"leica": None, "d405": None}

        self.setWindowTitle("ASTI — Calibration GUI")
        self.setStyleSheet(GLOBAL_STYLE)
        self._build_ui()

        # Status topic — thread-safe via signal
        rospy.Subscriber(
            "/asti/calibration/status", String,
            lambda msg: self._sig_status.emit(msg.data),
            queue_size=5,
        )
        self._sig_status.connect(self._cal_status_bar.push)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)   # 30 Hz display
        QtWidgets.QApplication.instance().aboutToQuit.connect(self._timer.stop)

    def _build_ui(self):
        self._panel_leica = CameraPanel(
            "Leica Proveo 8  (DeckLink)", "leica", accent=BLUE
        )
        self._panel_d405  = CameraPanel(
            "Intel D405  (RGB)", "d405", accent=BLUE
        )
        self._panel_world = WorldPanel()

        panels = QtWidgets.QHBoxLayout()
        panels.setSpacing(6)
        panels.addWidget(self._panel_leica, stretch=1)
        panels.addWidget(self._panel_d405,  stretch=1)
        panels.addWidget(self._panel_world, stretch=1)

        # ── Calibration status bar ─────────────────────────────────────────
        self._cal_status_bar = CalStatusBar()

        # ── Footer ────────────────────────────────────────────────────────
        self._btn_det = QtWidgets.QPushButton("Detail  [D]")
        self._btn_det.setCheckable(True)
        self._btn_det.setChecked(True)
        self._btn_det.setFixedHeight(34)
        self._btn_det.clicked.connect(
            lambda: setattr(self, "_det_mode", self._btn_det.isChecked())
        )

        self._btn_board = QtWidgets.QPushButton("F_board  [B]")
        self._btn_board.setCheckable(True)
        self._btn_board.setChecked(True)
        self._btn_board.setFixedHeight(34)
        self._btn_board.clicked.connect(
            lambda: setattr(self, "_board_mode", self._btn_board.isChecked())
        )

        self._btn_save = QtWidgets.QPushButton("Save  [S]")
        self._btn_save.setFixedHeight(34)
        self._btn_save.clicked.connect(self._save)

        self._lbl_footer = QtWidgets.QLabel("Waiting for streams...")
        self._lbl_footer.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:11px; border:none;"
        )

        footer = QtWidgets.QHBoxLayout()
        footer.setSpacing(8)
        footer.addWidget(self._btn_det)
        footer.addWidget(self._btn_board)
        footer.addWidget(self._btn_save)
        footer.addWidget(self._lbl_footer, stretch=1)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addLayout(panels, stretch=1)
        root.addWidget(self._cal_status_bar)
        root.addLayout(footer)

        QtWidgets.QShortcut(QtGui.QKeySequence("D"),      self, activated=self._btn_det.click)
        QtWidgets.QShortcut(QtGui.QKeySequence("B"),      self, activated=self._btn_board.click)
        QtWidgets.QShortcut(QtGui.QKeySequence("S"),      self, activated=self._save)
        QtWidgets.QShortcut(QtGui.QKeySequence("Q"),      self, activated=self.close)
        QtWidgets.QShortcut(QtGui.QKeySequence("Escape"), self, activated=self.close)

        self.resize(1820, 900)
        self.show()

    def _tick(self):
        # ── Leica panel ────────────────────────────────────────────────────
        leica_frame = self._leica_img.get()
        T_leica     = self._pose_leica.get()
        leica_bgr   = None
        leica_stats = {}

        if leica_frame is not None:
            h, w = leica_frame.shape[:2]
            # TODO: replace _fallback_K with real Leica intrinsics
            K_leica   = _fallback_K(w, h)
            leica_bgr, leica_stats = self._detector.draw(
                leica_frame, self._det_mode, T_leica, K_leica, self._board_mode
            )
            leica_stats["res"] = f"{w}×{h}"
            self._last_bgr["leica"] = leica_bgr

        self._panel_leica.push_frame(
            leica_bgr, self._leica_img.streaming, self._leica_img.fps, leica_stats
        )
        self._panel_leica.pose_readout.push(T_leica, label="leica topic")

        # ── D405 panel ─────────────────────────────────────────────────────
        d405_frame = self._d405_img.get()
        T_d405     = self._pose_d405.get()
        d405_bgr   = None
        d405_stats = {}

        if d405_frame is not None:
            h, w = d405_frame.shape[:2]
            # D405: K applied upstream
            K_d405   = _fallback_K(w, h)
            d405_bgr, d405_stats = self._detector.draw(
                d405_frame, self._det_mode, T_d405, K_d405, self._board_mode
            )
            d405_stats["res"] = f"{w}×{h}"
            self._last_bgr["d405"] = d405_bgr

        self._panel_d405.push_frame(
            d405_bgr, self._d405_img.streaming, self._d405_img.fps, d405_stats
        )
        self._panel_d405.pose_readout.push(T_d405, label="d405 topic")

        # ── World panel ────────────────────────────────────────────────────
        T_world = self._pose_world.get()
        self._panel_world.push(T_world, T_d405, self._pose_world.active)
        self._panel_world.pose_readout.push(T_world, label="world_from_leica")

        # ── Footer status ──────────────────────────────────────────────────
        leica_ok = self._leica_img.streaming
        d405_ok  = self._d405_img.streaming
        if leica_ok and d405_ok:
            self._lbl_footer.setText(
                "[D] detail  [B] F_board  [S] save  [Q] quit"
            )
            self._lbl_footer.setStyleSheet(
                f"color:{GREEN}; font-size:11px; border:none;"
            )
        else:
            missing = []
            if not leica_ok: missing.append("Leica")
            if not d405_ok:  missing.append("D405")
            self._lbl_footer.setText(f"Waiting: {', '.join(missing)}")
            self._lbl_footer.setStyleSheet(
                f"color:{AMBER}; font-size:11px; border:none;"
            )

    def _save(self):
        os.makedirs(self._save_dir, exist_ok=True)
        ts = datetime.now().strftime("%H%M%S_%f")[:10]
        for name, bgr in self._last_bgr.items():
            if bgr is not None:
                path = os.path.join(
                    self._save_dir, f"{name}_{ts}_{self._save_n:04d}.png"
                )
                cv2.imwrite(path, bgr)
        self._save_n += 1
        self._lbl_footer.setText(f"Saved → {self._save_dir}")
        self._lbl_footer.setStyleSheet(
            f"color:{GREEN}; font-size:11px; border:none;"
        )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rospy.init_node("asti_cal_gui", anonymous=False)

    leica_img  = ROSImageSub("/decklink/camera/image_raw", "Leica")
    d405_img   = ROSImageSub("/d405/color/image_raw",      "D405")
    pose_leica = ROSPoseSub("/asti/board_pose/leica")
    pose_d405  = ROSPoseSub("/asti/board_pose/d405")
    pose_world = ROSPoseSub("/asti/board_pose/world_from_leica")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = ASTICalGUI(leica_img, d405_img, pose_leica, pose_d405, pose_world)
    app.exec_()