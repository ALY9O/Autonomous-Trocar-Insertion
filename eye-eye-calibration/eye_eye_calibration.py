#!/usr/bin/env python3
"""
asti_eye_eye_calibration.py
Node: asti_eye_eye_calibration

Subscribes to the two board pose topics published by asti_board_pose_publisher.
Accumulates synchronized, board-still-filtered pose pairs, then solves:

    T_Leica_in_D405 = T_board_in_D405 · inv(T_board_in_Leica)

Transform chain (direct, not AX=XB — no robot motion needed):
    The board is a fixed rigid body; both cameras see it simultaneously.
    Each pair gives one independent estimate of T_Leica_in_D405.
    We average N_CAL_PAIRS estimates in SE(3) to get the calibration.

Validation strategy
-------------------
  Hold-out  : calibration computed on first N_CAL_PAIRS valid pairs;
              all subsequent pairs report a reprojection residual
              (translation error in mm, rotation error in deg) without
              touching the saved calibration.
  Live topic : /asti/board_pose/world_from_leica publishes
              T_board_in_D405_world = T_Leica_in_D405 · T_board_in_Leica
              This lets you eyeball agreement with /asti/board_pose/d405
              in rviz — useful for debug, not independent ground truth.

Topics
------
  Subscribed:
    /asti/board_pose/leica              geometry_msgs/PoseStamped
    /asti/board_pose/d405               geometry_msgs/PoseStamped

  Published:
    /asti/calibration/status            std_msgs/String  (JSON)
    /asti/board_pose/world_from_leica   geometry_msgs/PoseStamped  (post-cal only)

Saved file
----------
  ~/asti-calibration/eye_eye_<DDMONYYYY>.json
  e.g.  eye_eye_08APR2026.json

  JSON schema:
  {
    "date":             "08APR2026",
    "n_pairs":          20,
    "T_Leica_in_D405":  [[...], [...], [...], [...]],   // 4x4 row-major
    "t_mm":             [tx, ty, tz],
    "euler_deg_xyz":    [rx, ry, rz],
    "hold_out_residual_mm":  float,   // populated after first hold-out eval
    "hold_out_residual_deg": float
  }
"""

import rospy
import numpy as np
import json
import os
from datetime import datetime
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import String
import message_filters

# ── Parameters ────────────────────────────────────────────────────────────────
N_CAL_PAIRS   = 20       # pairs used to compute calibration
SYNC_SLOP_S   = 0.05     # ApproxTime slop (matches publisher node)
SYNC_QUEUE    = 10

# Board-still guard: reject a pair if the board moved more than these
# thresholds between the previous accepted pair in that camera.
STILL_T_MM    = 0.5      # translation threshold (mm)
STILL_R_DEG   = 0.3      # rotation threshold (deg)

SAVE_DIR      = os.path.expanduser("~/asti-calibration")


# ── SE(3) helpers ─────────────────────────────────────────────────────────────
def _T44_from_pose_msg(pose):
    """geometry_msgs/Pose → 4x4 SE(3) float64."""
    q = pose.orientation
    p = pose.position
    R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [p.x, p.y, p.z]
    return T


def _pose_stamped_from_T44(T44, stamp, frame_id):
    q = Rotation.from_matrix(T44[:3, :3]).as_quat()
    t = T44[:3, 3]
    msg = PoseStamped()
    msg.header.stamp    = stamp
    msg.header.frame_id = frame_id
    msg.pose = Pose(
        position    = Point(x=float(t[0]), y=float(t[1]), z=float(t[2])),
        orientation = Quaternion(x=float(q[0]), y=float(q[1]),
                                 z=float(q[2]), w=float(q[3])),
    )
    return msg


def _average_SE3(T_list):
    """
    Average a list of SE(3) matrices.
    Translation : arithmetic mean.
    Rotation    : quaternion mean with hemisphere flip (Markley eigenvector method).
    """
    ts   = np.array([T[:3, 3] for T in T_list])
    t_mean = ts.mean(axis=0)

    quats = np.array([Rotation.from_matrix(T[:3, :3]).as_quat() for T in T_list])
    ref   = quats[0]
    for i in range(1, len(quats)):
        if np.dot(quats[i], ref) < 0:
            quats[i] = -quats[i]
    q_mean = quats.mean(axis=0)
    q_mean /= np.linalg.norm(q_mean)

    T_out = np.eye(4)
    T_out[:3, :3] = Rotation.from_quat(q_mean).as_matrix()
    T_out[:3,  3] = t_mean
    return T_out


def _translation_error_mm(T_a, T_b):
    """L2 norm of translation difference in mm."""
    return np.linalg.norm((T_a[:3, 3] - T_b[:3, 3]) * 1000.0)


def _rotation_error_deg(T_a, T_b):
    """Geodesic rotation error between two SE(3) matrices in degrees."""
    dR    = T_a[:3, :3].T @ T_b[:3, :3]
    angle = np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(angle)


def _board_moved(T_prev, T_curr):
    """
    Returns True if the board moved more than STILL_T_MM or STILL_R_DEG
    between two consecutive poses in one camera.
    """
    if T_prev is None:
        return False   # first sample always accepted
    t_err = _translation_error_mm(T_prev, T_curr)
    r_err = _rotation_error_deg(T_prev,  T_curr)
    return t_err > STILL_T_MM or r_err > STILL_R_DEG


# ── Calibration solver ────────────────────────────────────────────────────────
def _solve_eye_eye(T_board_in_leica_list, T_board_in_d405_list):
    """
    Solve T_Leica_in_D405 from N synchronized pose pairs.

        T_Leica_in_D405 = T_board_in_D405[i] · inv(T_board_in_Leica[i])

    Each pair gives one estimate; we SE(3)-average all estimates.

    Returns
    -------
    T_cal     : (4, 4) float64  mean T_Leica_in_D405
    estimates : list of (4, 4)  individual estimates (for residual analysis)
    """
    estimates = []
    for T_l, T_d in zip(T_board_in_leica_list, T_board_in_d405_list):
        T_est = T_d @ np.linalg.inv(T_l)
        estimates.append(T_est)

    T_cal = _average_SE3(estimates)
    return T_cal, estimates


def _per_pair_residuals(T_cal, T_board_in_leica_list, T_board_in_d405_list):
    """
    For each pair compute:
      T_board_in_D405_predicted = T_cal · T_board_in_Leica
    and compare to directly observed T_board_in_D405.

    Returns arrays of t_err_mm and r_err_deg per pair.
    """
    t_errs, r_errs = [], []
    for T_l, T_d in zip(T_board_in_leica_list, T_board_in_d405_list):
        T_pred = T_cal @ T_l
        t_errs.append(_translation_error_mm(T_pred, T_d))
        r_errs.append(_rotation_error_deg(T_pred,  T_d))
    return np.array(t_errs), np.array(r_errs)


# ── ROS node ──────────────────────────────────────────────────────────────────
class EyeEyeCalibrationNode:
    def __init__(self):
        rospy.init_node("asti_eye_eye_calibration", anonymous=False)

        # ── Accumulation buffers ───────────────────────────────────────────
        # Cal set: first N_CAL_PAIRS valid pairs
        self._cal_leica  = []   # list of T44
        self._cal_d405   = []

        # Hold-out: everything after calibration is solved
        self._ho_leica   = []
        self._ho_d405    = []

        # Board-still guard: last accepted pose per camera
        self._prev_leica = None
        self._prev_d405  = None

        # State
        self._T_cal         = None   # solved T_Leica_in_D405
        self._cal_done      = False
        self._save_path     = None
        self._ho_t_mean_mm  = None
        self._ho_r_mean_deg = None

        # ── Publishers ─────────────────────────────────────────────────────
        self._pub_status = rospy.Publisher(
            "/asti/calibration/status", String, queue_size=5, latch=True
        )
        self._pub_world_leica = rospy.Publisher(
            "/asti/board_pose/world_from_leica", PoseStamped, queue_size=5
        )

        # ── Synchronized subscribers ───────────────────────────────────────
        leica_sub = message_filters.Subscriber(
            "/asti/board_pose/leica", PoseStamped, queue_size=10
        )
        d405_sub = message_filters.Subscriber(
            "/asti/board_pose/d405", PoseStamped, queue_size=10
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [leica_sub, d405_sub],
            queue_size=SYNC_QUEUE,
            slop=SYNC_SLOP_S,
        )
        self._sync.registerCallback(self._cb)

        self._publish_status("waiting", n_pairs=0)
        rospy.loginfo(
            f"[asti_eye_eye_cal] ready — collecting {N_CAL_PAIRS} pairs "
            f"(still guard: t<{STILL_T_MM}mm r<{STILL_R_DEG}deg)"
        )

    # ── Callback ───────────────────────────────────────────────────────────
    def _cb(self, leica_msg, d405_msg):
        T_l = _T44_from_pose_msg(leica_msg.pose)
        T_d = _T44_from_pose_msg(d405_msg.pose)

        # Board-still guard — reject if board moved in either camera
        leica_moved = _board_moved(self._prev_leica, T_l)
        d405_moved  = _board_moved(self._prev_d405,  T_d)

        if leica_moved or d405_moved:
            who = []
            if leica_moved: who.append("leica")
            if d405_moved:  who.append("d405")
            rospy.logdebug_throttle(
                1.0, f"[asti_eye_eye_cal] board-still reject ({', '.join(who)} moved)"
            )
            self._prev_leica = T_l
            self._prev_d405  = T_d
            return

        self._prev_leica = T_l
        self._prev_d405  = T_d

        # ── Accumulate ────────────────────────────────────────────────────
        if not self._cal_done:
            self._cal_leica.append(T_l)
            self._cal_d405.append(T_d)

            n = len(self._cal_leica)
            self._publish_status("collecting", n_pairs=n)
            rospy.loginfo_throttle(
                2.0, f"[asti_eye_eye_cal] collecting {n}/{N_CAL_PAIRS}"
            )

            if n >= N_CAL_PAIRS:
                self._solve_and_save(leica_msg.header.stamp)

        else:
            # Hold-out accumulation + live residual
            self._ho_leica.append(T_l)
            self._ho_d405.append(T_d)
            self._update_holdout_residual()

            # Derive and publish board in world from Leica observation
            T_world = self._T_cal @ T_l
            self._pub_world_leica.publish(
                _pose_stamped_from_T44(T_world, leica_msg.header.stamp, "d405_color_optical")
            )

    # ── Solver ────────────────────────────────────────────────────────────
    def _solve_and_save(self, stamp):
        rospy.loginfo("[asti_eye_eye_cal] solving T_Leica_in_D405 ...")
        T_cal, estimates = _solve_eye_eye(self._cal_leica, self._cal_d405)
        self._T_cal    = T_cal
        self._cal_done = True

        # In-sample residuals
        t_errs, r_errs = _per_pair_residuals(T_cal, self._cal_leica, self._cal_d405)

        t_mean = float(t_errs.mean())
        r_mean = float(r_errs.mean())
        t_max  = float(t_errs.max())
        r_max  = float(r_errs.max())

        euler = Rotation.from_matrix(T_cal[:3, :3]).as_euler("xyz", degrees=True)
        t_mm  = (T_cal[:3, 3] * 1000.0).tolist()

        rospy.loginfo(
            f"[asti_eye_eye_cal] SOLVED\n"
            f"  t = [{t_mm[0]:+.2f}, {t_mm[1]:+.2f}, {t_mm[2]:+.2f}] mm\n"
            f"  R = [{euler[0]:+.2f}, {euler[1]:+.2f}, {euler[2]:+.2f}] deg (XYZ)\n"
            f"  in-sample residual  t: mean={t_mean:.2f}mm max={t_max:.2f}mm\n"
            f"                      R: mean={r_mean:.2f}deg max={r_max:.2f}deg"
        )

        # ── Save JSON ──────────────────────────────────────────────────────
        date_str  = datetime.now().strftime("%d%b%Y").upper()   # 08APR2026
        os.makedirs(SAVE_DIR, exist_ok=True)
        fname     = f"eye_eye_{date_str}.json"
        fpath     = os.path.join(SAVE_DIR, fname)

        payload = {
            "date":                 date_str,
            "n_pairs":              N_CAL_PAIRS,
            "T_Leica_in_D405":      T_cal.tolist(),
            "t_mm":                 t_mm,
            "euler_deg_xyz":        euler.tolist(),
            "insample_t_mean_mm":   t_mean,
            "insample_t_max_mm":    t_max,
            "insample_r_mean_deg":  r_mean,
            "insample_r_max_deg":   r_max,
            # populated later by hold-out frames
            "holdout_t_mean_mm":    None,
            "holdout_r_mean_deg":   None,
        }

        with open(fpath, "w") as f:
            json.dump(payload, f, indent=2)

        self._save_path = fpath
        rospy.loginfo(f"[asti_eye_eye_cal] saved → {fpath}")

        self._publish_status(
            "calibrated",
            n_pairs=N_CAL_PAIRS,
            t_mean_mm=t_mean,
            r_mean_deg=r_mean,
            save_path=fpath,
        )

    # ── Hold-out residual ─────────────────────────────────────────────────
    def _update_holdout_residual(self):
        if len(self._ho_leica) < 2:
            return

        t_errs, r_errs = _per_pair_residuals(
            self._T_cal, self._ho_leica, self._ho_d405
        )
        self._ho_t_mean_mm  = float(t_errs.mean())
        self._ho_r_mean_deg = float(r_errs.mean())

        # Update JSON on disk with current hold-out numbers
        if self._save_path and os.path.exists(self._save_path):
            try:
                with open(self._save_path, "r") as f:
                    payload = json.load(f)
                payload["holdout_t_mean_mm"]  = self._ho_t_mean_mm
                payload["holdout_r_mean_deg"] = self._ho_r_mean_deg
                payload["holdout_n"]          = len(self._ho_leica)
                with open(self._save_path, "w") as f:
                    json.dump(payload, f, indent=2)
            except Exception as e:
                rospy.logwarn_throttle(10.0, f"[asti_eye_eye_cal] JSON update failed: {e}")

        rospy.loginfo_throttle(
            5.0,
            f"[asti_eye_eye_cal] hold-out n={len(self._ho_leica)} "
            f"t={self._ho_t_mean_mm:.2f}mm  R={self._ho_r_mean_deg:.2f}deg"
        )

        self._publish_status(
            "calibrated",
            n_pairs=N_CAL_PAIRS,
            t_mean_mm=self._ho_t_mean_mm,
            r_mean_deg=self._ho_r_mean_deg,
            save_path=self._save_path,
            holdout_n=len(self._ho_leica),
        )

    # ── Status publisher ──────────────────────────────────────────────────
    def _publish_status(self, state, n_pairs=0, t_mean_mm=None,
                        r_mean_deg=None, save_path=None, holdout_n=0):
        payload = {
            "state":        state,           # "waiting" | "collecting" | "calibrated"
            "n_pairs":      n_pairs,
            "n_cal_target": N_CAL_PAIRS,
            "t_mean_mm":    t_mean_mm,
            "r_mean_deg":   r_mean_deg,
            "holdout_n":    holdout_n,
            "save_path":    save_path,
        }
        self._pub_status.publish(String(data=json.dumps(payload)))

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        EyeEyeCalibrationNode().spin()
    except rospy.ROSInterruptException:
        pass