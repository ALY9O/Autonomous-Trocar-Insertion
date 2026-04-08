#!/usr/bin/env python3
"""
asti_board_pose_publisher.py
Node: asti_board_pose_publisher

Detects the composite calibration board (4x AprilTag36h11 corners + central
ChArUco 6x4) in both the Leica DeckLink RGB and Intel D405 RGB streams.

Publishes PoseStamped only on successful PnP — no message means no detection.
D405 is the frame-rate bottleneck; ApproximateTimeSynchronizer pairs frames
within SYNC_SLOP_S of each other.

Topics
------
  Subscribed:
    /decklink/camera/image_raw    sensor_msgs/Image
    /d405/color/image_raw         sensor_msgs/Image
    /d405/color/camera_info       sensor_msgs/CameraInfo  (one-shot, unregisters after first msg)

  Published:
    /asti/board_pose/leica        geometry_msgs/PoseStamped   (F_board in Leica frame)
    /asti/board_pose/d405         geometry_msgs/PoseStamped   (F_board in D405 frame)

Intrinsics
----------
  Leica : fallback K (f = 0.85 * max(w,h)) — swap in real intrinsics once calibrated
  D405  : pulled from /d405/color/camera_info at startup (real K + dist)
"""

import rospy
import numpy as np
import cv2
import cv2.aruco as aruco
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
import message_filters
from scipy.spatial.transform import Rotation

# ── Board geometry (matches composite_board_tracker_ros.py exactly) ───────────
SQUARE_MM    = 4.0
MARKER_RATIO = 0.70
TAG_MM       = 24.0
CELL_MM      = 28.0

def _t44(tx_mm, ty_mm):
    T = np.eye(4)
    T[0, 3] = tx_mm / 1000.0
    T[1, 3] = ty_mm / 1000.0
    return T

T_BOARD_TO_CHARUCO = _t44(1.5 * CELL_MM, 1.5 * CELL_MM)
T_BOARD_TO_AT = {
    0: _t44(0.5 * CELL_MM, 0.5 * CELL_MM),
    1: _t44(2.5 * CELL_MM, 0.5 * CELL_MM),
    2: _t44(0.5 * CELL_MM, 2.5 * CELL_MM),
    3: _t44(2.5 * CELL_MM, 2.5 * CELL_MM),
    4: _t44(1.5 * CELL_MM, 0.5 * CELL_MM),
    5: _t44(0.5 * CELL_MM, 1.5 * CELL_MM),
    6: _t44(2.5 * CELL_MM, 1.5 * CELL_MM),
    7: _t44(1.5 * CELL_MM, 2.5 * CELL_MM),
}
T_CHARUCO_TO_BOARD = np.linalg.inv(T_BOARD_TO_CHARUCO)
T_AT_TO_BOARD      = {k: np.linalg.inv(v) for k, v in T_BOARD_TO_AT.items()}

# ── Sync ──────────────────────────────────────────────────────────────────────
SYNC_SLOP_S = 0.05    # 1 DeckLink frame period at 20 Hz
SYNC_QUEUE  = 10

# ── Intrinsics ────────────────────────────────────────────────────────────────
_DIST_ZERO = np.zeros(5, dtype=np.float64)

def _fallback_K(w, h):
    """
    Reasonable stub for Leica until real intrinsics are calibrated.
    f = 0.85 * max(w, h) is a typical approximation for a moderately
    telephoto lens with no prior knowledge.
    TODO: replace with real Leica K loaded from calibration file.
    """
    f = max(w, h) * 0.85
    return np.array([[f, 0, w / 2],
                     [0, f, h / 2],
                     [0, 0,     1]], dtype=np.float64)


# ── Geometry helpers ──────────────────────────────────────────────────────────
def _T44_from_rvec_tvec(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = tvec.ravel()
    return T


def _pose_stamped(T44, stamp, frame_id):
    """Convert 4x4 SE(3) matrix to geometry_msgs/PoseStamped."""
    R = T44[:3, :3]
    t = T44[:3,  3]
    q = Rotation.from_matrix(R).as_quat()   # xyzw
    msg = PoseStamped()
    msg.header.stamp    = stamp
    msg.header.frame_id = frame_id
    msg.pose = Pose(
        position    = Point(x=float(t[0]), y=float(t[1]), z=float(t[2])),
        orientation = Quaternion(x=float(q[0]), y=float(q[1]),
                                 z=float(q[2]), w=float(q[3])),
    )
    return msg


# ── Detector ──────────────────────────────────────────────────────────────────
class CompositeBoardDetector:
    """
    Stateless PnP detector for the composite calibration board.
    Priority: ChArUco (more corners, more stable) > AprilTag average.
    """

    def __init__(self):
        april_dict   = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        charuco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

        self.charuco_board = aruco.CharucoBoard(
            (6, 6),
            SQUARE_MM / 1000.0,
            (SQUARE_MM * MARKER_RATIO) / 1000.0,
            charuco_dict,
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

        self._tag_half = (TAG_MM / 1000.0) / 2.0
        self._tag_obj  = np.float32([
            [-self._tag_half, -self._tag_half, 0],
            [ self._tag_half, -self._tag_half, 0],
            [ self._tag_half,  self._tag_half, 0],
            [-self._tag_half,  self._tag_half, 0],
        ])

    def detect(self, frame_rgb, K, dist):
        """
        Parameters
        ----------
        frame_rgb : (H, W, 3) uint8  RGB image
        K         : (3, 3) float64   camera intrinsic matrix
        dist      : (N,)   float64   distortion coefficients

        Returns
        -------
        T_cam_to_board : (4, 4) float64  SE(3), or None on detection failure
        source         : str  e.g. 'Ch+8c' or 'AT×3' or None
        """
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # ── AprilTags ──────────────────────────────────────────────────────
        corners, ids, _ = self.april_det.detectMarkers(gray)
        tag_board_results = []

        if ids is not None:
            for i, c in enumerate(corners):
                tag_id = int(ids[i][0])
                ok, rvec, tvec = cv2.solvePnP(
                    self._tag_obj, c[0], K, dist,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if not ok or tag_id not in T_AT_TO_BOARD:
                    continue
                T_cam_to_board = _T44_from_rvec_tvec(rvec, tvec) @ T_AT_TO_BOARD[tag_id]
                tag_board_results.append(T_cam_to_board)

        # ── ChArUco ────────────────────────────────────────────────────────
        ch_corners, ch_ids, _, _ = self.charuco_det.detectBoard(gray)

        if ch_ids is not None and len(ch_ids) >= 6:
            obj_pts = self.charuco_board.getChessboardCorners()[ch_ids.flatten()]
            ok, rvec_ch, tvec_ch = cv2.solvePnP(obj_pts, ch_corners, K, dist)
            if ok:
                T_cam_to_board = _T44_from_rvec_tvec(rvec_ch, tvec_ch) @ T_CHARUCO_TO_BOARD
                return T_cam_to_board, f"Ch+{len(ch_ids)}c"

        # ── AprilTag fallback ──────────────────────────────────────────────
        if tag_board_results:
            T_avg = _average_SE3(tag_board_results)
            return T_avg, f"AT×{len(tag_board_results)}"

        return None, None


def _average_SE3(T_list):
    """
    Average a list of SE(3) matrices.
    Translation : arithmetic mean.
    Rotation    : quaternion mean (Markley et al. eigenvector method).
    """
    ts     = np.array([T[:3, 3] for T in T_list])
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


# ── Image decoding ────────────────────────────────────────────────────────────
def _decode_image_msg(msg):
    try:
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        enc = msg.encoding
        h, w = msg.height, msg.width
        if   enc == "rgb8":
            return raw.reshape(h, w, 3)
        elif enc == "bgr8":
            return raw.reshape(h, w, 3)[:, :, ::-1].copy()
        elif enc == "mono8":
            return cv2.cvtColor(raw.reshape(h, w), cv2.COLOR_GRAY2RGB)
        elif enc in ("yuv422", "YUV422"):
            return cv2.cvtColor(raw.reshape(h, w, 2), cv2.COLOR_YUV2RGB_YUYV)
        elif enc == "uyvy":
            return cv2.cvtColor(raw.reshape(h, w, 2), cv2.COLOR_YUV2RGB_UYVY)
        elif enc == "mono16":
            r16 = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
            return cv2.cvtColor((r16 >> 8).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif enc == "bayer_rggb8":
            return cv2.cvtColor(raw.reshape(h, w), cv2.COLOR_BayerBG2RGB)
        else:
            arr = raw.reshape(h, w, -1)
            return arr[:, :, :3].copy()
    except Exception as e:
        rospy.logwarn_throttle(5.0, f"[board_pose_pub] decode ({msg.encoding}): {e}")
        return None


# ── ROS node ──────────────────────────────────────────────────────────────────
class BoardPosePublisher:
    def __init__(self):
        rospy.init_node("asti_board_pose_publisher", anonymous=False)

        self._detector = CompositeBoardDetector()

        # D405 intrinsics — populated on first camera_info message
        self._K_d405    = None
        self._dist_d405 = None
        self._info_sub  = rospy.Subscriber(
            "/d405/color/camera_info", CameraInfo,
            self._camera_info_cb, queue_size=1,
        )
        rospy.loginfo("[board_pose_pub] waiting for D405 camera_info ...")

        # Publishers
        self._pub_leica = rospy.Publisher(
            "/asti/board_pose/leica", PoseStamped, queue_size=5
        )
        self._pub_d405 = rospy.Publisher(
            "/asti/board_pose/d405", PoseStamped, queue_size=5
        )

        # Leica intrinsics — loaded once at startup
        _cal = np.load("../leica-camera-calibration/output/calibration_leica_08APR2026.npz")
        self._K_leica    = _cal["K"]
        self._dist_leica = np.zeros(5, dtype=np.float64)  # telephoto — dist negligible
        rospy.loginfo(
            f"[board_pose_pub] Leica K loaded  "
            f"fx={self._K_leica[0,0]:.1f}  fy={self._K_leica[1,1]:.1f}  "
            f"cx={self._K_leica[0,2]:.1f}  cy={self._K_leica[1,2]:.1f}"
        )


        # ApproximateTime synchronizer — D405 is the bottleneck
        leica_sub = message_filters.Subscriber(
            "/decklink/camera/image_raw", Image, queue_size=4, buff_size=2**24
        )
        d405_sub = message_filters.Subscriber(
            "/d405/color/image_raw", Image, queue_size=4, buff_size=2**24
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [leica_sub, d405_sub],
            queue_size=SYNC_QUEUE,
            slop=SYNC_SLOP_S,
        )
        self._sync.registerCallback(self._cb)

        self._n_pairs    = 0
        self._n_leica_ok = 0
        self._n_d405_ok  = 0

        rospy.loginfo("[asti_board_pose_publisher] ready — waiting for synced frames")

    def _camera_info_cb(self, msg):
        self._K_d405    = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        self._dist_d405 = np.array(msg.D, dtype=np.float64)
        self._info_sub.unregister()
        rospy.loginfo(
            f"[board_pose_pub] D405 K loaded  "
            f"fx={self._K_d405[0,0]:.1f}  fy={self._K_d405[1,1]:.1f}  "
            f"cx={self._K_d405[0,2]:.1f}  cy={self._K_d405[1,2]:.1f}  "
            f"dist={np.round(self._dist_d405, 4).tolist()}"
        )

    def _cb(self, leica_msg, d405_msg):
        # Gate on D405 K being ready
        if self._K_d405 is None:
            rospy.logwarn_throttle(2.0, "[board_pose_pub] D405 K not yet received — skipping")
            return

        self._n_pairs += 1

        leica_frame = _decode_image_msg(leica_msg)
        d405_frame  = _decode_image_msg(d405_msg)

        # ── Leica detection ────────────────────────────────────────────────
        if leica_frame is not None:
            h, w = leica_frame.shape[:2]
            # TODO: replace _fallback_K with real Leica intrinsics once calibrated
            K_leica = _fallback_K(w, h)
            T_leica, src_leica = self._detector.detect(leica_frame, K_leica, _DIST_ZERO)

            if T_leica is not None:
                self._n_leica_ok += 1
                msg = _pose_stamped(T_leica, leica_msg.header.stamp, "leica_optical")
                self._pub_leica.publish(msg)
        else:
            T_leica, src_leica = None, None

        # ── D405 detection ─────────────────────────────────────────────────
        if d405_frame is not None:
            T_d405, src_d405 = self._detector.detect(
                d405_frame, self._K_d405, self._dist_d405
            )

            if T_d405 is not None:
                self._n_d405_ok += 1
                msg = _pose_stamped(T_d405, d405_msg.header.stamp, "d405_color_optical")
                self._pub_d405.publish(msg)
        else:
            T_d405, src_d405 = None, None

        # ── Throttled status log ───────────────────────────────────────────
        if self._n_pairs % 30 == 0:
            rospy.loginfo(
                f"[board_pose_pub] pairs={self._n_pairs} "
                f"leica={self._n_leica_ok}({src_leica or '--'}) "
                f"d405={self._n_d405_ok}({src_d405 or '--'})"
            )

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        BoardPosePublisher().spin()
    except rospy.ROSInterruptException:
        pass