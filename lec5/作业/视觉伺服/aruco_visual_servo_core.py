"""Pure vision and coordinate math for the Week 6 simple visual servo."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


DICTIONARY_NAME = "DICT_4X4_50"
MARKER_ID = 0
MARKER_SIZE_M = 0.05


@dataclass(frozen=True)
class MarkerObservation:
    marker_id: int
    corners_px: np.ndarray
    center_px: np.ndarray
    rvec: np.ndarray
    tvec_m: np.ndarray
    reprojection_rms_px: float

    @property
    def depth_m(self):
        return float(self.tvec_m[2])


@dataclass(frozen=True)
class ServoDecision:
    action: str
    camera_delta_m: np.ndarray
    done: bool
    explanation: str


class ArucoTracker:
    def __init__(self, camera_matrix, dist_coeffs, marker_id=MARKER_ID,
                 marker_size_m=MARKER_SIZE_M):
        self.K = np.asarray(camera_matrix, dtype=float)
        self.dist = np.asarray(dist_coeffs, dtype=float)
        self.marker_id = int(marker_id)
        self.marker_size_m = float(marker_size_m)
        if self.K.shape != (3, 3) or not np.isfinite(self.K).all():
            raise ValueError("相机内参矩阵无效")
        if not np.isfinite(self.dist).all() or not 0 < self.marker_size_m < 1:
            raise ValueError("畸变参数或 ArUco 尺寸无效")
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY_NAME))
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def detect(self, frame):
        if frame is None or frame.ndim != 3:
            raise ValueError("需要一张 BGR 彩色图像")
        marker_corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None:
            return None
        matches = np.flatnonzero(ids.reshape(-1) == self.marker_id)
        if not len(matches):
            return None
        corners = np.asarray(marker_corners[int(matches[0])], dtype=np.float32).reshape(4, 2)
        half = self.marker_size_m / 2
        object_points = np.array([
            [-half, half, 0], [half, half, 0],
            [half, -half, 0], [-half, -half, 0],
        ], dtype=np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            object_points, corners, self.K, self.dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            return None
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, self.K, self.dist)
        rms = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - corners) ** 2, axis=1))))
        if float(tvec.reshape(3)[2]) <= 0:
            return None
        return MarkerObservation(
            marker_id=self.marker_id,
            corners_px=corners,
            center_px=corners.mean(axis=0),
            rvec=np.asarray(rvec, dtype=float).reshape(3),
            tvec_m=np.asarray(tvec, dtype=float).reshape(3),
            reprojection_rms_px=rms,
        )


def choose_step(observation: MarkerObservation | None, image_size,
                step_m=0.01, target_depth_m=0.20, depth_tolerance_m=0.01,
                central_fraction=0.20):
    """Choose one fixed 1 cm camera-frame move using the lecture's 40/20/40 zones."""
    if observation is None:
        return ServoDecision("wait", np.zeros(3), False, "未检测到 ID 0")
    width, height = map(float, image_size)
    if width <= 0 or height <= 0 or not 0 < central_fraction < 1:
        raise ValueError("图像尺寸或中央区域比例无效")
    low = (1 - central_fraction) / 2
    high = 1 - low
    x, y = observation.center_px
    violations = {
        "left": (low * width - x) / width,
        "right": (x - high * width) / width,
        "up": (low * height - y) / height,
        "down": (y - high * height) / height,
    }
    action, amount = max(violations.items(), key=lambda item: item[1])
    if amount > 0:
        directions = {
            "left": np.array([-1., 0., 0.]),
            "right": np.array([1., 0., 0.]),
            "up": np.array([0., -1., 0.]),
            "down": np.array([0., 1., 0.]),
        }
        labels = {"left": "向相机左侧移动", "right": "向相机右侧移动",
                  "up": "向相机上方移动", "down": "向相机下方移动"}
        return ServoDecision(action, directions[action] * step_m, False,
                             f"ArUco 位于中央区域外：{labels[action]} {step_m*100:.0f} cm")
    if observation.depth_m > target_depth_m + depth_tolerance_m:
        return ServoDecision("forward", np.array([0., 0., step_m]), False,
                             f"已居中，向前 {step_m*100:.0f} cm；当前距离约 {observation.depth_m*100:.1f} cm")
    if observation.depth_m < target_depth_m - depth_tolerance_m:
        return ServoDecision("too_close", np.zeros(3), True,
                             f"已居中但距离 {observation.depth_m*100:.1f} cm，小于目标距离")
    return ServoDecision("done", np.zeros(3), True,
                         f"已居中并到达目标距离 {observation.depth_m*100:.1f} cm")


def camera_delta_to_base(T_base_end_effector, T_end_effector_camera, camera_delta_m):
    T_BE = np.asarray(T_base_end_effector, dtype=float)
    T_EC = np.asarray(T_end_effector_camera, dtype=float)
    delta = np.asarray(camera_delta_m, dtype=float)
    if T_BE.shape != (4, 4) or T_EC.shape != (4, 4) or delta.shape != (3,):
        raise ValueError("坐标变换或位移维度无效")
    if not np.isfinite(T_BE).all() or not np.isfinite(T_EC).all() or not np.isfinite(delta).all():
        raise ValueError("坐标变换包含无效数值")
    return (T_BE[:3, :3] @ T_EC[:3, :3]) @ delta


def draw_overlay(frame, observation, decision, central_fraction=0.20):
    output = frame.copy()
    height, width = output.shape[:2]
    low = (1 - central_fraction) / 2
    high = 1 - low
    color = (45, 205, 130)
    cv2.rectangle(output, (int(low * width), int(low * height)),
                  (int(high * width), int(high * height)), color, 2)
    if observation is not None:
        corners = np.rint(observation.corners_px).astype(int)
        cv2.polylines(output, [corners], True, (0, 220, 255), 3, cv2.LINE_AA)
        center = tuple(np.rint(observation.center_px).astype(int))
        cv2.drawMarker(output, center, (0, 80, 255), cv2.MARKER_CROSS, 24, 3)
        cv2.putText(output, f"ID {observation.marker_id}  Z {observation.depth_m*100:.1f} cm",
                    (max(8, center[0] - 100), max(28, center[1] - 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(output, "ACTION: " + decision.action.upper(), (18, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 2, cv2.LINE_AA)
    return output
