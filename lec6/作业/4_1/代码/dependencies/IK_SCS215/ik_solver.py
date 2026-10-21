"""Position IK from URDF, using only NumPy and the Python standard library.

Internal units: radians and metres. No motor communication takes place here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _vector(text: str | None, default: str = "0 0 0") -> np.ndarray:
    values = np.asarray([float(v) for v in (text or default).split()])
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("Expected three finite numbers")
    return values


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' formula; axis must be a unit vector."""
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def _origin(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = _vector(element.get("xyz"))
        roll, pitch, yaw = _vector(element.get("rpy"))
        # URDF fixed-axis RPY: Rz(yaw) Ry(pitch) Rx(roll).
        transform[:3, :3] = (
            axis_rotation(np.array([0., 0., 1.]), yaw)
            @ axis_rotation(np.array([0., 1., 0.]), pitch)
            @ axis_rotation(np.array([1., 0., 0.]), roll)
        )
    return transform


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


class URDFArm:
    """A fixed-base serial chain selected by its base and tool link names.

    Visual/collision meshes are not loaded; this is a kinematic model, not a
    collision checker. Sibling joints (such as gripper opening) are excluded.
    """

    def __init__(self, path: str | Path, base: str = "base_link",
                 tool: str = "gripper_frame_link",
                 calibrated_limits: dict[str, tuple[float, float]] | None = None):
        root = ET.parse(path).getroot()
        links = {node.attrib["name"] for node in root.findall("link")}
        if base not in links or tool not in links:
            raise ValueError("Unknown base or tool link")
        by_child = {}
        for node in root.findall("joint"):
            child = node.find("child").attrib["link"]
            if child in by_child:
                raise ValueError("URDF child has multiple parents")
            by_child[child] = node
        chain, seen, current = [], set(), tool
        while current != base:
            if current in seen or current not in by_child:
                raise ValueError("Tool is not connected to the requested base")
            seen.add(current)
            node = by_child[current]
            kind = node.attrib["type"]
            if kind not in {"fixed", "revolute", "prismatic"}:
                raise ValueError(f"Unsupported joint type: {kind}")
            axis_node = node.find("axis")
            axis = _vector(axis_node.get("xyz") if axis_node is not None else None,
                           "1 0 0")
            lower = upper = 0.0
            if kind != "fixed":
                norm = np.linalg.norm(axis)
                if norm < 1e-12:
                    raise ValueError("Moving joint has a zero axis")
                axis /= norm
                limit = node.find("limit")
                if limit is None:
                    raise ValueError("Finite limits are required for each moving joint")
                lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
            chain.append(Joint(node.attrib["name"], kind, _origin(node.find("origin")),
                               axis, lower, upper))
            current = node.find("parent").attrib["link"]
        self.chain = list(reversed(chain))
        moving = [joint for joint in self.chain if joint.kind != "fixed"]
        self.names = [joint.name for joint in moving]
        self.lower = np.array([joint.lower for joint in moving])
        self.upper = np.array([joint.upper for joint in moving])
        for name, (lower, upper) in (calibrated_limits or {}).items():
            if name not in self.names:
                raise ValueError(f"Unknown chain joint: {name}")
            if not np.isfinite([lower, upper]).all() or lower > upper:
                raise ValueError(f"Invalid calibration limits: {name}")
            index = self.names.index(name)
            self.lower[index] = max(self.lower[index], lower)
            self.upper[index] = min(self.upper[index], upper)
        if (not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all()
                or np.any(self.lower > self.upper)):
            raise ValueError("Invalid or disjoint joint limits")
        if not self.names:
            raise ValueError("The selected chain has no moving joints")

    def configuration(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != self.lower.shape or not np.isfinite(q).all():
            raise ValueError(f"Expected {len(self.names)} finite joint values")
        return q

    def fk_jacobian(self, q) -> tuple[np.ndarray, np.ndarray]:
        """Return tool transform and base-frame position Jacobian (3 x n)."""
        q = self.configuration(q)
        transform, axes, origins, kinds = np.eye(4), [], [], []
        index = 0
        for joint in self.chain:
            transform = transform @ joint.origin
            if joint.kind == "fixed":
                continue
            axes.append(transform[:3, :3] @ joint.axis)
            origins.append(transform[:3, 3].copy())
            kinds.append(joint.kind)
            motion = np.eye(4)
            if joint.kind == "revolute":
                motion[:3, :3] = axis_rotation(joint.axis, q[index])
            else:
                motion[:3, 3] = joint.axis * q[index]
            transform = transform @ motion
            index += 1
        tip = transform[:3, 3]
        jacobian = np.column_stack([
            np.cross(axis, tip - origin) if kind == "revolute" else axis
            for axis, origin, kind in zip(axes, origins, kinds)
        ])
        return transform, jacobian

    def fk(self, q) -> np.ndarray:
        return self.fk_jacobian(q)[0]


def _box_damped_step(jacobian, error, lower, upper, damping):
    """Solve min ||J dq-e||² + damping² ||dq||² within a box.

    Cyclic coordinate descent solves this tiny convex quadratic problem.
    Each coordinate update has an exact scalar minimizer. Unlike clipping an
    unconstrained IK result, the other coordinates are reoptimized at limits.
    """
    hessian = jacobian.T @ jacobian + damping**2 * np.eye(jacobian.shape[1])
    rhs = jacobian.T @ error
    delta = np.clip(np.linalg.solve(hessian, rhs), lower, upper)
    for _ in range(60):
        previous = delta.copy()
        for i in range(len(delta)):
            delta[i] = np.clip(
                delta[i] + (rhs[i] - hessian[i] @ delta) / hessian[i, i],
                lower[i], upper[i],
            )
        if np.max(np.abs(delta - previous)) < 1e-8:
            break
    return delta


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    position: np.ndarray
    error_m: float
    success: bool
    iterations: int
    reason: str


class PositionIK:
    """Local, bounded damped-least-squares position IK; angles are radians.

    Does not constrain orientation or certify global non-reachability.
    """

    def __init__(self, arm: URDFArm, tolerance_m: float = 1e-5,
                 max_iterations: int = 100, damping: float = 0.003,
                 iteration_step: float = 0.15):
        if (not np.isfinite([tolerance_m, damping, iteration_step]).all()
                or min(tolerance_m, damping, iteration_step) <= 0
                or max_iterations < 1):
            raise ValueError("Solver settings must be finite and positive")
        self.arm = arm
        self.tolerance = tolerance_m
        self.max_iterations = max_iterations
        self.damping = damping
        self.iteration_step = iteration_step

    def solve(self, target, initial, max_joint_delta: float | None = None) -> IKResult:
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Target must be a finite XYZ vector in metres")
        q = self.arm.configuration(initial).copy()
        lower, upper = self.arm.lower.copy(), self.arm.upper.copy()
        if np.any(q < lower - 1e-10) or np.any(q > upper + 1e-10):
            raise ValueError("Initial configuration is outside the calibrated/model limits")
        if max_joint_delta is not None:
            if not np.isfinite(max_joint_delta) or max_joint_delta <= 0:
                raise ValueError("max_joint_delta must be finite and positive")
            lower = np.maximum(lower, q - max_joint_delta)
            upper = np.minimum(upper, q + max_joint_delta)
        q = np.clip(q, lower, upper)
        reason = "iteration_limit"
        iterations = 0
        for iterations in range(self.max_iterations):
            transform, jacobian = self.arm.fk_jacobian(q)
            error = target - transform[:3, 3]
            distance = np.linalg.norm(error)
            if distance <= self.tolerance:
                reason = "converged"
                break
            delta = _box_damped_step(
                jacobian, error,
                np.maximum(lower - q, -self.iteration_step),
                np.minimum(upper - q, self.iteration_step),
                self.damping,
            )
            # The linearized step must also improve the actual nonlinear FK.
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                candidate = np.clip(q + scale * delta, lower, upper)
                new_distance = np.linalg.norm(target - self.arm.fk(candidate)[:3, 3])
                if new_distance < distance - 1e-12:
                    q = candidate
                    accepted = True
                    break
            if not accepted:
                reason = "stalled_or_constrained"
                break
        position = self.arm.fk(q)[:3, 3].copy()
        error_m = float(np.linalg.norm(target - position))
        success = error_m <= self.tolerance
        return IKResult(q, position, error_m, success, iterations + 1,
                        "converged" if success else reason)


@dataclass(frozen=True)
class StepResult:
    q: np.ndarray
    position: np.ndarray
    requested_position: np.ndarray
    mode: str
    message: str


def cartesian_step(ik: PositionIK, current_q, delta_xyz,
                   max_joint_delta: float = np.deg2rad(3)) -> StepResult:
    """Try an exact local step, a shorter collinear step, then a projection.

    Intended for small Cartesian increments. Returns a candidate only; a motor
    controller must independently check feedback, faults and collision risks.
    """
    q = ik.arm.configuration(current_q)
    if np.any(q < ik.arm.lower) or np.any(q > ik.arm.upper):
        raise ValueError("Current configuration is outside the calibrated/model limits")
    delta = np.asarray(delta_xyz, dtype=float)
    if delta.shape != (3,) or not np.isfinite(delta).all():
        raise ValueError("delta_xyz must contain three finite numbers")
    start = ik.arm.fk(q)[:3, 3]
    length = np.linalg.norm(delta)
    requested = start + delta
    if length < 1e-12:
        return StepResult(q.copy(), start.copy(), requested, "idle", "保持位置")
    direction = delta / length
    full = ik.solve(requested, q, max_joint_delta=max_joint_delta)
    for fraction in (1.0, 0.5, 0.25, 0.125):
        result = full if fraction == 1 else ik.solve(
            start + fraction * delta, q, max_joint_delta=max_joint_delta)
        if result.success:
            mode = "exact" if fraction == 1 else "shortened"
            message = "目标步可达" if fraction == 1 else "IK 无解（原目标步）：已沿原方向缩短步长"
            return StepResult(result.q, result.position, requested, mode, message)
    # Soft position IK provides a locally nearest candidate. Accept it only if
    # actual FK confirms forward progress, bounded displacement and improvement.
    movement = full.position - start
    progress = float(movement @ direction)
    if (progress > max(1e-7, ik.tolerance * 0.1)
            and np.linalg.norm(movement) <= length + ik.tolerance
            and full.error_m < length - 1e-8):
        return StepResult(full.q, full.position, requested, "projected",
                          "IK 无解（原目标步）：沿保留前进分量的可行方向移动")
    return StepResult(q.copy(), start.copy(), requested, "blocked",
                      "IK 无解：当前未找到可接受的前进方向，保持位置")
