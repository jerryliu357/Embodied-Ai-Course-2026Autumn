"""Feedback pacing and bounded outer joint integral correction (raw counts).

This is host-side control, not a write to the SCS215 firmware PID registers.
The nominal IK reference and the compensated motor command stay separate.
"""
from __future__ import annotations

import numpy as np


def ramp_down(error, soft, hard):
    return float(np.clip((hard - error) / (hard - soft), 0., 1.))


class JointTracker:
    def __init__(self, mapping, initial, hardware_limits=None):
        self.cfg = mapping.cfg
        self.lower = mapping.raw_lower.astype(float).copy()
        self.upper = mapping.raw_upper.astype(float).copy()
        if hardware_limits is not None:
            self.lower = np.maximum(self.lower, hardware_limits[:5, 0])
            self.upper = np.minimum(self.upper, hardware_limits[:5, 1])
        if np.any(self.lower > self.upper):
            raise ValueError("硬件限位与标定没有交集。")
        self.bias = np.zeros(5)
        self.output = np.asarray(initial[:5], dtype=float).copy()
        self.counts_per_second = self.cfg["max_joint_speed_deg_s"] / mapping.degrees_per_count

    def command(self, reference, measured, dt):
        """Integrate persistent reference error, with slew/saturation anti-windup.

        Clamps are on the *combined* command, not just the nominal IK solution.
        Integral updates that push farther into any saturation are rejected;
        opposite-signed updates can unwind an existing bias.
        """
        reference = np.asarray(reference[:5], dtype=float)
        error = reference - np.asarray(measured[:5], dtype=float)
        update = np.where(np.abs(error) > self.cfg["integral_deadband_counts"],
                          self.cfg["tracking_ki_per_s"] * error, 0.)
        update = np.clip(update, -self.cfg["integral_rate_counts_s"],
                         self.cfg["integral_rate_counts_s"]) * dt
        limit = self.cfg["integral_limit_counts"]
        proposed_bias = np.clip(self.bias + update, -limit, limit)
        low = np.maximum(self.lower, self.output - self.counts_per_second * dt)
        high = np.minimum(self.upper, self.output + self.counts_per_second * dt)
        requested = reference + proposed_bias
        limited = np.clip(requested, low, high)
        windup = (requested - limited) * update > 1e-9
        self.bias = np.where(windup, self.bias, proposed_bias)
        self.bias = np.clip(self.bias, self.lower-reference, self.upper-reference)
        self.output = np.clip(reference + self.bias, low, high)
        # Preserve fractional output/bias internally; rounding does not lose
        # accumulated small steps. Never use the rounded output as the next state.
        raw = np.rint(self.output).astype(int)
        return np.clip(raw, self.lower, self.upper).astype(int)


def pace(cfg, reference, measured, cartesian_error_mm):
    """Scale planned speed continuously; five-count final tolerance is NOT a gate."""
    count_error = float(np.max(np.abs(np.asarray(reference[:5]) - measured[:5])))
    return min(ramp_down(count_error, cfg["tracking_soft_counts"], cfg["tracking_pause_counts"]),
               ramp_down(cartesian_error_mm, cfg["tracking_soft_mm"], cfg["tracking_pause_mm"]))
