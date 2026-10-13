"""Project robot base axes into a saved hand-eye validation camera image."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrate_intrinsics import read_image, write_image


def project(K, dist, T_camera_base, points_base):
    rvec, _ = cv2.Rodrigues(T_camera_base[:3, :3])
    pixels, _ = cv2.projectPoints(np.asarray(points_base, np.float64), rvec,
                                  T_camera_base[:3, 3], K, dist)
    return np.rint(pixels.reshape(-1, 2)).astype(int)


def annotate(folder: Path):
    result = json.loads((folder / "result.json").read_text(encoding="utf-8-sig"))
    intrinsics = json.loads(Path(result["intrinsics_file"]).read_text(encoding="utf-8-sig"))
    K = np.asarray(intrinsics["camera_matrix"], dtype=float)
    dist = np.asarray(intrinsics["dist_coeffs"], dtype=float)
    T_BE = np.asarray(result["T_base_end_effector"], dtype=float)
    T_EC = np.asarray(result["T_end_effector_camera"], dtype=float)
    T_BM = np.asarray(result["T_base_board"], dtype=float)
    T_CB = np.linalg.inv(T_BE @ T_EC)
    origin = T_BM[:3, 3]
    length = .03  # 3 cm, chosen so both arrows remain inside this image.
    points = [origin, origin + [length, 0, 0], origin + [0, length, 0]]
    pixels = project(K, dist, T_CB, points)
    image = read_image(folder / "image.png")
    start, x_end, y_end = (tuple(p) for p in pixels)
    cv2.circle(image, start, 14, (255, 0, 255), 4)
    cv2.arrowedLine(image, start, x_end, (0, 140, 255), 7, cv2.LINE_AA, tipLength=.22)
    cv2.arrowedLine(image, start, y_end, (255, 255, 0), 7, cv2.LINE_AA, tipLength=.22)
    cv2.putText(image, "M origin", (start[0] - 115, start[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "+X_base", (x_end[0] - 85, x_end[1] + 35),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 140, 255), 3, cv2.LINE_AA)
    cv2.putText(image, "+Y_base", (min(y_end[0] - 90, image.shape[1] - 150), y_end[1] + 35),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 0), 3, cv2.LINE_AA)
    cv2.rectangle(image, (12, 12), (630, 82), (20, 30, 40), -1)
    cv2.putText(image, "ORANGE: robot base +X   CYAN: robot base +Y", (25, 55),
                cv2.FONT_HERSHEY_SIMPLEX, .72, (255, 255, 255), 2, cv2.LINE_AA)
    output = folder / "base_xy_axes.jpg"
    write_image(output, image)
    print(output.resolve())
    print({"origin_px": start, "x_end_px": x_end, "y_end_px": y_end})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    annotate(args.folder.resolve())
