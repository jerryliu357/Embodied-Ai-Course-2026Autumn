"""Regression check for the camera-preview window growth seen in the recording."""
import json
from pathlib import Path
from types import SimpleNamespace
import time
import tkinter as tk
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory

import numpy as np

from fk_capture_core import FK_DIR, load_mapping
from fk_hand_eye_gui import App
import calibrate_intrinsics


class PreviewLayoutTest(unittest.TestCase):
    def test_full_resolution_frame_does_not_change_requested_window_size(self):
        data = json.loads((FK_DIR / "calibration.json").read_text(encoding="utf-8-sig"))
        mapping = load_mapping(data)
        root = tk.Tk()
        store = SimpleNamespace(folder=Path(__file__).parent / "diagnostics", samples=[])
        app = App(root, data, mapping, store, demo=True, camera=False)
        root.update_idletasks()
        before = (root.winfo_reqwidth(), root.winfo_reqheight())

        class FakeCamera:
            frames = 1
            def sample(self):
                return time.monotonic(), np.zeros((720, 1280, 3), np.uint8)

        app.camera = FakeCamera()
        app.update_preview()
        root.update_idletasks()
        after = (root.winfo_reqwidth(), root.winfo_reqheight())
        self.assertIsInstance(app.preview, tk.Canvas)
        self.assertEqual(after, before)
        app.pool.shutdown(wait=True, cancel_futures=True)
        root.destroy()

    def test_intrinsics_preview_frame_does_not_grow_window(self):
        class FakeCamera:
            frames = 1
            process = SimpleNamespace(poll=lambda: None)
            def __init__(self, _folder):
                pass
            def sample(self):
                return time.monotonic(), np.zeros((720, 1280, 3), np.uint8)
            def close(self):
                pass

        with TemporaryDirectory() as folder, \
                patch.object(calibrate_intrinsics, "HERE", Path(folder)), \
                patch.object(calibrate_intrinsics, "Camera", FakeCamera):
            root = tk.Tk()
            app = calibrate_intrinsics.App(root, 25.0)
            root.update_idletasks()
            before = (root.winfo_reqwidth(), root.winfo_reqheight())
            for _ in range(5):
                app.tick()
                root.update_idletasks()
            after = (root.winfo_reqwidth(), root.winfo_reqheight())
            self.assertIsInstance(app.video, tk.Canvas)
            self.assertEqual(after, before)
            app.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
