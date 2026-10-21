"""Microphone -> PCM WAV; Windows local speech playback with explicit cleanup."""
import os
from pathlib import Path
import subprocess
import threading
import wave

import numpy as np


class Recorder:
    def __init__(self):
        self.stream = None
        self.frames = []
        self.rate = 16000
        self.lock = threading.Lock()
        self.error = None

    @staticmethod
    def devices():
        import sounddevice as sd
        return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]

    def start(self, device=None):
        import sounddevice as sd
        self.close()
        self.frames, self.error = [], None
        self.rate = int(sd.query_devices(device, "input")["default_samplerate"])
        def callback(data, frames, timing, status):
            with self.lock:
                if status:
                    self.error = str(status)
                self.frames.append(data.copy())
        self.stream = sd.InputStream(device=device, samplerate=self.rate, channels=1,
                                     dtype="int16", callback=callback)
        try:
            self.stream.start()
        except BaseException:
            self.close()
            raise

    def finish(self, path):
        self.close()
        with self.lock:
            frames = self.frames
            self.frames = []
        if not frames:
            raise ValueError("没有录到音频，请检查麦克风。")
        samples = np.concatenate(frames).astype("<i2")
        path = Path(path)
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(self.rate)
            out.writeframes(samples.tobytes())
        return path

    def close(self):
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()


class Speaker:
    def __init__(self):
        self.process = None
        self.lock = threading.Lock()
        self.closed = False
        self.generation = 0

    def token(self):
        with self.lock:
            return self.generation

    def speak(self, text, token=None):
        if os.name != "nt":
            return
        script = """[Console]::InputEncoding=[System.Text.UTF8Encoding]::new();
Add-Type -AssemblyName System.Speech;
$speaker=New-Object System.Speech.Synthesis.SpeechSynthesizer;
try {
 $voice=$speaker.GetInstalledVoices() | Where-Object {$_.VoiceInfo.Culture.Name -like 'zh-*'} | Select-Object -First 1;
 if($voice){$speaker.SelectVoice($voice.VoiceInfo.Name)};
 $speaker.Speak([Console]::In.ReadToEnd())
} finally {$speaker.Dispose()}
"""
        with self.lock:
            if self.closed or (token is not None and token != self.generation):
                return
            self._stop_locked()
            self.process = subprocess.Popen(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW)
            process = self.process
        try:
            process.communicate(text.encode("utf-8"), timeout=180)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()

    def _stop_locked(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=5)
        self.process = None

    def stop(self):
        with self.lock:
            self.generation += 1
            self._stop_locked()

    def close(self):
        with self.lock:
            self.closed = True
            self.generation += 1
            self._stop_locked()
