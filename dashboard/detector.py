"""Host inference and supervised OAK/replay sources."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/rtdetr-baseline"
REPLAY = ROOT / "recordings/oak/20261016T205810_f6a2ab4a"


class OakSource:
    def __init__(self, args):
        self.temporary = tempfile.TemporaryDirectory(prefix="turtlehero-camera-")
        self.folder = Path(self.temporary.name)
        command = [str(args.oak_python), "-u", str(Path(__file__).with_name("oak_capture.py")),
                   "--output", str(self.folder)]
        if args.oak_device:
            command += ["--device", args.oak_device]
        self.log = (self.folder / "capture.log").open("wb")
        try:
            self.process = subprocess.Popen(command, stdout=self.log, stderr=self.log)
        except BaseException:
            self.log.close()
            self.temporary.cleanup()
            raise
        self.last_frame = None

    def read(self):
        import cv2
        import numpy as np

        if self.process.poll() is not None:
            self.log.flush()
            detail = (self.folder / "capture.log").read_text(errors="replace")[-300:]
            raise RuntimeError("OAK capture stopped. " + detail)
        path = self.folder / "frame.bin"
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        header, data = data.split(b"\n", 1)
        meta = json.loads(header)
        if meta["captured_at"] == self.last_frame:
            return None
        self.last_frame = meta["captured_at"]
        if time.time() - self.last_frame > 5:
            return None
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Invalid camera image")
        return image, meta["captured_at"]

    def message(self):
        try:
            return json.loads((self.folder / "status.json").read_text())["message"]
        except (FileNotFoundError, json.JSONDecodeError):
            return "Opening OAK camera"

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=6)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()
        self.temporary.cleanup()


class ReplaySource:
    def __init__(self, path):
        self.records = [json.loads(row) for row in (path / "manifest.jsonl").read_text().splitlines()
                        if row.strip()]
        if not self.records:
            raise ValueError("Replay session has no frames")
        self.path, self.index = path.resolve(), 0

    def read(self):
        import cv2

        row = self.records[self.index]
        self.index = (self.index + 1) % len(self.records)
        path = (self.path / row["files"]["rgb"]).resolve()
        if not path.is_relative_to(self.path):
            raise ValueError("Replay image is outside the session")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("Could not read replay frame " + path.name)
        # Source capture time remains distinct from current replay presentation time.
        from datetime import datetime
        return image, datetime.fromisoformat(row["host_received_utc"]).timestamp()

    def message(self):
        return "Replaying saved OAK footage"

    def close(self):
        pass


def run_detector(state, args):
    from dashboard.runtime import local_runtime
    local_runtime()
    import cv2
    import numpy as np
    import torch
    from ultralytics import RTDETR
    from dashboard.runtime import rtdetr_decode
    from turtle_geometry import prepare_rgb_geometry

    torch.set_num_threads(4)
    device = args.compute
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        if not args.weights.is_file():
            raise FileNotFoundError("Trained checkpoint missing: " + str(args.weights))
        state.status("starting", "Loading RT-DETR-L on " + device)
        model = RTDETR(str(args.weights)).model.float().to(device).eval()
    except Exception as error:
        state.status("error", str(error)[:500])
        return

    source = None
    generation = -1
    last_processed = time.monotonic()
    try:
        while not state.stop.is_set():
            mode, confidence, revision = state.config()
            if generation != revision or source is None:
                if source is not None:
                    source.close()
                    source = None
                generation = revision
                if mode == "none":
                    state.status("idle", "Receiving peer observations", generation)
                    state.stop.wait(.3)
                    continue
                try:
                    source = OakSource(args) if mode == "oak" else ReplaySource(args.replay_path)
                    state.status("waiting", source.message(), generation)
                except Exception as error:
                    state.status("error", str(error)[:500], generation)
                    state.stop.wait(3)
                    continue
            try:
                frame = source.read()
                if frame is None:
                    if time.monotonic() - last_processed > 5:
                        state.status("waiting", source.message(), generation)
                    state.stop.wait(.08)
                    continue
                bgr, captured_at = frame
                start = time.perf_counter()
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image, _, transform = prepare_rgb_geometry(rgb)
                with torch.inference_mode():
                    tensor = torch.from_numpy(image[None]).to(device)
                    predictions = rtdetr_decode(model, tensor, transform, bgr.shape[1], bgr.shape[0])
                detections = [{"box": p[:4], "score": p[4]} for p in predictions if p[4] >= confidence]
                latency = (time.perf_counter() - start) * 1000
                scale = min(1, 960 / bgr.shape[1])
                annotated = cv2.resize(bgr, None, fx=scale, fy=scale)
                for detection in detections:
                    x1, y1, x2, y2 = [round(v * scale) for v in detection["box"]]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (63, 231, 178), 2)
                    label = f"Turtle {detection['score']:.0%}"
                    cv2.rectangle(annotated, (x1, max(0, y1 - 27)), (min(x1 + 155, annotated.shape[1]), y1), (38, 74, 43), -1)
                    cv2.putText(annotated, label, (x1 + 5, max(19, y1 - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
                ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ok:
                    raise RuntimeError("Could not encode dashboard frame")
                now = time.monotonic()
                fps = 1 / max(now - last_processed, .001)
                last_processed = now
                state.frame(base64.b64encode(encoded).decode(), detections, captured_at,
                            latency, fps, generation)
                state.stop.wait(max(0, 1 / args.fps - (time.perf_counter() - start)))
            except Exception as error:
                state.status("error", str(error)[:500], generation)
                source.close()
                source = None
                state.stop.wait(3)
    finally:
        if source is not None:
            source.close()


def detector_thread(state, args):
    """Keep import/runtime failures visible to browsers."""
    try:
        if args.source == "none":
            # Relay stations can run with only the Python standard library.
            while not state.stop.is_set() and state.config()[0] == "none":
                state.status("idle", "Receiving peer observations")
                state.stop.wait(.3)
        if not state.stop.is_set():
            run_detector(state, args)
    except Exception as error:
        state.status("error", "Detector could not start: " + str(error)[:450])
