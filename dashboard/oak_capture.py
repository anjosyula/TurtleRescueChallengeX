"""DepthAI v3 camera bridge. Publishes only the latest complete frame to a private directory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import threading
import time

import cv2
import depthai as dai

stop = threading.Event()


def atomic(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    def status(message):
        atomic(args.output / "status.json", json.dumps({"message": message}).encode())

    while not stop.is_set():
        try:
            devices = [d for d in dai.Device.getAllAvailableDevices()
                       if not args.device or d.deviceId == args.device]
            if not devices:
                raise RuntimeError("Waiting for an OAK camera. Connect its USB data cable.")
            if len(devices) > 1:
                raise RuntimeError("Multiple OAK cameras found. Restart with --oak-device ID.")
            status("Opening OAK camera")
            with dai.Device(devices[0]) as device:
                with dai.Pipeline(device) as pipeline:
                    camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
                    queue = camera.requestOutput(
                        (1280, 960), type=dai.ImgFrame.Type.NV12, fps=10
                    ).createOutputQueue(maxSize=1, blocking=False)
                    pipeline.start()
                    status("OAK camera connected")
                    last_frame = time.monotonic()
                    while not stop.is_set() and pipeline.isRunning():
                        frame = queue.tryGet()
                        if frame is None:
                            if time.monotonic() - last_frame > 8:
                                raise RuntimeError("Camera stopped sending frames. Reconnecting.")
                            stop.wait(0.02)
                            continue
                        last_frame = time.monotonic()
                        ok, jpeg = cv2.imencode(".jpg", frame.getCvFrame(),
                                                [cv2.IMWRITE_JPEG_QUALITY, 88])
                        if not ok:
                            continue
                        metadata = json.dumps({
                            "captured_at": time.time(),
                            "device_timestamp": frame.getTimestampDevice().total_seconds(),
                            "sequence": frame.getSequenceNum(),
                            "device_id": devices[0].deviceId,
                        }).encode()
                        atomic(args.output / "frame.bin", metadata + b"\n" + jpeg.tobytes())
        except Exception as error:
            status(str(error))
            stop.wait(3)


if __name__ == "__main__":
    main()
