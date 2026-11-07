#!/usr/bin/env python3
"""Collect local, timestamp-paired OAK-D Pro training samples with DepthAI v3."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time
import uuid

import cv2
import numpy as np

STREAMS = ("rgb", "mono_left", "mono_right", "rectified_left", "rectified_right",
           "depth", "confidence_raw")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    """Replace a complete JSON document, without exposing a half-written file."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def metric_depth(raw):
    """DepthAI uint16 millimeters -> float32 camera-axis meters; 0 is unknown."""
    if raw.dtype != np.uint16 or raw.ndim != 2:
        raise ValueError("Expected a 2-D uint16 millimeter depth map")
    valid = raw != 0
    depth = raw.astype(np.float32) * np.float32(0.001)
    depth[~valid] = 0
    return depth, valid


def timestamp_span_ms(timestamps):
    values = list(timestamps.values())
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Missing/nonfinite timestamps")
    return (max(values) - min(values)) * 1000


def frame_metadata(frame):
    transform = frame.getTransformation()
    geometry = None
    if transform.isValid():
        extrinsics = transform.getExtrinsics()
        geometry = {
            "size_wh": list(transform.getSize()),
            "source_size_wh": list(transform.getSourceSize()),
            "intrinsics": transform.getIntrinsicMatrix(),
            "source_intrinsics": transform.getSourceIntrinsicMatrix(),
            "source_to_output_matrix": transform.getMatrix(),
            "distortion_coefficients": transform.getDistortionCoefficients(),
            "distortion_model": str(transform.getDistortionModel()),
            "extrinsics": {
                "matrix_4x4": extrinsics.getTransformationMatrix(),
                "length_unit": str(extrinsics.lengthUnit),
                "to_camera_socket": str(extrinsics.toCameraSocket),
                "to_device_id": extrinsics.toDeviceId,
            },
            "provenance": "SDK-reported; confidence-map geometry is not a guarantee of RGB alignment",
        }
    return {
        "sequence": frame.getSequenceNum(),
        "device_monotonic_s": frame.getTimestampDevice().total_seconds(),
        "host_synced_monotonic_s": frame.getTimestamp().total_seconds(),
        "exposure_us": frame.getExposureTime().total_seconds() * 1e6,
        "iso": frame.getSensitivity(),
        "lens_position": frame.getLensPosition(),
        "type": str(frame.getType()),
        "geometry": geometry,
    }


def build_pipeline(dai, device, args):
    pipeline = dai.Pipeline(device)
    rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    features = device.getConnectedCameraFeatures()
    if any(f.socket == dai.CameraBoardSocket.CAM_A and f.hasAutofocus for f in features):
        lens = device.readCalibration().getLensPosition(dai.CameraBoardSocket.CAM_A)
        rgb.initialControl.setManualFocus(lens)
    rgb_out = rgb.requestOutput((1280, 960), type=dai.ImgFrame.Type.NV12,
                                fps=args.fps, enableUndistortion=True)
    left_out = left.requestOutput((1280, 800), type=dai.ImgFrame.Type.GRAY8, fps=args.fps)
    right_out = right.requestOutput((1280, 800), type=dai.ImgFrame.Type.GRAY8, fps=args.fps)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_ACCURACY)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(args.extended_disparity)
    stereo.setSubpixel(not args.extended_disparity)
    stereo.setRectification(True)
    # Use measured EEPROM translations, rather than nominal hardware dimensions.
    stereo.setDisparityToDepthUseSpecTranslation(False)
    stereo.setDepthAlignmentUseSpecTranslation(False)
    stereo.setRectifyEdgeFillColor(0)
    config = stereo.initialConfig
    config.setDepthUnit(dai.StereoDepthConfig.AlgorithmControl.DepthUnit.MILLIMETER)
    config.setConfidenceThreshold(args.confidence_threshold)
    # Preserve holes: don't substitute past or neighboring measurements for unknowns.
    config.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
    config.postProcessing.temporalFilter.enable = False
    config.postProcessing.spatialFilter.enable = False
    config.postProcessing.speckleFilter.enable = False
    config.postProcessing.decimationFilter.decimationFactor = 1
    left_out.link(stereo.left)
    right_out.link(stereo.right)
    if args.align == "rgb":
        rgb_out.link(stereo.inputAlignTo)
    else:
        stereo.setDepthAlign(dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)

    outputs = dict(rgb=rgb_out, mono_left=stereo.syncedLeft, mono_right=stereo.syncedRight,
                   rectified_left=stereo.rectifiedLeft, rectified_right=stereo.rectifiedRight,
                   depth=stereo.depth, confidence_raw=stereo.confidenceMap)
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=args.sync_ms))
    sync.setSyncAttempts(-1)
    for name, output in outputs.items():
        output.link(sync.inputs[name])
    queue = sync.out.createOutputQueue(maxSize=2, blocking=False)
    return pipeline, queue


def save_sample(session, index, frames, args):
    metadata = {name: frame_metadata(frame) for name, frame in frames.items()}
    span = timestamp_span_ms({n: m["device_monotonic_s"] for n, m in metadata.items()})
    if span > args.sync_ms + 0.001:
        raise ValueError(f"Stream timestamps differ by {span:.3f} ms")
    raw = frames["depth"].getFrame()
    depth, valid = metric_depth(raw)
    images = {n: f.getCvFrame() for n, f in frames.items() if n != "depth"}
    target = "rgb" if args.align == "rgb" else "rectified_left"
    if depth.shape != images[target].shape[:2]:
        raise ValueError("Depth and alignment target have different dimensions")
    name = f"{index:06d}"
    temporary = session / "samples" / ("." + name + ".partial")
    destination = session / "samples" / name
    temporary.mkdir()
    files = {}
    for stream, array in images.items():
        filename = stream + ".png"
        if not cv2.imwrite(str(temporary / filename), array, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(f"Could not save {filename}")
        files[stream] = f"samples/{name}/{filename}"
    np.save(temporary / "depth_m.npy", depth, allow_pickle=False)
    if not cv2.imwrite(str(temporary / "validity.png"), valid.astype(np.uint8) * 255):
        raise OSError("Could not save depth validity")
    files.update(depth=f"samples/{name}/depth_m.npy", validity=f"samples/{name}/validity.png")
    info = {
        "schema_version": 1, "sample_id": name, "session_id": session.name,
        "host_received_utc": utc_now(), "streams": metadata, "sync_span_ms": span,
        "depth_alignment": target, "depth_units": "meters", "depth_kind": "camera_axis_z",
        "depth_valid_fraction": float(valid.mean()), "files": files,
        "scene_hint": args.scene, "observation_id": args.observation,
        "individual_id": args.individual, "annotation_status": "unreviewed",
        "boxes_xyxy": None,
    }
    write_json(temporary / "sample.json", info)
    temporary.rename(destination)
    # The manifest only references samples whose complete directory has been committed.
    with (session / "manifest.jsonl").open("a") as manifest:
        manifest.write(json.dumps(info, allow_nan=False) + "\n")
        manifest.flush()
    return info


def show_preview(frames, count, args):
    rgb = frames["rgb"].getCvFrame()
    depth, valid = metric_depth(frames["depth"].getFrame())
    color = cv2.applyColorMap(np.uint8(np.clip(depth / args.preview_max_m, 0, 1) * 255),
                             cv2.COLORMAP_TURBO)
    color[~valid] = 0
    target = rgb if args.align == "rgb" else frames["rectified_left"].getCvFrame()
    if target.ndim == 2:
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
    blend = target.copy()
    blend[valid] = cv2.addWeighted(target, 0.6, color, 0.4, 0)[valid]
    views = []
    for im in (rgb, blend):
        scale = min(640 / im.shape[1], 480 / im.shape[0])
        resized = cv2.resize(im, (round(im.shape[1] * scale), round(im.shape[0] * scale)))
        views.append(cv2.copyMakeBorder(resized, 0, 480 - resized.shape[0],
                                       0, 640 - resized.shape[1], cv2.BORDER_CONSTANT))
    display = np.hstack(views)
    cv2.putText(display, f"Saved {count} | q: stop | space: extra sample | unknown depth is uncolored",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("TurtleHero: RGB and depth overlay", display)
    return cv2.waitKey(1) & 0xFF


def record(args):
    import depthai as dai

    devices = dai.Device.getAllAvailableDevices()
    if args.list_devices:
        print(json.dumps([{"id": d.deviceId, "state": str(d.state), "protocol": str(d.protocol)}
                          for d in devices], indent=2))
        return 0
    candidates = [d for d in devices if not args.device or d.deviceId == args.device]
    if len(candidates) != 1:
        raise RuntimeError("Expected one available OAK. Check USB/power, close other camera programs, "
                           "and use --list-devices / --device ID when multiple cameras are attached.")
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    session = args.output.resolve() / session_id
    session.mkdir(parents=True)
    (session / "samples").mkdir()
    (session / "manifest.jsonl").touch()
    print(f"Session: {session}", flush=True)
    summary = {"started_utc": utc_now(), "status": "starting", "samples": 0,
               "groups_received": 0, "sequence_gaps": 0, "out_of_order_groups": 0,
               "session": str(session)}
    write_json(session / "summary.json", summary)
    try:
        with dai.Device(candidates[0]) as device:
            if device.getPlatform() != dai.Platform.RVC2:
                raise RuntimeError("This recorder targets the RVC2 OAK-D Pro.")
            calibration = device.readCalibration()
            calibration.eepromToJsonFile(str(session / "calibration.json"))
            pipeline, queue = build_pipeline(dai, device, args)
            session_metadata = {
                "schema_version": 1, "session_id": session_id, "created_utc": utc_now(),
                "device": {"name": device.getDeviceName(), "id": candidates[0].deviceId,
                           "platform": str(device.getPlatform()), "usb_speed": str(device.getUsbSpeed()),
                           "cameras": [str(f) for f in device.getConnectedCameraFeatures()],
                           "ir_drivers": device.getIrDrivers()},
                "software": {p: importlib.metadata.version(p) for p in
                             ("depthai", "numpy", "opencv-python")},
                "python": platform.python_version(), "host": platform.platform(),
                "recorder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "geometry": {
                    "rgb": "CAM_A, 1280x960, SDK undistorted; per-frame transform records intrinsics/crop",
                    "mono_left": "CAM_B, 1280x800, unrectified grayscale",
                    "mono_right": "CAM_C, 1280x800, unrectified grayscale",
                    "rectified_left": "StereoDepth rectified left view",
                    "rectified_right": "StereoDepth rectified right view",
                    "depth": "float32 camera-axis meters, zero unknown; aligned to " + args.align,
                    "validity": "uint8: 255 measured depth, 0 unknown",
                    "confidence_raw": "uint8 native stereo matching grid, NOT RGB aligned; "
                        "no LR-check or postprocessing. RVC2 convention: 0 best, 255 worst. "
                        "Preserved raw; do not resize or multiply into RGB-aligned depth.",
                    "aligned_confidence": "unavailable; downstream fallback may use binary validity "
                        "as an uncalibrated quality gate, never as a probability",
                    "orientation": "native sensor orientation; no host rotation, mirroring, or letterbox",
                    "calibration": "factory EEPROM calibration.json, native DepthAI units (extrinsics cm)",
                },
                "stereo": {"preset": "FAST_ACCURACY", "lr_check": True,
                           "disparity_to_depth_translation": "calibration",
                           "depth_alignment_translation": "calibration",
                           "subpixel": not args.extended_disparity,
                           "extended_disparity": args.extended_disparity,
                           "spatial_filter": False, "temporal_filter": False,
                           "speckle_filter": False, "median_filter": "OFF", "decimation": 1},
                "source": {"type": "local_camera", "license": args.license,
                           "operator": args.operator},
                "split_policy": "Keep the entire session, observation, and individual in one split.",
                "labels": "No inferred ground truth. Unreviewed boxes are null, not an empty negative label.",
            }
            write_json(session / "session.json", session_metadata)
            with pipeline:
                pipeline.start()
                if device.getIrDrivers():
                    if not device.setIrLaserDotProjectorIntensity(args.ir_dot):
                        raise RuntimeError("Failed to configure IR dot projector")
                    if not device.setIrFloodLightIntensity(args.ir_flood):
                        raise RuntimeError("Failed to configure IR flood light")
                elif args.ir_dot or args.ir_flood:
                    raise RuntimeError("IR illumination requested but device has no IR driver")
                started = last_group = time.monotonic()
                next_sample = started + args.warmup
                last_sequence = None
                last_timestamp = None
                summary["status"] = "recording"
                summary["pipeline_started_utc"] = utc_now()
                write_json(session / "summary.json", summary)
                reason = "duration"
                while pipeline.isRunning():
                    now = time.monotonic()
                    if args.duration and now - started >= args.duration:
                        break
                    if now - last_group > args.timeout:
                        raise RuntimeError("Timed out waiting for synchronized frames. Check USB/power "
                                           "or lower --fps; inspect --sync-ms if camera clocks differ.")
                    group = queue.tryGet()
                    if group is None:
                        time.sleep(0.005)
                        continue
                    last_group = now
                    frames = {name: group[name] for name in STREAMS}
                    summary["groups_received"] += 1
                    seq = frames["rgb"].getSequenceNum()
                    timestamp = frames["rgb"].getTimestampDevice().total_seconds()
                    if last_timestamp is not None and timestamp <= last_timestamp:
                        summary["out_of_order_groups"] += 1
                        continue
                    if last_sequence is not None:
                        summary["sequence_gaps"] += max(0, seq - last_sequence - 1)
                    last_sequence, last_timestamp = seq, timestamp
                    key = show_preview(frames, summary["samples"], args) if args.preview else -1
                    if key == ord("q"):
                        reason = "user_stop"
                        break
                    if now < started + args.warmup:
                        continue
                    if now >= next_sample or key == ord(" "):
                        if shutil.disk_usage(session).free < args.min_free_gb * 1e9:
                            raise OSError("Free disk space is below --min-free-gb; recording stopped")
                        info = save_sample(session, summary["samples"], frames, args)
                        summary["samples"] += 1
                        next_sample = time.monotonic() + args.interval
                        print(f"Saved {info['sample_id']} | valid depth {info['depth_valid_fraction']:.1%} "
                              f"| sync {info['sync_span_ms']:.2f} ms", flush=True)
                        if args.max_samples and summary["samples"] >= args.max_samples:
                            reason = "max_samples"
                            break
                else:
                    raise RuntimeError("Camera pipeline stopped unexpectedly")
                summary["pipeline_elapsed_s"] = time.monotonic() - started
                if not summary["samples"]:
                    raise RuntimeError("No samples recorded; allow more time after warmup")
                summary.update(status="complete", stop_reason=reason)
    except KeyboardInterrupt:
        summary.update(status="interrupted", stop_reason="keyboard_interrupt")
    except Exception as exc:
        summary.update(status="error", error=str(exc))
        raise
    finally:
        summary["ended_utc"] = utc_now()
        write_json(session / "summary.json", summary)
        if args.preview:
            cv2.destroyAllWindows()
        print(json.dumps(summary, indent=2), flush=True)
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--device", help="Device ID from --list-devices")
    p.add_argument("--output", type=Path, default=Path("recordings/oak"))
    p.add_argument("--preview", action="store_true", help="Show RGB/depth; q stops, space saves extra sample")
    p.add_argument("--duration", type=float, default=300, help="Seconds after pipeline starts; 0 until interrupted")
    p.add_argument("--max-samples", type=int, default=0, help="0 means no sample-count limit")
    p.add_argument("--fps", type=float, default=10)
    p.add_argument("--interval", type=float, default=0.5, help="Minimum seconds between saved groups")
    p.add_argument("--warmup", type=float, default=2, help="Seconds for exposure to settle")
    p.add_argument("--sync-ms", type=float, default=20, help="Maximum stream timestamp span")
    p.add_argument("--timeout", type=float, default=20, help="Fail after this many seconds without a group")
    p.add_argument("--min-free-gb", type=float, default=1)
    p.add_argument("--align", choices=("rgb", "left"), default="rgb")
    p.add_argument("--extended-disparity", action="store_true", help="Closer stereo range; disables subpixel")
    p.add_argument("--confidence-threshold", type=int, default=200)
    p.add_argument("--ir-dot", type=float, default=0, help="Dot-projector intensity 0..1")
    p.add_argument("--ir-flood", type=float, default=0, help="IR flood intensity 0..1")
    p.add_argument("--preview-max-m", type=float, default=5, help="Fixed depth color scale; saved data unchanged")
    p.add_argument("--scene", choices=("unknown", "turtle", "background"), default="unknown")
    for field in ("site", "observation", "individual", "species", "lighting", "occlusion",
                  "operator", "license", "notes"):
        p.add_argument("--" + field, default=None)
    p.add_argument("--distance-m", type=float, help="Optional manually measured camera-to-animal distance")
    p.add_argument("--shell-length-cm", type=float)
    return p


def main():
    p = parser()
    args = p.parse_args()
    for name in ("duration", "interval", "warmup", "min_free_gb", "max_samples"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            p.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    for name in ("fps", "sync_ms", "timeout", "preview_max_m"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("ir_dot", "ir_flood"):
        if not 0 <= getattr(args, name) <= 1:
            p.error(f"--{name.replace('_', '-')} must be in 0..1")
    for name in ("distance_m", "shell_length_cm"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            p.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not 0 <= args.confidence_threshold <= 255:
        p.error("--confidence-threshold must be in 0..255")
    if args.sync_ms >= 1000 / (2 * args.fps):
        p.error("--sync-ms must be less than half a frame period (500 / fps)")
    try:
        return record(args)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
