> Portable checkout: use the Python executable inside `.venv` created by `setup_dashboard.py`. The `.venv-oak` and `/Volumes/ESD-USB` examples below describe the original Mac. For another laptop pass an explicit local `--output` directory to `collect_oak.py`. Stop the dashboard before recording.

# Collecting live-turtle data with OAK-D Pro

The recorder saves local paired samples whether or not a turtle moves or is detected. No inference, external upload, or training runs during collection. Begin with a short session and inspect the images before collecting many similar frames.

## Run now

To record directly to the mounted USB drive `ESD-USB`, run:

```sh
./record_oak_usb.command                 # 60 seconds, preview; q stops early
./record_oak_usb.command --duration 0    # until q or Ctrl-C
```

This writes new timestamped sessions to `/Volumes/ESD-USB/recordings/oak/` and refuses to run if the USB volume is not mounted and writable. It uses `.venv-oak` and saves every synchronized group the host recorder can process (`--interval 0`), rather than intentionally spacing samples. Queue or disk delays can still drop frames; this is not a guaranteed lossless video stream or Bayer-RAW capture. Saved contents are unannotated RGB/mono/rectified PNGs, metric depth, validity, raw stereo confidence, calibration and timestamps. No detector or training runs. The camera must be connected separately from the USB storage drive.

On 2026-10-30, USB destination creation and launcher checks passed, but repeated DepthAI discovery returned no cameras. No new recording was created at that check.

The camera environment is already installed in `.venv-oak`. From the TurtleHero directory:

```sh
.venv-oak/bin/python collect_oak.py --list-devices
.venv-oak/bin/python collect_oak.py --preview --duration 60 --scene turtle --site home --observation turtle-001 --individual turtle-A --lighting daylight
```

Point the camera at the animal before starting. The default capture is 10 camera frames/second, with one saved synchronized group approximately every 0.5 seconds, after a two-second exposure warmup. Saving and preview add overhead, so the saved sample rate is not guaranteed. `q` stops the preview; Space saves an extra group; Ctrl-C stops headless recording. All commands create a new timestamped session under `recordings/oak/` and print its full path. Nothing remains running after the process exits.

Collect stationary, withdrawn, partly obscured, and moving turtles under the intended lighting and camera angles. Include different apparent target sizes, backgrounds, and individuals where available. Record observations naturally; do not make motion a prerequisite. Capture separate no-turtle sessions with `--scene background`, including rocks, leaves, shadows, and debris. A scene hint does not automatically assign labels to every frame.

Useful options:

| Option | Purpose |
| --- | --- |
| `--duration 60` | Stop after 60 seconds from pipeline start, including warmup; default 300, 0 until stopped |
| `--max-samples 20` | Stop after 20 saved groups |
| `--interval 2` | Leave at least two seconds between saved groups |
| `--site`, `--observation`, `--individual` | Group related captures to prevent train/test leakage |
| `--species`, `--shell-length-cm`, `--distance-m` | Optional manually supplied animal and setup information |
| `--lighting`, `--occlusion`, `--notes` | Describe conditions; unknown fields remain null |
| `--operator`, `--license` | Record provenance and rights without assuming a license |
| `--align left` | Align depth to the rectified left mono view for a separate mono/IR dataset |
| `--extended-disparity` | Try closer stereo range; disables subpixel; verify on your setup |
| `--ir-dot 0.2`, `--ir-flood 0.2` | Explicitly set IR emitter intensities in 0..1; both default off |
| `--sync-ms 20` | Reject groups whose timestamp span exceeds the limit; must be below half a frame period |
| `--device ID` | Select a specific device if more than one is attached |
| `--min-free-gb 1` | Stop if less than this free space remains |

For example, a mono-aligned session with an explicitly enabled IR flood:

```sh
.venv-oak/bin/python collect_oak.py --preview --align left --ir-flood 0.2 --duration 30 --lighting low-light --scene turtle --observation turtle-002 --individual turtle-A
```

IR settings are recorded. Mono data with the emitters off is not automatically an illuminated IR dataset. Near-IR is not thermal imagery. IR-on capture has not been tested by this implementation session.

## Installation on another machine

Use a separate Python 3.12 environment; the existing training project requires Python 3.14 and is unchanged.

```sh
uv venv --python 3.12 .venv-oak
uv pip install --python .venv-oak/bin/python -r requirements-oak.txt
```

Tested environment: Python 3.12.13, DepthAI 3.11.0, NumPy 2.5.4, OpenCV 5.0.0.93, macOS ARM64. The program currently targets RVC2 OAK-D hardware with CAM_A/B/C and rejects other platforms. It uses the DepthAI v3 Camera, StereoDepth, and Sync APIs. If no camera appears, close other camera applications, check USB data/power connections, and run `--list-devices` again.

## What gets saved

```text
recordings/oak/<session>/
  session.json          device, versions, configuration, provenance, source-code hash
  calibration.json      full factory EEPROM calibration
  manifest.jsonl        one record per committed sample
  summary.json          outcome, counts, observed sequence gaps, errors
  samples/000000/
    rgb.png             undistorted RGB view, 1280 x 960
    mono_left.png       original mono view, CAM_B, 1280 x 800
    mono_right.png      original mono view, CAM_C, 1280 x 800
    rectified_left.png  stereo-rectified left mono view
    rectified_right.png stereo-rectified right mono view
    depth_m.npy         float32 camera-axis Z in meters, aligned to selected view
    validity.png        uint8: 255 valid, 0 unknown
    confidence_raw.png raw uint8 stereo matching confidence
    sample.json         paths, timestamps, exposure, intrinsics, transforms, quality
  annotations/          created later by the review tool
```

PNG files are lossless. OpenCV reads color images as BGR; convert with `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` before RGB model preprocessing. Data retains sensor orientation; host preview scaling does not alter saved pixels. There is no 640-pixel training resize or padding at acquisition time. Apply a documented, shared resize-and-pad transform to images, depth, validity, and boxes when building a training adapter.

DepthAI millimeter depth is converted to float32 meters. Zero means unknown and is always invalid. No host interpolation fills holes. Stereo uses left-right checking with temporal, spatial, speckle, and median filters disabled and no decimation. Subpixel is enabled unless extended disparity is requested. RGB alignment uses the SDK's RVC2 `StereoDepth.inputAlignTo` path with an undistorted RGB target. Depth conversion and alignment explicitly use EEPROM calibration translations rather than the SDK's default design-spec translations. Factory calibration and SDK-reported per-frame transforms are preserved for later auditing. Autofocus cameras are set to their calibrated lens position; the connected test unit reports fixed focus.

**Confidence is separate from aligned depth.** Luxonis's RVC2 documentation defines 0 as best and 255 as worst. The confidence output is not subject to RGB alignment, left-right checks, or the depth postprocessing according to the node contract. The Python binding's confidence docstring states the opposite score direction; the platform-specific RVC2 documentation is the recorded convention. Preserve the raw values and confirm the convention before introducing a confidence-based training adapter. Some SDK metadata, including dimensions/intrinsics, can appear copied from the aligned output; it is not proof of pixel correspondence. Do not resize the map and treat it as RGB confidence. No normalized or calibrated RGB confidence is fabricated here; a future adapter may explicitly use binary validity as an uncalibrated fallback gate.

`device_monotonic_s` is camera time; it is not UTC. `host_synced_monotonic_s` is the SDK's host-synchronized clock. `host_received_utc` is host receipt time, not exposure time. Per-stream exposure durations are saved because timestamp proximity alone does not imply identical exposure windows.

Sample folders are renamed into place only after all files are written, then appended to the manifest. A failed write can leave a `.partial` folder; validation flags partial or orphan folders for inspection. Stop recording before validation. Completed samples remain usable after a disconnect; check the summary's status and error. Capture queues are bounded and nonblocking, so slow disks or preview can skip groups; observed RGB sequence gaps are reported. These are sampled datasets, not lossless continuous video or a holistic IMU recording.

Expect several megabytes per sample. Default interval, duration, and free-space limits keep collection bounded; use a slower interval for long stationary observations.

## Validate and annotate

Replace the example path with the session path printed by the recorder:

```sh
.venv-oak/bin/python validate_oak.py recordings/oak/SESSION
.venv-oak/bin/python annotate_oak.py recordings/oak/SESSION
# For the separately aligned mono dataset:
.venv-oak/bin/python annotate_oak.py recordings/oak/SESSION --view rectified_left
.venv-oak/bin/python -m unittest discover -s tests -v
```

The validator checks file integrity, shapes, timestamps, monotonic ordering, intrinsics, depth units/validity, manifest consistency, sample counts, and alignment against session settings. Existing annotations are checked for sample/view identity, dimensions, class, box bounds, and status consistency. The annotation tool validates the session before opening it. These checks do not measure geometric accuracy or establish ground truth. The summary records pipeline elapsed time for normally completed runs.

In the annotation window, `b` opens box selection. Drag a tight box around each visible turtle; Enter/Space accepts each box and Esc finishes. `e` explicitly confirms no visible turtles. `n` skips without labeling, `p` returns to the previous sample, and `q` quits. Canceling box selection does not create a background label. Existing annotations are shown and are replaced only when you submit new boxes or explicitly confirm empty.

Labels are separate JSON files with class 0 `turtle` and original-image `xyxy` boxes. For partly visible turtles, enclose visible animal pixels; do not extrapolate hidden body extent. Entirely invisible animals have no visible-object box. Original capture records retain `boxes_xyxy: null` and `annotation_status: unreviewed`; reviewed labels live under `annotations/<view>/`. Keep all captures of the same observation and individual together when splitting datasets, and reserve separate real sessions/sites for evaluation. This collector does not yet export a multimodal training dataset or implement the proposed fusion model.

### Review with detector proposals

Local review queues were prepared for both turtle recordings. Each queue selects 24 frames: temporal coverage plus additional multiple-detection, empty-prediction, or low-score cases. These selection heuristics are not calibrated uncertainty. All 159 source frames remain unreviewed until someone explicitly saves labels.

```sh
.venv-oak/bin/python annotate_oak.py recordings/oak/20261016T205218_98151fbf --proposals runs/challenge/oak_review_205218/proposals.json
.venv-oak/bin/python annotate_oak.py recordings/oak/20261016T205810_f6a2ab4a --proposals runs/challenge/oak_review_205810/proposals.json
```

Amber boxes are predictions, not annotations. Press `a` only when **every** proposed box is correct and no visible turtle is missed. Use `b` to replace the proposals with manually drawn boxes; use `e` only after confirming that no turtle is visible. `a` never converts an empty prediction into a background label, and it does not overwrite existing reviewed labels. `n` skips without labeling. Image hashes, sample/session identities, dimensions, camera view, and box bounds are checked before showing proposals. Accepted proposals retain the declared checkpoint hash and an explicit review-method field. Contact sheets are saved beside each proposal file; they are previews, not ground truth.

For another saved session, first run `predict_challenge.py`, then build a queue:

```sh
.venv-oak/bin/python oak_review.py recordings/oak/SESSION --predictions runs/challenge/PREDICTIONS/predictions.json --model-card models/challenge-20261019/model.json --output runs/challenge/NEW_REVIEW
```

Both current recordings lack observation/individual identifiers. Do not randomly split their adjacent frames into train and validation. Resolve their relationship first and keep related captures together. The queue deliberately includes failure cases and is not an unbiased evaluation sample.

## Hardware checks on 2026-10-16

The connected camera identified as OAK-D-PRO, ID `14442C10819DE9D600`, RVC2, USB SUPER. Sensors: IMX378 RGB and two OV9282 mono cameras; LM3644 IR driver present. Both smoke tests kept IR emitters off and recorded an unlabeled indoor scene, not a live-turtle encounter.

| Check | Result |
| --- | --- |
| RGB alignment, three samples | Saved and validated; maximum timestamp span 0.175 ms; mean valid depth 19.33%; zero observed RGB sequence gaps |
| Left alignment + extended disparity + preview, three samples | Saved and validated; maximum span 0.173 ms; mean valid depth 16.48%; two observed RGB sequence gaps |
| Final recorder, RGB alignment + preview + per-frame extrinsics, two samples | Saved and validated; maximum span 0.190 ms; mean valid depth 12.57%; two observed RGB sequence gaps |
| Automated checks | All 11 passed: units, invalid input, timestamps, structural validation, corrupt/missing files, stale groups, and partial writes |

Evidence is local under `evidence/oak-smoke/`, `evidence/oak-left-smoke/`, and `evidence/oak-final/`. These checks establish acquisition and file structure only. Live-turtle collection, annotation interaction, IR-on behavior, calibration accuracy, useful detection distance, and field performance still need measurements.

## Luxonis Agent Toolkit installation

Installed with Codex CLI 0.155.1 using the [official Luxonis Codex installation steps](https://docs.luxonis.com/software-v3/agent-tools/agent-skills.md):

```sh
codex plugin marketplace add luxonis/skills
codex plugin add luxonis@luxonis
codex plugin list --marketplace luxonis --json
```

The CLI confirmed `luxonis@luxonis` version 2.0.0 is installed and enabled. Its cache contains eight skills (luxonis, workspace, app, device-setup, record, inspect, troubleshoot, model) and an HTTP MCP configuration for `https://mcp.luxonis.com/mcp`. Installation did not prompt for additional approval. MCP authorization/connectivity in a fresh session has not been verified.

Start a **new Codex session** in this project (or restart Codex and open a new session) to load the plugin. The newly installed skills and MCP tools are **not claimed available in the installation session**. The camera program itself can run immediately, independently of that restart.

In the new session, check `/plugins` for Luxonis, then ask:

```text
$luxonis Verify that the Luxonis skills are loaded and use the Luxonis MCP server to retrieve current StereoDepth documentation.
```

Confirm that the new session actually makes a Luxonis MCP call successfully, rather than merely reading this file. Approve the Luxonis server connection if Codex presents a consent prompt then. `codex plugin list` alone verifies installed/enabled state, not MCP connectivity.

Implementation sources: [Depth alignment example](https://docs.luxonis.com/software-v3/depthai/examples/image_align/depth_align.md), [StereoDepth RVC2 behavior](https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/stereo_depth.md), [stereo acquisition example](https://docs.luxonis.com/software-v3/depthai/examples/stereo_depth/stereo_depth.md), and installed DepthAI 3.11.0 API signatures.

## Follow-up review and capture, 2026-10-16

The new session exposed all eight Luxonis skills and the MCP tools. The actual MCP overview request was rejected: `MCP tool call requires approval, but approval policy is never`. MCP retrieval remains blocked. Official current StereoDepth, alignment, and oakctl Markdown documentation was downloaded directly as the documented fallback. `oakctl` is absent; the existing RVC2 recorder runs through `.venv-oak`. Workspace notes and glossary now exist. For a later oakctl installation, the official documented command is `bash -c "$(curl -fsSL https://oakctl-releases.luxonis.com/oakctl-installer.sh)"`; it was not run during this capture.

Review fixes explicitly select measured calibration translations for depth conversion and alignment, record pipeline elapsed time, check session alignment and existing annotation integrity, and validate captures before annotation. All 13 automated checks passed.

Requested capture used `--preview --duration 60 --scene turtle`, IR off, device `14442C10819DE9D600`. Session: `recordings/oak/20261016T191058_cedea2d7`. The preview received `q` and stopped after 20.675 seconds; this was not a full 60-second recording. Saved 27 samples (228 MiB), received 203 groups, observed two RGB sequence gaps and no out-of-order groups. Structural validation passed: maximum synchronization span 0.204 ms, mean valid depth 3.763%. See the session's `validation.json` and `summary.json`.

Visual spot checks of samples 000010 and 000026 showed an indoor desk scene, with motion blur in the earlier frame; no turtle was visible in those two inspected frames. The turtle scene setting is only the requested hint, not verified content. No annotations, uploads, or training were performed. Depth accuracy and live-turtle usefulness remain unverified.
