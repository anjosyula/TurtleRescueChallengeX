# TurtleHero — Turtle Rescue Challenge X

Plug a Luxonis OAK camera into a laptop, run the trained turtle detector, and view its images and possible-turtle alerts from another computer on the same network. Stations can also exchange sightings directly through the peer-to-peer dashboard.

**Start here:** [prompt for an agent on the camera laptop](docs/OTHER_LAPTOP_PROMPT.md).

## Camera laptop setup

Use a native macOS, Windows, or Linux laptop with **Python 3.12**, Git, and a Luxonis OAK USB camera. Python 3.13 is also accepted by the installer. The built-in webcam is not the OAK camera source. WSL/containers require separate USB forwarding and are not the default setup.

The repository includes two turtle-trained RT-DETR-L checkpoints, about 66 MB each. A normal clone includes them; Git LFS and a training run are not required. Raw training photos, camera recordings, local environments, and presentation files are not included.

### macOS / Linux

```sh
git clone https://github.com/anjosyula/TurtleRescueChallengeX.git
cd TurtleRescueChallengeX
python3.12 setup_dashboard.py
.venv/bin/python check_setup.py --camera
python3.12 run_dashboard.py
```

### Windows PowerShell

```powershell
git clone https://github.com/anjosyula/TurtleRescueChallengeX.git
cd TurtleRescueChallengeX
py -3.12 setup_dashboard.py
.\.venv\Scripts\python.exe check_setup.py --camera
py -3.12 run_dashboard.py
```

The installer creates `.venv` in the checkout and installs pinned packages. It does not change global Python, install drivers, modify the firewall, or flash the camera. It uses CPU PyTorch wheels on Windows/Linux by default. On Apple Silicon, the normal build can use MPS. For a compatible NVIDIA laptop, `setup_dashboard.py --cuda` selects the standard GPU build; driver compatibility still needs checking on that laptop. `--compute cpu` is a supported fallback at launch.

Connect the OAK with a **USB data cable**, and close other programs using it. Start only one camera pipeline per physical device. If more than one OAK is connected, use the ID printed by the check:

```sh
python3.12 run_dashboard.py --oak-device DEVICE_ID
```

The camera sends RGB to the laptop. RT-DETR inference runs on the laptop, not on the OAK VPU. A live feed must say **LIVE OAK** and show a fresh image. Device discovery alone does not prove that frames are arriving.

## View from the other computer

Keep the camera-laptop process running. Open `http://localhost:8765` there. On the second computer, use the **Open on the second computer** address printed in the terminal, or the dashboard's **Copy link** button. It looks like `http://192.168.1.25:8765`.

Both computers must use the same Wi-Fi or Ethernet network. `localhost` on the second computer points to itself, not the camera laptop. A guest network can isolate devices. If the OS asks whether Python may receive connections, allow it on the trusted local network for this demo. Stop the server with **Ctrl+C** in its terminal.

The page displays possible turtle detections, scores, source station, recent sightings, and connection state. Model scores are not calibrated probabilities. No detections does not mean a site is clear.

## Independent peer station

A browser is enough for viewing. To run a second independent receiver, clone the repo on that computer and use ordinary Python 3.10+:

```sh
python turtlehero_dashboard.py --source none --name "Site office" --peer http://CAMERA_LAPTOP_IP:8765
```

This receiver needs only Python's standard library. Keep its source at **Receive from peers only**. Open its own `http://localhost:8765`. Stations preserve origin IDs, suppress duplicate relayed sightings, mark stale frames, and reconnect after interruption. A peer address can also be entered in the UI. This is a local HTTP prototype with manually configured peers, not an outdoor radio mesh or internet service.

## Included checkpoints

| Launcher | Checkpoint | Model-selection results |
|---|---|---|
| `run_dashboard.py` | Original RT-DETR-L | 44/51 turtles, 9 false boxes |
| `run_dashboard.py --model usb` | Five-epoch USB background fine-tune | 43/51 turtles, 6 false boxes |

These results use the same 50 validation photos and separately selected operating thresholds. The USB model trained on 29 user-confirmed background frames plus 198 labeled training photos. At the unchanged original threshold, it flagged 8/117 USB frames instead of 42/117. That USB sequence is training footage, not independent field validation. The original remains the default because it found one more validation turtle. See [model comparison](models/comparison.json). Each checkpoint has a SHA-256 manifest; the launcher verifies it before starting.

## Replay and recording

Replay requires an OAK recording supplied separately:

```sh
python3.12 run_dashboard.py --source replay --replay-path /path/to/oak-session
```

The dashboard labels replay as **Recorded footage**. It cycles samples at the processing rate, not the original video timing. The old recording paths in research examples are not available in a fresh clone.

For a new RGB/depth dataset, stop the dashboard first and use the separate recorder:

```sh
.venv/bin/python collect_oak.py --list-devices
.venv/bin/python collect_oak.py --preview --duration 60 --scene turtle --output recordings/oak
```

On Windows, replace `.venv/bin/python` with `.\.venv\Scripts\python.exe`. Review `collect_oak.py --help` and [capture contract](OAK_CAPTURE.md) before collecting. Raw recordings stay local and are ignored by Git. Collection does not automatically label or train on a recording.

## Code layout

- `turtlehero_dashboard.py`, `dashboard/`: HTTP UI, peer exchange, supervised OAK capture, standalone inference runtime.
- `run_dashboard.py`, `setup_dashboard.py`, `check_setup.py`: portable launch/setup and checkpoint checks.
- `collect_oak.py`, `validate_oak.py`, `oak_review.py`, `annotate_oak.py`: recording and review.
- `train_*.py`, `evaluate_*.py`, `turtle_geometry.py`, `turtle_fusion.py`: research training, evaluation and geometry.
- `render_oak_splat_cpu.py`, `local_site_sfm.py`, `fit_site_splat.py`: measured-depth snapshots and site reconstruction research.
- `cluster/hopper/`, `vendor/metalsplat_site/`: optional cluster workflow and attributed upstream splat helper.

The research tools require additional source data, assets, optional packages, and sometimes historical model packages that are not needed by the dashboard and are not bundled here. [Research workflow notes](docs/research-workflows.md) describe that earlier local work. The site-splat CPU renderer currently uses macOS `clang++` dynamic libraries; it is separate from the cross-platform dashboard.

## Checks

```sh
.venv/bin/python -m unittest discover -s tests -p test_dashboard.py -v
.venv/bin/python -m unittest discover -s tests -p 'test_oak*.py' -v
```

On November 7, 2026, a fresh Python 3.12 environment installed successfully on macOS ARM64. All 37 dashboard, capture/review and geometry checks passed. Standalone inference matched 239 original decoded predictions exactly on a real source frame, without importing training modules. Real turtle and background replay frames reached a second local HTTP dashboard. Windows/Linux wheel availability was checked, but those operating systems have not been run here. A live USB feed and another physical laptop must be verified on the target hardware; this repository does not claim those tests happened on your second laptop.

The requested solar posts, SNN/neuromorphic processing, smell sensing, and shared site map remain future work.

## References

The package pins match the project checkpoints: [Ultralytics RT-DETR](https://docs.ultralytics.com/models/rtdetr/), [Ultralytics 8.4.157](https://pypi.org/project/ultralytics/8.4.157/), and [DepthAI 3.11.0](https://pypi.org/project/depthai/3.11.0/). Camera setup follows the [DepthAI v3 documentation](https://docs.luxonis.com/software-v3/depthai). See [third-party notices](THIRD_PARTY.md).
