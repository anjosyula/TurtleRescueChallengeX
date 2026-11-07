# TurtleHero repository instructions

## Immediate demo task

Read `README.md`, `docs/dashboard.md`, and `docs/OTHER_LAPTOP_PROMPT.md`. The user wants a real OAK camera feed on one laptop and its detections visible in a browser on another computer. Do the authorized setup and verify the named output.

- Use native Python 3.12 (or 3.13) and the project-local `.venv` made by `setup_dashboard.py`. Run `run_dashboard.py`. The OAK bridge uses the same environment by default. An existing separate camera environment can be passed with `--oak-python`.
- This existing RVC2 host app runs directly through DepthAI v3. No oakctl installation is required to launch it. If an installed oakctl is useful for diagnostics, inspect `oakctl --help`; do not invent commands. An empty oakctl inventory does not prove no RVC2 camera exists.
- Verify current DepthAI APIs through official Luxonis docs before changing the camera pipeline. Do not mix v2 and v3.
- `models/rtdetr-baseline/turtle.pt` is the baseline. The optional USB model has its own threshold and manifest. Do not download a generic COCO model as a substitute; it has not learned this turtle class.
- Preserve the established square letterbox and RT-DETR decoder in `dashboard/runtime.py`. The dashboard intentionally does not import training/evaluation modules.
- Raw datasets, recordings, private configs, environments and logs stay out of Git. Local cameras and paths must be rediscovered on each laptop.
- Only one process can use a camera at a time. Stop this app before running `collect_oak.py`. Do not kill unrelated user applications without authorization.
- Do not claim live inference from discovery, replay, mock boxes or a running process alone. Check the actual image and `/api/state`. If no turtle is present, report that honestly.
- A second browser is enough for the two-computer demo. `--source none` supports a standard-library-only receiver. Use the camera laptop's actual LAN address, not localhost, on the other computer.
- Ask before sudo/admin changes, firmware/OS updates, flashing, factory reset, or adopting a device. Project-local dependency installation and the requested local camera/dashboard are within the setup request.

## Research context

`instructions.md` describes the broader research design, not a requirement to retrain or redesign the working demo. Extra training, upload, and new recording jobs need a user request. Research commands in `docs/research-workflows.md` may require separately supplied private data and earlier model packages.

The original model found 44/51 validation turtles with 9 false boxes. The optional USB model found 43/51 with 6 at its selected threshold. This is a 50-photo model-selection set, not independent field accuracy. All derivatives of the confirmed no-turtle USB encounter remain training-only. Missing depth stays unknown; preserve calibration, source groups, timestamps and provenance.

## Verification after changes

Run the relevant dashboard tests and camera contract tests from README. For inference changes, compare actual outputs through the original preprocessing/decoder and a real input. For camera changes, verify a fresh live frame on the target hardware. Keep unverified physical-laptop claims explicit.
