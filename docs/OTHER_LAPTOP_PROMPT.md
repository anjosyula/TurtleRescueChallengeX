# Prompt for the agent on the laptop with the camera

Copy the text below into the other agent:

---

Set up and run TurtleHero on this laptop with the connected Luxonis OAK camera. Repository: https://github.com/anjosyula/TurtleRescueChallengeX.git

I authorize cloning/updating this repository, installing its dependencies in a project-local environment, using the connected camera, and starting the dashboard on the local network. Complete the setup and prove the live feed works.

1. Clone the repository into a suitable project folder. If it already exists, inspect its status and remote first; preserve local changes and use a fast-forward pull when safe. Read `AGENTS.md`, `README.md`, and `docs/dashboard.md`.
2. Identify the OS and available Python. Use native Python 3.12 if possible (3.13 is accepted). Run `python3.12 setup_dashboard.py` on macOS/Linux, or `py -3.12 setup_dashboard.py` on Windows. This creates `.venv`. Use the bundled turtle-trained checkpoint; do not replace it with a generic model or train a new one.
3. Run `check_setup.py --camera` with the new `.venv` Python. Resolve ordinary setup issues and check the USB data cable, power, and whether another process owns the OAK. If several cameras appear, select the intended one with `--oak-device ID`. Do not run competing camera programs. Ask before administrator changes, firmware changes, or closing an unrelated user application.
4. Start `run_dashboard.py` using Python 3.12. It uses the original baseline model by default. `--model usb` selects the optional background-trained model. Try `--compute cpu` if the automatic accelerator fails. Keep the server running in a terminal the user can stop.
5. Open `http://localhost:8765` and verify **LIVE OAK**, a fresh camera image, and running inference. A process that merely starts or a device that only enumerates is not proof. Do not substitute replay or synthetic detections for the live camera. If no turtle is in view, verify the fresh live image and inference without claiming a turtle detection.
6. Give me the actual LAN URL printed by the server so another computer on the same network can open it. Verify the API and image endpoint locally. Test the second physical computer if accessible; otherwise state that this final network check remains. Do not give `localhost` as the other computer's address. A browser on that computer is sufficient; an independent receiving station can run `python turtlehero_dashboard.py --source none --peer http://CAMERA_LAPTOP_IP:8765`.

Finish with the exact start/stop commands, model in use, camera ID, working LAN URL, what you directly verified, and any specific remaining blocker. Preserve recorded/live labels. Keep recordings local and do not upload footage or change thresholds just to manufacture a detection.
