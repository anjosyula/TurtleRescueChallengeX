# Portable checkout

Run `setup_dashboard.py` first using Python 3.12, then `run_dashboard.py`. See the root README for Windows and Linux commands. Both trained model files are included. Replay recordings are separate and must be supplied with `--replay-path`. The macOS `.command` launchers require the setup-created `.venv`.

# TurtleHero camera dashboard

## Two-computer demo

1. Connect the OAK camera to this Mac with a USB data cable. Close other OAK camera programs.
2. Double-click `start_dashboard.command` in the TurtleHero project, or run:

   ```sh
   cd /path/to/TurtleRescueChallengeX
   ./start_dashboard.command
   ```

3. Open `http://localhost:8765` on the camera laptop. Keep the terminal open. The page waits and reconnects automatically when an OAK is attached.
4. Connect the second computer to the same Wi-Fi or Ethernet network. Open the address printed after **Open on the second computer**, or use **Copy link** in the dashboard. The second computer only needs a browser. Do not use `localhost` there; that means the second computer itself.
5. Point the camera at a turtle. The dashboard shows the camera image, boxes and model scores, originating station, recent sightings, and camera/connection state. A live image is labeled **LIVE OAK**. Scores are model confidence, not calibrated probabilities or proof that a turtle is present.

If macOS asks whether Python may receive incoming connections, allow it for this local demo. If the laptop page works but the other computer cannot connect, check the address and firewall, and use a network that allows devices to communicate. Guest Wi-Fi often isolates devices. Recheck the printed address after changing networks. Stop with Control-C in the terminal.

## Try saved footage

Choose **Saved OAK footage** under **This station → Camera source**, or start with:

```sh
./start_dashboard.command --source replay
```

This runs the actual trained detector over a separately supplied local OAK session. The page and sightings explicitly say **Recorded footage**. It cycles saved frames at the processing rate, rather than reproducing the original video timing. Choose **Live OAK camera** to return to the USB camera. Switching sources removes the previous current frame; past sightings retain their source label.

## Direct peer connections

A browser on the second computer already satisfies the two-computer demonstration. To run an independent receiving/relay station, copy `turtlehero_dashboard.py` and the `dashboard` folder to it and use Python 3.10 or newer:

```sh
python3 turtlehero_dashboard.py --source none --name "Site office" --peer http://CAMERA_LAPTOP_IP:8765
```

On Windows use `py` in place of `python3` if needed. Open `http://localhost:8765` on the receiver. Receive-only mode needs the Python standard library; it does not need model weights or a camera. Keep its source set to **Receive from peers only**. Its station selector automatically picks an available camera until you explicitly choose one.

Alternatively, start both stations and enter either station's address under **Peer connections** on the other. A configured peer exchanges observations in both directions; only one connection is needed between each pair. Other peers can relay those observations across a chain. Each sighting retains its original station and ID so loops do not duplicate it. Interrupted links retry automatically. Peer configuration is in memory; use `--peer` to restore it when launching.

For a same-laptop relay check, give the second instance a different port:

```sh
python3 turtlehero_dashboard.py --source none --name "Remote observer" --port 8766 --peer http://127.0.0.1:8765
```

Up to eight direct connections and 24 stations are supported. This is an HTTP LAN prototype with manual peer addresses. It is not an internet service, radio mesh or automatic network discovery. Use a trusted local network. Everyone who can reach the dashboard can view its images and operate its controls. An optional common `TURTLEHERO_PEER_KEY` environment variable restricts station synchronization; it does not add browser login or encryption. Peer redirects are rejected.

## Camera and detector setup

This project already has the dependencies and trained checkpoint. The main process uses `.venv` for Ultralytics/PyTorch inference. A supervised child uses the same `.venv` for DepthAI v3 RGB acquisition unless `--oak-python` selects another environment. It keeps only the latest frame in a temporary directory and cleans it up on shutdown. This dashboard does not save new camera recordings or train weights.

The detector is the project's Ultralytics RT-DETR-L checkpoint at `models/rtdetr-baseline/turtle.pt`, not a new NVIDIA TAO deployment. It preserves the existing 640-square letterbox preprocessing and decoder. Default threshold: 0.633166; adjustable in the page. Inference uses the laptop's available CUDA/MPS/CPU backend. The OAK supplies RGB; inference does not run on its VPU in this version. Other project environments and detector defaults remain separate.

Options include `--oak-device DEVICE_ID` for multiple cameras, `--oak-python PATH`, `--weights PATH`, `--compute cpu`, `--fps 4`, `--name`, and `--port`. Run `./start_dashboard.command --help` for the full list. Dependency versions are inherited from the project's existing pinned training and OAK environments; a generic webcam is not supported by this OAK source.

## What the display means

- **Stations online** counts recent station heartbeats. **Possible turtles in view** refers to the selected station's current detections.
- Heartbeats alone cannot keep an old image current. After eight seconds without a fresh observation, live counts disappear and the last image is dimmed. Camera failures show a waiting/error message; browser failures show **CONNECTION LOST**.
- A sighting groups consecutive positive frames, with a new group after more than three seconds without a positive. It is not animal tracking or a count of distinct turtles. The UI shows the latest 30 retained sightings; memory holds at most 100. Restarting clears history. There is no cross-camera animal re-identification.
- Processing time is host inference/preprocessing time, excluding camera capture, peer transfer and browser refresh. Peer exchange is about once a second, browser refresh about every 0.7 seconds; this is an annotated image feed, not full-frame-rate video.
- GPS/site mapping, silt-fence pole hardware, solar power, SNN inference and smell sensors remain future work.

## Verification on 2026-11-06

The saved OAK session produced real RT-DETR detections and annotated images on a second local server through peer sync. Automated checks cover two-way exchange, origin retention across three hops, duplicate suppression, stale frames, source-switch races, malformed requests, shared keys and actual server disconnect/reconnect. Camera acquisition/review regression checks also pass. Live OAK discovery returned no camera. An actual USB live feed and a second physical computer on the LAN remain unverified.

```sh
.venv/bin/python -m unittest discover -s tests -p test_dashboard.py -v
.venv/bin/python -m unittest discover -s tests -p 'test_oak*.py' -v
```


## USB-trained checkpoint

`run_dashboard.py --model usb` selects the optional background-adapted checkpoint with its validation-selected threshold. The original stays the default because it detected one more validation turtle. See `models/comparison.json`. Raw USB footage is not in the repository.
