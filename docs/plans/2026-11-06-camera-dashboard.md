# OAK camera and peer dashboard

The user requested this topology on 2026-11-06 and selected the OAK camera.

Use DepthAI v3 Camera output, the existing RT-DETR-L decoder and 640-square letterbox. Keep acquisition in .venv-oak and inference in .venv to preserve their compatible Python versions. One supervised acquisition child publishes the latest complete JPEG atomically. Browser requests use HTTP on the laptop. Peers synchronize bounded snapshots and deduplicated sightings, retaining origin identity, age and source mode.

Flow: OAK RGB or saved OAK frames feed host RT-DETR-L, then the station HTTP server. The browser connects to the station. Stations synchronize directly with each other.

The page shows an annotated image, source station, source mode, confidence, recent sightings, camera state, peer health and a copyable LAN address. Camera disconnection must produce a waiting or stale state. Replay is visibly labeled.

Proof uses a saved OAK session, real detector inference and a second server instance. Verify detections and matching origin IDs through the remote server, deduplication, offline and reconnect behavior, safe rejection of malformed messages, and the rendered browser interface. Confirm a live frame separately when OAK hardware is connected.

Source API verified through Luxonis MCP on 2026-11-06.
https://docs.luxonis.com/software-v3/depthai/examples/camera/camera_output
The existing project contract selects .venv-oak for RVC2. oakctl remains absent.

Implementation complete. Verification: 13 dashboard tests and 19 existing OAK capture/review tests passed; real RT-DETR replay reached a second independent HTTP server. Browser desktop view and 390px mobile layout checked (no horizontal overflow); recorded footage labels and bounding boxes visible. The live-camera selector is exercised; physical OAK and a second physical LAN computer still require hardware verification. Guide: docs/dashboard.md.
