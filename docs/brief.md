# TurtleHero camera network demo

A laptop with a USB OAK camera detects possible turtles. A second computer on the same local network opens a browser dashboard and sees the originating station, annotated image and detection time. Additional TurtleHero stations exchange observations directly.

The demo uses the existing RT-DETR-L checkpoint. The first delivery covers RGB camera input, a live dashboard, recent sightings and peer relay with duplicate suppression and recovery. Saved OAK recordings support development while the camera is disconnected. Live hardware and an actual second computer remain separate verification steps. GPS mapping, outdoor radio, SNNs and smell sensing are future work.
