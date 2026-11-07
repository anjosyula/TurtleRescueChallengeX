#!/usr/bin/env python3
"""Run TurtleHero's OAK detector, LAN dashboard and direct peer synchronization."""
from __future__ import annotations

import argparse
import base64
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
import socket
import sys
import subprocess
import threading
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["WANDB_MODE"] = "disabled"

from dashboard.network import MAX_BODY, State, synchronize


def lan_urls(port):
    addresses = set()
    try:
        for result in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET):
            ip = result[4][0]
            if not ip.startswith("127."):
                addresses.add(ip)
    except OSError:
        pass
    # macOS often resolves its hostname only to loopback.
    if os.name == "posix":
        for interface in ("en0", "en1"):
            try:
                ip = subprocess.check_output(["ipconfig", "getifaddr", interface],
                                             stderr=subprocess.DEVNULL, timeout=2).decode().strip()
                if ip:
                    addresses.add(ip)
            except (OSError, subprocess.SubprocessError):
                pass
    if not addresses and Path("/sbin/ifconfig").is_file():
        try:
            interfaces = subprocess.check_output(["/sbin/ifconfig"],
                                                stderr=subprocess.DEVNULL, timeout=2).decode()
            addresses.update(ip for ip in re.findall(r"\binet (\d+\.\d+\.\d+\.\d+)\b", interfaces)
                             if not ip.startswith("127."))
        except (OSError, subprocess.SubprocessError):
            pass
    return [f"http://{ip}:{port}" for ip in sorted(addresses)]


def handler_for(state, key=""):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def send(self, status, body, content_type="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == "/api/state":
                packet = state.snapshot()
                with state.lock:
                    packet.update(local_id=state.node_id, urls=state.urls,
                                  source=state.source, confidence=state.confidence,
                                  peers=list(state.peers.values()))
                self.send(200, packet)
            elif parsed.path in ("/api/frame", "/api/event-image"):
                query = parse_qs(parsed.query)
                body = state.image(node_id=query.get("node", [""])[0],
                                   event_id=query.get("event", [None])[0])
                self.send(200 if body else 404, body or b"", "image/jpeg")
            elif parsed.path in ("/", "/app.js", "/style.css"):
                name = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}[parsed.path]
                kind = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8",
                        "style.css": "text/css; charset=utf-8"}[name]
                self.send(200, (ROOT / "dashboard" / name).read_bytes(), kind)
            else:
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            try:
                peer = self.path == "/api/peer/sync"
                if peer:
                    if not hmac.compare_digest(self.headers.get("X-TurtleHero-Key", ""), key):
                        self.send(403, {"error": "Peer key does not match"})
                        return
                    if self.headers.get("Origin"):
                        self.send(403, {"error": "Peer sync requires a station"})
                        return
                else:
                    origin = self.headers.get("Origin")
                    if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                        self.send(403, {"error": "Open controls on this dashboard"})
                        return
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("Use application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError("Invalid message length")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete request")
                body = json.loads(raw)
                if not isinstance(body, dict):
                    raise ValueError("Expected an object")
                if peer:
                    state.merge(body)
                    packet = state.snapshot(images=True)
                    for event in packet["events"]:
                        event["jpeg"] = ""
                    self.send(200, packet)
                elif self.path == "/api/control":
                    state.control(body.get("source", state.source),
                                  body.get("confidence", state.confidence))
                    self.send(200, {"ok": True})
                elif self.path == "/api/peers":
                    url = state.add_peer(body["url"])
                    self.send(200, {"url": url})
                elif self.path == "/api/peers/remove":
                    with state.lock:
                        state.peers.pop(body["url"], None)
                    self.send(200, {"ok": True})
                else:
                    self.send(404, {"error": "Not found"})
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.send(400, {"error": str(error)[:300]})
    return Handler


def make_server(state, host, port, key=""):
    server = ThreadingHTTPServer((host, port), handler_for(state, key))
    server.daemon_threads = True
    return server


def arguments():
    from dashboard.detector import MODEL, REPLAY
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--name", default="Field station 1")
    p.add_argument("--node-id")
    p.add_argument("--source", choices=("oak", "replay", "none"), default="oak")
    p.add_argument("--peer", action="append", default=[])
    p.add_argument("--peer-key", default=os.environ.get("TURTLEHERO_PEER_KEY", ""))
    p.add_argument("--confidence", type=float, default=.6331660747528076)
    p.add_argument("--weights", type=Path, default=MODEL / "turtle.pt")
    p.add_argument("--replay-path", type=Path, default=REPLAY)
    p.add_argument("--oak-python", type=Path, default=Path(sys.executable))
    p.add_argument("--oak-device")
    p.add_argument("--compute", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    p.add_argument("--fps", type=float, default=4)
    args = p.parse_args()
    if not .01 <= args.confidence <= .99 or not .2 <= args.fps <= 30:
        p.error("Confidence must be 0.01–0.99 and FPS 0.2–30")
    if not 1 <= args.port <= 65535 or not 1 <= len(args.name) <= 80:
        p.error("Choose a valid port and station name")
    args.node_id = args.node_id or f"{socket.gethostname()}-{args.port}"
    if not 1 <= len(args.node_id) <= 64:
        p.error("Node ID must be 1–64 characters")
    return args


def main():
    args = arguments()
    state = State(args.node_id, args.name, args.source, args.confidence, lan_urls(args.port))
    for url in args.peer:
        state.add_peer(url)
    server = make_server(state, args.host, args.port, args.peer_key)
    state.heartbeat()
    from dashboard.detector import detector_thread
    workers = [
        threading.Thread(target=detector_thread, args=(state, args), daemon=True),
        threading.Thread(target=synchronize, args=(state, args.peer_key), daemon=True),
    ]
    for worker in workers:
        worker.start()

    def stop(*_):
        state.stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    print(f"TurtleHero dashboard   http://localhost:{args.port}", flush=True)
    for url in state.urls:
        print("Open on the second computer   " + url, flush=True)
    if not state.urls:
        print("LAN address unavailable. Connect to Wi-Fi and use this laptop's IP address.", flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        state.stop.set()
        server.server_close()
        for worker in workers:
            worker.join(timeout=10)


if __name__ == "__main__":
    main()
