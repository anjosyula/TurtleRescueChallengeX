"""Bounded, origin-preserving peer synchronization using the Python standard library."""
from __future__ import annotations

import base64
from collections import OrderedDict
import concurrent.futures
import copy
import ipaddress
import json
import math
import socket
import threading
import time
from urllib.parse import urlsplit
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler
import uuid

MAX_NODES = 24
MAX_EVENTS = 100
MAX_BODY = 12_000_000
STALE_SECONDS = 8


def number(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a number")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Number out of range")
    return value


def string(value, limit=160):
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ValueError("Invalid text")
    return value


def jpeg(value):
    if not isinstance(value, str) or len(value) > 600_000:
        raise ValueError("Image exceeds limit")
    if value:
        raw = base64.b64decode(value, validate=True)
        if not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
            raise ValueError("Expected JPEG")
    return value


def peer_url(value):
    string(value, 300)
    value = value.strip().rstrip("/")
    if "://" not in value:
        value = "http://" + value
    parsed = urlsplit(value)
    if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Use a local HTTP address, such as http://192.168.1.20:8765")
    if parsed.path or parsed.query or parsed.fragment:
        raise ValueError("Enter only the station address and port")
    port = parsed.port or 80
    # Peer connections belong to this local network. Resolve before each connection.
    for result in socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM):
        address = ipaddress.ip_address(result[4][0])
        if not (address.is_private or address.is_loopback) or address.is_unspecified or address.is_multicast:
            raise ValueError("Peer address must be on the local network")
    return value


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A peer must not redirect images or the shared key to another destination.
        return None


class State:
    def __init__(self, node_id, name, source, confidence, urls):
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.nodes = OrderedDict()
        self.events = OrderedDict()
        self.peers = {}
        self.urls = urls
        self.node_id = node_id
        self.source = source
        self.confidence = confidence
        self.generation = 0
        self.frame_time = None
        self.encounter = None
        self.last_positive = 0.
        self.local = dict(
            id=node_id, name=name, boot=uuid.uuid4().hex, started=time.time(), seq=0,
            source=source, status="starting", detail="Starting station",
            jpeg="", detections=[], captured_at=None, frame_age=None,
            latency_ms=None, fps=None, model="RT-DETR-L",
        )

    def control(self, source, confidence):
        if source not in ("oak", "replay", "none"):
            raise ValueError("Unknown source")
        number(confidence, .01, .99)
        with self.lock:
            if source != self.source:
                self.generation += 1
                self.encounter = None
                self.frame_time = None
                self.local.update(jpeg="", detections=[], captured_at=None, frame_age=None,
                                  status="starting", detail="Switching source")
            self.source = source
            self.confidence = confidence
            self.local["source"] = source

    def config(self):
        with self.lock:
            return self.source, self.confidence, self.generation

    def status(self, status, detail, generation=None):
        with self.lock:
            if generation is not None and generation != self.generation:
                return
            self.local.update(status=status, detail=detail)
            if status != "running":
                self.local["detections"] = []
                self.encounter = None

    def frame(self, image, detections, captured_at, latency_ms, fps, generation):
        now = time.monotonic()
        with self.lock:
            if generation != self.generation:
                return
            self.frame_time = now
            self.local.update(jpeg=image, detections=detections, captured_at=captured_at,
                              latency_ms=latency_ms, fps=fps, status="running",
                              detail="Detection running", frame_age=0)
            if detections:
                if self.encounter is None or now - self.last_positive > 3:
                    self.encounter = uuid.uuid4().hex
                    event = dict(id=self.node_id + "/" + self.local["boot"] + "/" + self.encounter,
                                 node_id=self.node_id, name=self.local["name"],
                                 source=self.source, at=time.time(),
                                 confidence=max(d["score"] for d in detections),
                                 count=len(detections), jpeg=image)
                    self.events[event["id"]] = event
                    self._trim_events()
                self.last_positive = now
            elif now - self.last_positive > 3:
                self.encounter = None
            self._heartbeat()

    def _trim_events(self):
        while len(self.events) > MAX_EVENTS:
            self.events.popitem(last=False)

    def _heartbeat(self):
        now = time.monotonic()
        self.local["seq"] += 1
        self.local["frame_age"] = None if self.frame_time is None else now - self.frame_time
        self.nodes[self.node_id] = (copy.deepcopy(self.local), now)

    def heartbeat(self):
        with self.lock:
            self._heartbeat()

    def snapshot(self, images=False):
        with self.lock:
            nodes = []
            for node, received in self.nodes.values():
                entry = copy.deepcopy(node)
                age = max(0, time.monotonic() - received)
                if images and age > 60 and entry["id"] != self.node_id:
                    continue
                entry["age"] = age
                if entry.get("frame_age") is not None:
                    entry["frame_age"] += age
                if not images:
                    entry.pop("jpeg", None)
                    entry["online"] = age < STALE_SECONDS
                    entry["fresh"] = (age < STALE_SECONDS and entry["status"] == "running"
                                      and entry["frame_age"] is not None
                                      and entry["frame_age"] < STALE_SECONDS)
                    if not entry["fresh"]:
                        entry["detections"] = []
                nodes.append(entry)
            events = list(self.events.values())[-30:]
            if not images:
                events = [{k: v for k, v in event.items() if k != "jpeg"} for event in events]
            return dict(version=1, nodes=nodes, events=copy.deepcopy(events))

    def merge(self, packet):
        """Validate the entire message before committing any remote state."""
        if not isinstance(packet, dict) or packet.get("version") != 1:
            raise ValueError("Unsupported peer protocol")
        nodes, events = packet.get("nodes"), packet.get("events")
        if not isinstance(nodes, list) or len(nodes) > MAX_NODES:
            raise ValueError("Too many stations")
        if not isinstance(events, list) or len(events) > MAX_EVENTS:
            raise ValueError("Too many events")
        checked_nodes, checked_events = [], []
        for raw in nodes:
            if not isinstance(raw, dict):
                raise ValueError("Invalid station")
            node = {k: raw[k] for k in self.local}
            string(node["id"]); string(node["name"]); string(node["boot"])
            number(node["started"], 0, 4e9); number(node["seq"], 0, 1e15)
            if not isinstance(node["seq"], int):
                raise ValueError("Invalid sequence")
            if node["source"] not in ("oak", "replay", "none"):
                raise ValueError("Invalid source")
            if node["status"] not in ("starting", "waiting", "running", "error", "idle"):
                raise ValueError("Invalid status")
            string(node["detail"], 500); string(node["model"], 100)
            jpeg(node["jpeg"])
            age = number(raw.get("age", 0), 0, 86400)
            for field, limit in (("frame_age", 86400), ("captured_at", 4e9),
                                 ("latency_ms", 1e7), ("fps", 10000)):
                if node[field] is not None:
                    number(node[field], 0, limit)
            # snapshot() already includes time in transit before this hop.
            if node["frame_age"] is not None:
                node["frame_age"] = max(0, node["frame_age"] - age)
            if not isinstance(node["detections"], list) or len(node["detections"]) > 100:
                raise ValueError("Invalid detections")
            for det in node["detections"]:
                if not isinstance(det, dict):
                    raise ValueError("Invalid detection")
                number(det["score"], 0, 1)
                if not isinstance(det["box"], list) or len(det["box"]) != 4:
                    raise ValueError("Invalid box")
                for coordinate in det["box"]:
                    number(coordinate, 0, 10000)
            checked_nodes.append((node, age))
        for event in events:
            if not isinstance(event, dict):
                raise ValueError("Invalid event")
            value = {k: event[k] for k in ("id", "node_id", "name", "source", "at",
                                          "confidence", "count", "jpeg")}
            for key in ("id", "node_id", "name"):
                string(value[key], 350 if key == "id" else 160)
            if value["source"] not in ("oak", "replay"):
                raise ValueError("Invalid event source")
            number(value["at"], 0, 4e9); number(value["confidence"], 0, 1)
            number(value["count"], 1, 100); jpeg(value["jpeg"])
            checked_events.append(value)
        with self.lock:
            for node, age in checked_nodes:
                if node["id"] == self.node_id or age > 60:
                    continue
                existing = self.nodes.get(node["id"])
                version = (node["started"], node["seq"])
                if existing and version <= (existing[0]["started"], existing[0]["seq"]):
                    continue
                if not existing and len(self.nodes) >= MAX_NODES:
                    continue
                self.nodes[node["id"]] = (node, time.monotonic() - age)
            for event in checked_events:
                if event["id"] not in self.events:
                    self.events[event["id"]] = event
            self._trim_events()

    def image(self, node_id=None, event_id=None):
        with self.lock:
            if event_id is not None:
                encoded = self.events.get(event_id, {}).get("jpeg", "")
            else:
                encoded = self.nodes.get(node_id, ({},))[0].get("jpeg", "")
            return base64.b64decode(encoded) if encoded else None

    def add_peer(self, url):
        url = peer_url(url)
        with self.lock:
            if len(self.peers) >= 8 and url not in self.peers:
                raise ValueError("Maximum eight direct peers")
            self.peers.setdefault(url, dict(url=url, status="connecting", detail="", last_ok=None))
        return url


def synchronize(state, key):
    opener = build_opener(ProxyHandler({}), NoRedirect())

    def exchange(url, payload):
        try:
            peer_url(url)
            request = Request(url + "/api/peer/sync", payload, headers={
                "Content-Type": "application/json", "X-TurtleHero-Key": key,
            })
            started = time.monotonic()
            with opener.open(request, timeout=2) as response:
                body = response.read(MAX_BODY + 1)
                if len(body) > MAX_BODY:
                    raise ValueError("Peer message exceeds size limit")
                packet = json.loads(body)
            # Charge the whole request duration conservatively to received observations.
            for node in packet.get("nodes", []):
                delay = time.monotonic() - started
                node["age"] = node.get("age", 0) + delay
                if node.get("frame_age") is not None:
                    node["frame_age"] += delay
            state.merge(packet)
            update = dict(status="connected", detail="Exchanging observations", last_ok=time.time())
        except Exception as error:
            update = dict(status="offline", detail=str(error)[:300])
            if isinstance(error, HTTPError):
                error.close()
        with state.lock:
            if url in state.peers:
                state.peers[url].update(update)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        while not state.stop.is_set():
            state.heartbeat()
            with state.lock:
                peers = list(state.peers)
            if peers:
                # Cap shared historical images to keep each exchange bounded.
                packet = state.snapshot(images=True)
                for event in packet["events"]:
                    event["jpeg"] = ""
                payload = json.dumps(packet, allow_nan=False).encode()
                list(pool.map(lambda url: exchange(url, payload), peers))
            state.stop.wait(1)
