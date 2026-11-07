"""Network/freshness contracts; no model weights or camera required."""
import base64
import copy
import http.client
import json
import threading
import time
import unittest
from unittest.mock import patch

from dashboard.network import MAX_BODY, State, peer_url, synchronize
from turtlehero_dashboard import make_server, lan_urls

JPEG = base64.b64encode(b'\xff\xd8test\xff\xd9').decode()
BOX = {"box": [10, 20, 50, 60], "score": .9}


def station(name, source="oak"):
    state = State(name, name, source, .63, [])
    state.heartbeat()
    return state


def positive(state):
    state.frame(JPEG, [copy.deepcopy(BOX)], time.time(), 12, 4, state.generation)


def node(state, name):
    return next(n for n in state.snapshot()["nodes"] if n["id"] == name)


class StateTests(unittest.TestCase):
    def test_lan_address_fallback_excludes_loopback(self):
        def output(command, **kwargs):
            if command[0] == "/sbin/ifconfig":
                return b"lo0:\n inet 127.0.0.1 netmask 0xff000000\nen0:\n inet 192.168.1.10 netmask 0xffffff00\n"
            raise OSError("ipconfig unavailable")
        with patch("turtlehero_dashboard.socket.getaddrinfo", return_value=[]), \
                patch("turtlehero_dashboard.subprocess.check_output", side_effect=output), \
                patch("turtlehero_dashboard.Path.is_file", return_value=True):
            self.assertEqual(lan_urls(8765), ["http://192.168.1.10:8765"])

    def test_origin_and_event_survive_three_hops_without_duplicates(self):
        a, b, c = station("a"), station("b"), station("c")
        positive(a)
        b.merge(a.snapshot(images=True))
        c.merge(b.snapshot(images=True))
        a.merge(c.snapshot(images=True))
        b.merge(c.snapshot(images=True))
        self.assertEqual(node(c, "a")["detections"], [BOX])
        for state in (a, b, c):
            self.assertEqual(len(state.events), 1)
            self.assertEqual(next(iter(state.events.values()))["node_id"], "a")
        self.assertEqual(c.image("a"), base64.b64decode(JPEG))

    def test_loop_does_not_refresh_offline_observations(self):
        with patch("dashboard.network.time.monotonic", return_value=100) as clock:
            a, b, c = station("a"), station("b"), station("c")
            positive(a)
            b.merge(a.snapshot(images=True))
            clock.return_value = 106
            c.merge(b.snapshot(images=True))
            clock.return_value = 109
            b.merge(c.snapshot(images=True))
            self.assertFalse(node(b, "a")["fresh"])
            self.assertFalse(node(c, "a")["online"])
            self.assertEqual(node(b, "a")["detections"], [])
            self.assertAlmostEqual(node(c, "a")["frame_age"], 9)

    def test_heartbeats_do_not_keep_old_frames_fresh(self):
        with patch("dashboard.network.time.monotonic", return_value=100) as clock:
            a = station("a")
            positive(a)
            clock.return_value = 110
            a.heartbeat()
            self.assertTrue(node(a, "a")["online"])
            self.assertFalse(node(a, "a")["fresh"])

    def test_source_switch_rejects_in_flight_old_frame(self):
        a = station("a", "replay")
        revision = a.generation
        positive(a)
        a.control("oak", .7)
        a.frame(JPEG, [BOX], time.time(), 12, 4, revision)
        a.status("running", "old source", revision)
        a.heartbeat()
        self.assertFalse(node(a, "a")["fresh"])
        self.assertIsNone(a.image("a"))
        self.assertEqual(node(a, "a")["source"], "oak")
        self.assertEqual(len(a.events), 1)

    def test_bad_message_cannot_partially_commit(self):
        a, b = station("a"), station("b")
        positive(a)
        for mutation in (lambda p: p["events"][0].update(confidence=float("nan")),
                         lambda p: p["nodes"][0].update(detections=[None]),
                         lambda p: p["nodes"][0].update(jpeg="not a jpeg")):
            packet = a.snapshot(images=True)
            mutation(packet)
            with self.assertRaises(ValueError):
                b.merge(packet)
            self.assertNotIn("a", b.nodes)
            self.assertEqual(len(b.events), 0)

    def test_newer_source_restart_replaces_old_state(self):
        a, b = station("a"), station("b")
        positive(a)
        b.merge(a.snapshot(images=True))
        old = a.snapshot(images=True)
        restarted = copy.deepcopy(old)
        restarted["nodes"][0].update(started=a.local["started"] + 1, seq=0,
                                      boot="new-boot", detections=[])
        b.merge(restarted)
        b.merge(old)
        self.assertEqual(node(b, "a")["boot"], "new-boot")
        self.assertEqual(node(b, "a")["detections"], [])

    def test_sightings_group_frames_and_memory_is_bounded(self):
        with patch("dashboard.network.time.monotonic", return_value=100) as clock:
            a = station("a")
            for _ in range(20):
                positive(a)
            self.assertEqual(len(a.events), 1)
            for i in range(110):
                clock.return_value = 104 + 4 * i
                positive(a)
            self.assertEqual(len(a.events), 100)
            self.assertEqual(len(a.snapshot()["events"]), 30)

    def test_long_dead_nodes_are_not_gossiped(self):
        with patch("dashboard.network.time.monotonic", return_value=100) as clock:
            a, b = station("a"), station("b")
            b.merge(a.snapshot(images=True))
            clock.return_value = 100000
            b.heartbeat()
            c = station("c")
            c.merge(b.snapshot(images=True))
            self.assertNotIn("a", c.nodes)

    def test_peer_url_rejects_invalid_and_external_destinations(self):
        for value in ([], "https://127.0.0.1:8765", "http://user@127.0.0.1", "8.8.8.8", "0.0.0.0", "http://127.0.0.1/path"):
            with self.assertRaises(ValueError):
                peer_url(value)
        self.assertEqual(peer_url("127.0.0.1:8765/"), "http://127.0.0.1:8765")


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.state = station("http")
        self.server = make_server(self.state, "127.0.0.1", 0, "test-key")
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(3)

    def post(self, path, body, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        conn.request("POST", path, json.dumps(body),
                     {"Content-Type": "application/json", **(headers or {})})
        response = conn.getresponse()
        result = response.status, json.loads(response.read())
        conn.close()
        return result

    def test_share_link_follows_network_changes_without_losing_peers(self):
        self.state.add_peer("http://192.168.1.163:8765")

        def addresses():
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
            conn.request("GET", "/api/state")
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            result = json.loads(response.read())
            conn.close()
            return result["urls"]

        with patch("turtlehero_dashboard.time") as clock, \
                patch("turtlehero_dashboard.lan_urls") as discover:
            clock.monotonic.return_value = 100
            discover.return_value = [f"http://10.151.159.250:{self.port}"]
            self.assertEqual(addresses(), discover.return_value)
            clock.monotonic.return_value = 111
            discover.return_value = [f"http://192.168.1.254:{self.port}"]
            self.assertEqual(addresses(), discover.return_value)
            clock.monotonic.return_value = 122
            discover.return_value = []
            self.assertEqual(addresses(), [])
        self.assertIn("http://192.168.1.163:8765", self.state.peers)

    def test_peer_key_and_browser_origin_are_enforced(self):
        packet = station("remote").snapshot(images=True)
        self.assertEqual(self.post("/api/peer/sync", packet)[0], 403)
        self.assertEqual(self.post("/api/peer/sync", packet,
                                   {"X-TurtleHero-Key": "test-key", "Origin": "http://foreign"})[0], 403)
        status, result = self.post("/api/peer/sync", packet, {"X-TurtleHero-Key": "test-key"})
        self.assertEqual(status, 200)
        self.assertIn("remote", [n["id"] for n in result["nodes"]])
        self.assertEqual(self.post("/api/control", {"source": "none"}, {"Origin": "http://foreign"})[0], 403)
        self.assertEqual(self.state.source, "oak")

    def test_bad_controls_and_oversize_requests_fail_without_mutation(self):
        for path, body in (("/api/peers", {"url": []}), ("/api/control", {"source": "video"}),
                           ("/api/control", {"confidence": float("nan")})):
            self.assertEqual(self.post(path, body)[0], 400)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        conn.request("POST", "/api/control", b"", {"Content-Type": "application/json", "Content-Length": str(MAX_BODY + 1)})
        self.assertEqual(conn.getresponse().status, 400)
        conn.close()
        self.assertEqual(self.state.source, "oak")

    def test_sync_exchanges_both_directions_and_recovers_after_failure(self):
        remote = station("remote")
        positive(remote)
        positive(self.state)
        url = remote.add_peer(f"http://127.0.0.1:{self.port}")
        worker = threading.Thread(target=synchronize, args=(remote, "wrong-key"), daemon=True)
        worker.start()

        def until(predicate):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if predicate():
                    return
                time.sleep(.03)
            self.fail("Peer did not reach the expected state")

        try:
            until(lambda: remote.peers[url]["status"] == "offline")
        finally:
            remote.stop.set()
            worker.join(3)
        remote.stop.clear()
        worker = threading.Thread(target=synchronize, args=(remote, "test-key"), daemon=True)
        worker.start()
        try:
            until(lambda: remote.peers[url]["status"] == "connected")
            self.assertEqual(node(remote, "http")["detections"], [BOX])
            self.assertEqual(node(self.state, "remote")["detections"], [BOX])
            self.assertEqual(len(remote.events), 2)
            self.assertEqual(len(self.state.events), 2)
            self.assertEqual(remote.image("http"), base64.b64decode(JPEG))
            # The same sync worker must recover when the actual server goes away.
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(3)
            until(lambda: remote.peers[url]["status"] == "offline")
            self.server = make_server(self.state, "127.0.0.1", self.port, "test-key")
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            until(lambda: remote.peers[url]["status"] == "connected")
            self.assertEqual(len(remote.events), 2)
        finally:
            remote.stop.set()
            worker.join(3)


if __name__ == "__main__":
    unittest.main()
