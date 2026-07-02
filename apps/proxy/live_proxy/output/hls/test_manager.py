"""
Unit tests for the HLS output manager's LL-HLS descriptor lifecycle. The manager
depends on Django/Redis modules that are not importable standalone, so this test
injects lightweight stubs into sys.modules BEFORE importing the manager, then
drives its store/publish methods directly against a fake Redis. Run standalone:

    python3 -m unittest apps.proxy.live_proxy.output.hls.test_manager

The pure segmenter/renderer invariants are covered separately in test_segmenter.
"""

import json
import sys
import types
import unittest


# --- Stub the manager's external dependencies before importing it -----------

def _install_stub_modules():
    """Inject fake core/config/redis-key/buffer modules so manager.py imports."""

    class FakeRedis:
        """Dict-backed Redis supporting the subset the manager uses, plus call
        counters the tests assert on (expire refreshes, descriptor writes)."""

        def __init__(self):
            self.store = {}
            self.expire_calls = []   # (key, ttl)
            self.setex_calls = []    # (key, ttl, value)

        def get(self, key):
            return self.store.get(key)

        def set(self, key, value, nx=False, ex=None):
            if nx and key in self.store:
                return None
            self.store[key] = value
            return True

        def setex(self, key, ttl, value):
            self.store[key] = value
            self.setex_calls.append((key, ttl, value))
            return True

        def expire(self, key, ttl):
            self.expire_calls.append((key, ttl))
            return key in self.store

        def delete(self, *keys):
            for k in keys:
                self.store.pop(k, None)
            return len(keys)

    class FakeBuffer:
        """Stand-in for FMP4StreamBuffer: put_fragment INCRs the index and the
        chunk index doubles as the HLS media sequence, exactly like the real one.
        ``fail_next`` forces one put_fragment failure for the reset test."""

        def __init__(self, channel_id, redis_client=None, fmt="hls"):
            self.channel_id = channel_id
            self.redis_client = redis_client if redis_client is not None else FakeRedis()
            self.fmt = fmt
            self.index = 0
            self.chunk_ttl = 60
            self.fail_next = False
            self.fragments = []

        def put_fragment(self, data):
            if self.fail_next:
                self.fail_next = False
                return False
            self.index += 1
            self.fragments.append(data)
            return True

        def cleanup_redis(self):
            pass

    # Shared fake redis instances so tests and the manager see the same store.
    buffer_redis = FakeRedis()
    client_redis = FakeRedis()

    class RedisClient:
        _buffer = buffer_redis
        _client = client_redis

        @staticmethod
        def get_buffer():
            return RedisClient._buffer

        @staticmethod
        def get_client():
            return RedisClient._client

    class ConfigHelper:
        @staticmethod
        def get(key, default=None):
            return default

        @staticmethod
        def new_client_behind_seconds():
            return 0

    class RedisKeys:
        @staticmethod
        def output_playlist(cid, fmt):
            return f"output:{fmt}:{cid}:playlist"

        @staticmethod
        def output_state(cid, fmt):
            return f"output:{fmt}:{cid}:state"

        @staticmethod
        def output_owner(cid, fmt):
            return f"output:{fmt}:{cid}:owner"

        @staticmethod
        def output_part(cid, fmt, seq, part):
            return f"output:{fmt}:{cid}:part:{seq}:{part}"

    class _Logger:
        def info(self, *a, **k):
            pass

        debug = info
        warning = info
        error = info

    def get_logger():
        return _Logger()

    # core.utils.RedisClient
    core = types.ModuleType("core")
    core_utils = types.ModuleType("core.utils")
    core_utils.RedisClient = RedisClient
    core.utils = core_utils
    sys.modules["core"] = core
    sys.modules["core.utils"] = core_utils

    # apps.proxy.live_proxy.output.fmp4[.buffer].FMP4StreamBuffer
    # (manager.py's `from ..fmp4.buffer import ...` resolves under .output)
    fmp4_pkg = types.ModuleType("apps.proxy.live_proxy.output.fmp4")
    fmp4_pkg.__path__ = []  # mark as package
    fmp4_buffer = types.ModuleType("apps.proxy.live_proxy.output.fmp4.buffer")
    fmp4_buffer.FMP4StreamBuffer = FakeBuffer
    sys.modules["apps.proxy.live_proxy.output.fmp4"] = fmp4_pkg
    sys.modules["apps.proxy.live_proxy.output.fmp4.buffer"] = fmp4_buffer

    # apps.proxy.live_proxy.{redis_keys,config_helper,utils}
    rk_mod = types.ModuleType("apps.proxy.live_proxy.redis_keys")
    rk_mod.RedisKeys = RedisKeys
    sys.modules["apps.proxy.live_proxy.redis_keys"] = rk_mod

    ch_mod = types.ModuleType("apps.proxy.live_proxy.config_helper")
    ch_mod.ConfigHelper = ConfigHelper
    sys.modules["apps.proxy.live_proxy.config_helper"] = ch_mod

    utils_mod = types.ModuleType("apps.proxy.live_proxy.utils")
    utils_mod.get_logger = get_logger
    sys.modules["apps.proxy.live_proxy.utils"] = utils_mod

    return RedisClient, FakeBuffer


_RedisClient, _FakeBuffer = _install_stub_modules()

from . import manager as mgr  # noqa: E402  (import after stubs installed)
from .segmenter import Segment, Part  # noqa: E402


def _fresh_redises():
    """Reset the shared fake redis stores between tests."""
    _RedisClient._buffer = _RedisClient.get_buffer().__class__()
    _RedisClient._client = _RedisClient.get_client().__class__()


class ManagerLifecycleTests(unittest.TestCase):
    def setUp(self):
        _fresh_redises()
        self.buffer_redis = _RedisClient.get_buffer()
        self.client_redis = _RedisClient.get_client()
        self.m = mgr.HLSOutputManager("chan1234abcd", ts_buffer=object(), worker_id="w1")
        self.playlist_key = mgr.RedisKeys.output_playlist("chan1234abcd", "hls")
        self.owner_key = mgr.RedisKeys.output_owner("chan1234abcd", "hls")
        self.state_key = mgr.RedisKeys.output_state("chan1234abcd", "hls")

    def _descriptor(self):
        raw = self.client_redis.get(self.playlist_key)
        return json.loads(raw) if raw else None

    def test_frozen_constants_computed(self):
        # adv_target = ceil(2*4) = 8; adv_part = 0.5*1.12 = 0.56.
        self.assertEqual(self.m.adv_target, 8)
        self.assertEqual(self.m.adv_part, 0.56)

    def test_no_empty_window_publish(self):
        # Parts before any segment must NOT publish a descriptor (the degenerate
        # zero-segment playlist was the cold-start black screen).
        self.m._store_part(Part(b"\x47" * 188, 0.5, independent=True))
        self.assertIsNone(self._descriptor())
        # The part bytes are still stored so they are ready when the segment lands.
        self.assertTrue(any(":part:" in k for k in self.buffer_redis.store))
        # The first stored segment anchors the window and publishes.
        self.m._store_segment(Segment(b"\x47" * 188, 4.0))
        desc = self._descriptor()
        self.assertIsNotNone(desc)
        self.assertTrue(desc["window"])
        self.assertEqual(desc["adv_target"], 8)
        self.assertEqual(desc["part_target"], 0.56)  # frozen adv_part, not raw 0.5

    def test_store_part_no_phantom_when_buffer_absent(self):
        # A missing buffer Redis must not advertise a part whose bytes were never
        # stored (every request for that URI would 404 after a 3s block).
        self.m._store_segment(Segment(b"\x47" * 188, 4.0))  # anchor window
        self.m.segment_buffer.redis_client = None
        before = list(self.m._building_parts)
        self.m._store_part(Part(b"\x47" * 188, 0.5, independent=True))
        self.assertEqual(self.m._building_parts, before)  # not appended

    def test_atomic_publish_gate(self):
        # _store_part(publish=False) stores bytes but must not refresh the
        # descriptor (used for a final part whose closing segment publishes next).
        self.m._store_segment(Segment(b"\x47" * 188, 4.0))  # anchor window
        n_before = len([c for c in self.client_redis.setex_calls if c[0] == self.playlist_key])
        self.m._store_part(Part(b"\x47" * 188, 0.5, independent=True), publish=False)
        n_after = len([c for c in self.client_redis.setex_calls if c[0] == self.playlist_key])
        self.assertEqual(n_after, n_before)  # no publish for the suppressed part

    def test_store_segment_failure_resets_building_state(self):
        # Seed some in-progress LL state, then fail the fragment write.
        self.m._building_parts = [[0.5, True], [0.5, False]]
        self.m._building_seq = 5
        self.m._building_disc = True
        self.m.segment_buffer.fail_next = True
        self.m._store_segment(Segment(b"\x47" * 188, 4.0))
        self.assertEqual(self.m._building_parts, [])
        self.assertIsNone(self.m._building_seq)
        self.assertFalse(self.m._building_disc)
        # Nothing published and no window growth on a failed store.
        self.assertIsNone(self._descriptor())
        self.assertEqual(self.m._window, [])

    def test_window_entries_carry_pdt_and_disc(self):
        self.m._store_segment(Segment(b"\x47" * 188, 4.0, discontinuity=True))
        entry = self._descriptor()["window"][-1]
        self.assertIn("pdt", entry)
        self.assertTrue(entry["pdt"].endswith("+00:00"))
        self.assertTrue(entry["disc"])

    def test_media_sequence_never_decreases_and_window_caps(self):
        seqs = []
        for _ in range(mgr.DEFAULT_WINDOW_SIZE + 3):
            self.m._store_segment(Segment(b"\x47" * 188, 4.0))
            seqs.append(self._descriptor()["window"][-1]["seq"])
        self.assertEqual(seqs, sorted(seqs))            # monotonic (RFC 8216 6.2.2)
        self.assertEqual(len(self._descriptor()["window"]), mgr.DEFAULT_WINDOW_SIZE)

    def test_owner_heartbeat_refreshes_then_stops(self):
        self.m.running = True
        self.client_redis.set(self.owner_key, "w1")     # we own it
        self.client_redis.set(self.state_key, "active")
        self.m._store_segment(Segment(b"\x47" * 188, 4.0))
        # Both keys refreshed while owned.
        refreshed = {k for (k, _ttl) in self.client_redis.expire_calls}
        self.assertIn(self.owner_key, refreshed)
        self.assertIn(self.state_key, refreshed)
        self.assertTrue(self.m.running)
        # Ownership moves to another worker -> loop must stop cleanly.
        self.client_redis.set(self.owner_key, "w2")
        self.m._store_segment(Segment(b"\x47" * 188, 4.0))
        self.assertFalse(self.m.running)

    def test_restart_seeds_descriptor_constants_and_window(self):
        # Pre-populate a descriptor as if a prior worker had been running, with
        # DIFFERENT frozen constants, then construct a fresh manager.
        prior = {
            "window": [{"seq": 41, "dur": 5.0, "disc": False, "pdt": "x"},
                       {"seq": 42, "dur": 5.0, "disc": False, "pdt": "y"}],
            "adv_target": 9,
            "part_target": 0.62,
        }
        self.client_redis.set(self.playlist_key, json.dumps(prior))
        m2 = mgr.HLSOutputManager("chan1234abcd", ts_buffer=object(), worker_id="w9")
        self.assertEqual(m2._window, prior["window"])   # seeded, MSN continuity
        self.assertEqual(m2.adv_target, 9)              # reuse stored constant
        self.assertEqual(m2.adv_part, 0.62)             # not recomputed to 0.56

    def test_part_key_ttl_covers_advertised_lifetime(self):
        # rfc8216bis 6.2.2: a part must stay fetchable while listed (~3 target
        # durations) AND ~one playlist duration after removal (~3 more).
        self.assertGreaterEqual(mgr.PART_KEY_TTL, 3 * self.m.adv_target + 3 * self.m.adv_target)

    def test_chunk_ttl_sized_to_window(self):
        # RFC 8216 6.2.2: storage must back the advertised window, not the 60s
        # default (10 segments * (4+3) + 30 = 100s).
        self.assertGreaterEqual(self.m.segment_buffer.chunk_ttl, 100)


if __name__ == "__main__":
    unittest.main()
