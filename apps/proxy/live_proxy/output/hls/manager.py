"""
HLS Output Manager

Reads from the shared TS Redis buffer, splits the stream into
keyframe-aligned TS segments (pure packet copy, no remux, no subprocess;
see segmenter.py), stores one segment per Redis chunk via the shared
format-parameterized output buffer, and maintains a rolling live playlist
descriptor in Redis that the playlist view renders per request.

One instance per channel per cluster - coordinated via the shared
output:{fmt}:owner lock, exactly like the fMP4 remux manager.
"""

import json
import threading
import time
from datetime import datetime, timedelta, timezone

from core.utils import RedisClient
from ..fmp4.buffer import FMP4StreamBuffer
from .segmenter import TSSegmenter, Segment, Part
from ...redis_keys import RedisKeys
from ...config_helper import ConfigHelper
from ...utils import get_logger

logger = get_logger()

# Output manager states stored in Redis (shared vocabulary with fMP4)
HLS_STATE_INITIALIZING = "initializing"
HLS_STATE_ACTIVE = "active"
HLS_STATE_STOPPED = "stopped"

# Redis TTL for state/owner/playlist keys
HLS_KEY_TTL = 3600

# Defaults; both overridable via proxy settings
DEFAULT_SEGMENT_DURATION = 4
# Retain 10 segments (~40s) in the rolling live window. A player starts
# near the live edge regardless of window length, so a longer window adds
# no latency; it only keeps older segments available so a client that
# briefly falls behind (a stall, a slow network hiccup) can still fetch the
# segment it is on instead of getting a 404 once it has rolled off. 6 (~24s)
# proved too tight for AVPlayer after a stall (CoreMedia -12938 / HTTP 404).
DEFAULT_WINDOW_SIZE = 10

# Low-Latency HLS partial-segment target (seconds). 0 disables LL-HLS
# (segments only). ~0.5s parts put the live edge within ~1.5s (PART-HOLD-BACK
# = 3 x PART-TARGET) for players that support Blocking Playlist Reload, while
# non-LL players ignore the part tags and use the whole segments unchanged.
DEFAULT_PART_TARGET = 0.5
# A part must stay fetchable while it is advertised (up to ~3 segments back) AND
# for ~one playlist duration after it rolls off (RFC 8216bis 6.2.2). At the 4s
# target that worst case is ~3*8 + 3*8 = 48s, so 60 covers it with margin. The
# old 15s expired advertised parts on long-segment channels, 404ing them during
# recovery.
PART_KEY_TTL = 60
# How many recent segments keep their parts in the descriptor and get their
# EXT-X-PART lines rendered. Matches Apple's ~3-target-durations guidance.
PARTS_RETAINED_SEGMENTS = 3


class HLSOutputManager:
    """
    Reads the TS Redis buffer for a channel, cuts keyframe-aligned HLS
    segments, and publishes them plus a rolling playlist window to Redis.
    """

    def __init__(self, channel_id, ts_buffer, worker_id, fmt='hls'):
        self.channel_id = channel_id
        self.ts_buffer = ts_buffer
        self.worker_id = worker_id
        self.fmt = fmt
        self.running = False
        self._thread = None

        self.segment_duration = ConfigHelper.get('HLS_SEGMENT_DURATION', DEFAULT_SEGMENT_DURATION)
        self.window_size = ConfigHelper.get('HLS_WINDOW_SIZE', DEFAULT_WINDOW_SIZE)
        self.part_target = ConfigHelper.get('HLS_PART_TARGET', DEFAULT_PART_TARGET)
        # Session constants, frozen once and carried in the descriptor so the
        # rendered playlist never changes them across reloads (RFC 8216 6.2.1 /
        # RFC 8216bis 6.2.1). Recomputing per render was the root cause of the
        # AVPlayer freezes (TARGETDURATION flapped 5/6/7, PART-TARGET 0.52/0.55).
        # adv_target: 2x the cut target (one GOP of headroom); the segmenter
        # force-cuts anything that would exceed it. adv_part: just above the emit
        # threshold so every part is <= it while non-final parts stay >= 85% of
        # it (RFC 8216bis 4.4.4.9). PART-HOLD-BACK is 3x adv_part.
        self.adv_target = int(2 * self.segment_duration + 0.999)
        self.adv_part = round(self.part_target * 1.12, 3) if self.part_target > 0 else 0.0

        # Same Redis-backed chunk store the fMP4 manager uses; it is
        # format-parameterized by design ("adding a new output format only
        # requires a new manager" - redis_keys.py). One HLS segment per
        # chunk; the chunk index doubles as the HLS media sequence number.
        self.segment_buffer = FMP4StreamBuffer(
            channel_id, redis_client=RedisClient.get_buffer(), fmt=fmt
        )
        # Size the chunk TTL to the advertised window + a playlist of post-removal
        # availability (RFC 8216 6.2.2); the default 60s cannot back a 10-segment
        # window of 5-6.5s segments.
        try:
            self.segment_buffer.chunk_ttl = max(
                self.segment_buffer.chunk_ttl,
                int(self.window_size * (self.segment_duration + 3) + 30),
            )
        except Exception:
            pass
        self._redis = RedisClient.get_client()
        self._window = []
        # Video codec family ("h264"/"h265"/...) learned from the PMT once
        # the segmenter has parsed it; surfaced in the playlist descriptor so
        # the playlist view can advertise it and refuse formats a client
        # cannot decode (HEVC-in-TS).
        self._video_codec = None
        # Low-Latency HLS part state. _building_seq is the media sequence the
        # in-progress segment will get when stored (put_fragment INCRs, so it is
        # the current index + 1); _building_parts accumulates [dur, independent]
        # for that segment; _parts_by_seq keeps a small tail of completed
        # segments' parts for the descriptor.
        self._building_seq = None
        self._building_parts = []
        # Whether the in-progress segment began after a discontinuity, mirrored
        # from the segmenter each loop iteration so building["disc"] is truthful
        # from the first published part (see _segmenter_loop).
        self._building_disc = False
        self._parts_by_seq = {}
        # Seed the window + frozen constants from an existing descriptor so a
        # worker restart/takeover never regresses MEDIA-SEQUENCE (RFC 8216 6.2.2).
        if self._redis:
            try:
                existing = self._redis.get(RedisKeys.output_playlist(self.channel_id, self.fmt))
                if existing:
                    prior = json.loads(existing)
                    if prior.get("window"):
                        self._window = prior["window"]
                    if prior.get("adv_target"):
                        self.adv_target = prior["adv_target"]
                    if prior.get("part_target"):
                        self.adv_part = prior["part_target"]
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API (same surface as FMP4RemuxManager)
    # ------------------------------------------------------------------

    def start(self):
        """Acquire the output owner lock and spawn the segmenter thread."""
        if not self._acquire_owner_lock():
            logger.info(f"[HLS:{self.channel_id}] Another worker owns HLS output, skipping start")
            return False

        self.running = True
        self._set_state(HLS_STATE_INITIALIZING)

        short_id = self.channel_id[:8]
        self._thread = threading.Thread(
            target=self._segmenter_loop, daemon=True,
            name=f"hls-seg-{short_id}"
        )
        self._thread.start()

        logger.info(
            f"[HLS:{self.channel_id}] Started "
            f"(target={self.segment_duration}s, window={self.window_size})"
        )
        return True

    def stop(self):
        """Stop the segmenter thread and clean up all Redis keys."""
        if not self.running:
            return
        self.running = False
        logger.info(f"[HLS:{self.channel_id}] Stopping")

        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=2)
            except Exception:
                pass

        self._cleanup_redis()
        logger.info(f"[HLS:{self.channel_id}] Stopped")

    # ------------------------------------------------------------------
    # Segmenter loop
    # ------------------------------------------------------------------

    def _segmenter_loop(self):
        """Read TS chunks from Redis and feed them through the segmenter."""
        segmenter = TSSegmenter(
            target_duration=self.segment_duration,
            part_target=self.part_target,
            max_segment_duration=self.adv_target,
            # Clamp emitted parts to exactly the advertised PART-TARGET (adv_part)
            # so no EXT-X-PART DURATION can exceed the frozen constant.
            part_ceiling=self.adv_part,
        )

        # Start behind live so the first segments cover the same window a
        # new TS client would receive, matching fMP4 writer positioning.
        behind_seconds = ConfigHelper.new_client_behind_seconds()
        start_index = self.ts_buffer.find_chunk_index_by_time(behind_seconds) if behind_seconds > 0 else None
        if start_index is None:
            start_index = self.ts_buffer.index
        local_index = start_index
        first_segment_stored = False
        logger.debug(
            f"[HLS:{self.channel_id}] Segmenter started at buffer index "
            f"{local_index} ({behind_seconds}s behind live)"
        )

        try:
            while self.running:
                chunks, new_index = self.ts_buffer.get_optimized_client_data(local_index)

                if chunks:
                    local_index = new_index
                    for chunk in chunks:
                        if not self.running:
                            break
                        events = segmenter.feed(chunk)
                        for i, event in enumerate(events):
                            self._video_codec = segmenter.video_codec
                            self._building_disc = segmenter.current_discontinuity
                            if isinstance(event, Part):
                                # Suppress the descriptor publish for a final part
                                # immediately followed by its Segment (the segment
                                # publish supersedes it microseconds later), so the
                                # transient state never advertises a PRELOAD-HINT
                                # for a part of a segment that is closing.
                                publish = not (i + 1 < len(events) and isinstance(events[i + 1], Segment))
                                self._store_part(event, publish=publish)
                                continue
                            self._store_segment(event)
                            if not first_segment_stored:
                                first_segment_stored = True
                                self._set_state(HLS_STATE_ACTIVE)
                                logger.info(
                                    f"[HLS:{self.channel_id}] First segment stored "
                                    f"({event.duration:.2f}s, {len(event.data)} bytes)"
                                )
                else:
                    if self.ts_buffer.index > local_index + 20:
                        # Fell too far behind (slow consumer / provider burst):
                        # skip forward and mark the gap for the playlist.
                        local_index = self.ts_buffer.index - 5
                        segmenter.flag_discontinuity()
                        logger.debug(
                            f"[HLS:{self.channel_id}] Skipped forward to index {local_index}"
                        )
                    time.sleep(0.05)

        except Exception as e:
            logger.error(f"[HLS:{self.channel_id}] Segmenter loop error: {e}", exc_info=True)
        finally:
            logger.debug(f"[HLS:{self.channel_id}] Segmenter loop exited")

    def _store_part(self, part, publish=True):
        """Store one Low-Latency HLS partial segment for the in-progress segment
        and refresh the descriptor so the live edge advances every ~part_target.
        ``publish=False`` stores the bytes but skips the descriptor refresh (used
        for a final part whose closing Segment publishes right after)."""
        if self.part_target <= 0:
            return
        if self._building_seq is None:
            # The in-progress segment takes the next media sequence number
            # (put_fragment INCRs the index when it is eventually stored).
            self._building_seq = self.segment_buffer.index + 1
        part_index = len(self._building_parts)
        # Store parts in the same buffer Redis as the segment chunks so the part
        # view reads them exactly like hls_segment reads chunks.
        buf = self.segment_buffer.redis_client
        if buf:
            try:
                buf.setex(
                    RedisKeys.output_part(self.channel_id, self.fmt, self._building_seq, part_index),
                    PART_KEY_TTL,
                    part.data,
                )
            except Exception as e:
                logger.error(f"[HLS:{self.channel_id}] Error storing part: {e}")
                return
        self._building_parts.append([round(part.duration, 5), bool(part.independent)])
        # Publish only once a full segment anchors the window: a descriptor with
        # an empty window renders a degenerate zero-segment playlist AVPlayer will
        # not start on (the cold-start black screen). The bytes are still stored
        # above so they are ready the moment the first segment closes.
        if publish and self._window:
            self._publish_playlist_state()

    def _store_segment(self, segment):
        """Store one finished segment and refresh the playlist descriptor."""
        if not self.segment_buffer.put_fragment(segment.data):
            # Redis write failed: drop the in-progress LL state so the next part
            # re-derives its seq from Redis rather than accumulating two segments'
            # parts under a stale seq (which would 404 every advertised part).
            self._building_parts = []
            self._building_seq = None
            self._building_disc = False
            return
        seq = self.segment_buffer.index
        # Wall-clock anchor for the segment START (Apple's Low-Latency profile
        # requires EXT-X-PROGRAM-DATE-TIME on all media playlists; it also drives
        # AVPlayer's recommendedTimeOffsetFromLive).
        seg_start = datetime.now(timezone.utc) - timedelta(seconds=segment.duration)
        self._window.append({
            "seq": seq,
            "dur": round(segment.duration, 3),
            "disc": bool(segment.discontinuity),
            "pdt": seg_start.isoformat(timespec="milliseconds"),
        })
        if len(self._window) > self.window_size:
            self._window = self._window[-self.window_size:]

        # The parts accumulated while building now belong to this completed
        # segment (its seq equals the seq tracked during building). Hand them
        # over, start a fresh in-progress segment, and prune old parts.
        if self.part_target > 0:
            self._parts_by_seq[str(seq)] = self._building_parts
            self._building_parts = []
            self._building_seq = self.segment_buffer.index + 1
            keep = {str(e["seq"]) for e in self._window[-PARTS_RETAINED_SEGMENTS:]}
            self._parts_by_seq = {k: v for k, v in self._parts_by_seq.items() if k in keep}

        self._publish_playlist_state()
        # Heartbeat ownership every stored segment (see broadcompat rationale):
        # a >1h stream must not silently lose the owner lock to a second worker.
        self._heartbeat_ownership()

        logger.debug(
            f"[HLS:{self.channel_id}] Segment {seq}: "
            f"{segment.duration:.2f}s, {len(segment.data)} bytes"
            f"{' [discontinuity]' if segment.discontinuity else ''}"
        )

    def _publish_playlist_state(self):
        """Write the rolling playlist descriptor to Redis for the playlist view
        to render on demand. Includes LL-HLS part data when enabled."""
        if not self._redis:
            return
        try:
            playlist_state = {
                "window": self._window,
                "target": self.segment_duration,
                # Frozen session constant so the rendered TARGETDURATION never
                # changes across reloads (RFC 8216 6.2.1).
                "adv_target": self.adv_target,
                "vcodec": self._video_codec,
            }
            if self.part_target > 0:
                # part_target carries the FROZEN advertised PART-TARGET (adv_part),
                # not the raw emit threshold, so PART-TARGET/PART-HOLD-BACK are
                # also constant across reloads (RFC 8216bis 6.2.1).
                playlist_state["part_target"] = self.adv_part
                playlist_state["parts"] = self._parts_by_seq
                playlist_state["building"] = {
                    "seq": self._building_seq,
                    "parts": self._building_parts,
                    "disc": self._building_disc,
                }
            self._redis.setex(
                RedisKeys.output_playlist(self.channel_id, self.fmt),
                HLS_KEY_TTL,
                json.dumps(playlist_state),
            )
        except Exception as e:
            logger.error(f"[HLS:{self.channel_id}] Error updating playlist state: {e}")

    # ------------------------------------------------------------------
    # Redis helpers (mirror FMP4RemuxManager)
    # ------------------------------------------------------------------

    def _acquire_owner_lock(self) -> bool:
        if not self._redis:
            return True
        owner_key = RedisKeys.output_owner(self.channel_id, self.fmt)
        acquired = self._redis.set(owner_key, self.worker_id, nx=True, ex=HLS_KEY_TTL)
        if acquired:
            return True
        existing = self._redis.get(owner_key)
        return existing == self.worker_id

    def _set_state(self, state: str):
        if self._redis:
            self._redis.setex(RedisKeys.output_state(self.channel_id, self.fmt), HLS_KEY_TTL, state)

    def _heartbeat_ownership(self):
        """Re-extend the owner lock + state TTL while we still own them; stop the
        loop if another worker has taken over. Called once per stored segment."""
        if not self._redis:
            return
        try:
            owner_key = RedisKeys.output_owner(self.channel_id, self.fmt)
            if self._redis.get(owner_key) == self.worker_id:
                self._redis.expire(owner_key, HLS_KEY_TTL)
                self._redis.expire(RedisKeys.output_state(self.channel_id, self.fmt), HLS_KEY_TTL)
            else:
                logger.info(f"[HLS:{self.channel_id}] Output ownership moved to another worker; stopping")
                self.running = False
        except Exception as e:
            logger.error(f"[HLS:{self.channel_id}] Ownership heartbeat error: {e}")

    def _cleanup_redis(self):
        """Delete all HLS output Redis keys for this channel."""
        if not self._redis:
            return
        try:
            keys_to_delete = [
                RedisKeys.output_state(self.channel_id, self.fmt),
                RedisKeys.output_owner(self.channel_id, self.fmt),
                RedisKeys.output_playlist(self.channel_id, self.fmt),
            ]
            self._redis.delete(*keys_to_delete)
            self.segment_buffer.cleanup_redis()
        except Exception as e:
            logger.error(f"[HLS:{self.channel_id}] Error during Redis cleanup: {e}")
