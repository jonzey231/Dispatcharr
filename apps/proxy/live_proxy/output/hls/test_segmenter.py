"""
Unit tests for the HLS TS segmenter. Dependency-free (stdlib unittest, no
Django/Redis), so they run standalone:

    python3 -m unittest apps.proxy.live_proxy.output.hls.test_segmenter
"""

import unittest

from .segmenter import (
    Part,
    Segment,
    TSSegmenter,
    TS_PACKET_SIZE,
    extract_pts,
    packet_pid,
    parse_pat,
    parse_pmt,
    render_media_playlist,
    starts_keyframe,
)

VIDEO_PID = 256
PMT_PID = 4096
H264 = 0x1B


def make_packet(pid, payload, pusi=False, random_access=False):
    """Build one 188-byte TS packet with the given payload bytes."""
    header = bytearray(4)
    header[0] = 0x47
    header[1] = ((0x40 if pusi else 0x00) | (pid >> 8)) & 0xFF
    header[2] = pid & 0xFF

    if random_access:
        # adaptation field present + payload
        body_len = TS_PACKET_SIZE - 4 - 2 - len(payload)
        assert body_len >= 0, "payload too large for packet with AF"
        header[3] = 0x30  # AF + payload
        af = bytearray([1 + body_len, 0x40])  # af_length, RAI flag
        af.extend(b"\xff" * body_len)
        packet = bytes(header) + bytes(af) + bytes(payload)
    else:
        header[3] = 0x10  # payload only
        packet = bytes(header) + bytes(payload)
        packet += b"\xff" * (TS_PACKET_SIZE - len(packet))
    assert len(packet) == TS_PACKET_SIZE
    return packet


def make_pat():
    # pointer + table header (8 bytes from table_id) + one program entry
    payload = bytearray([0x00])                      # pointer_field
    payload += bytes([0x00, 0xB0, 0x0D, 0x00, 0x01, 0xC1, 0x00, 0x00])
    payload += bytes([0x00, 0x01, 0xE0 | (PMT_PID >> 8), PMT_PID & 0xFF])
    payload += bytes(4)                              # CRC placeholder
    return make_packet(0, payload, pusi=True)


def make_pmt():
    payload = bytearray([0x00])                      # pointer_field
    # table_id, section_length covers from after length to CRC
    es_loop = bytes([H264, 0xE0 | (VIDEO_PID >> 8), VIDEO_PID & 0xFF, 0xF0, 0x00])
    section_length = 9 + len(es_loop) + 4            # post-length header + loop + CRC
    payload += bytes([0x02, 0xB0 | (section_length >> 8), section_length & 0xFF])
    payload += bytes([0x00, 0x01, 0xC1, 0x00, 0x00]) # tsid, ver, sec, last
    payload += bytes([0xE0 | (VIDEO_PID >> 8), VIDEO_PID & 0xFF, 0xF0, 0x00])  # PCR PID, prog info len
    payload += es_loop
    payload += bytes(4)                              # CRC placeholder
    return make_packet(PMT_PID, payload, pusi=True)


def make_video_pes(pts_seconds, keyframe, use_rai=False):
    """A PUSI video packet opening a PES with the given PTS."""
    pts = int(pts_seconds * 90000)
    p = bytearray()
    p += bytes([0x00, 0x00, 0x01, 0xE0, 0x00, 0x00])  # PES start, stream_id, length
    p += bytes([0x80, 0x80, 0x05])                    # flags, PTS-only, header len 5
    p += bytes([
        0x21 | (((pts >> 30) & 0x07) << 1),
        (pts >> 22) & 0xFF,
        0x01 | (((pts >> 15) & 0x7F) << 1),
        (pts >> 7) & 0xFF,
        0x01 | ((pts & 0x7F) << 1),
    ])
    # NAL start code + type
    if keyframe and not use_rai:
        p += bytes([0x00, 0x00, 0x00, 0x01, 0x65])    # IDR slice
    else:
        p += bytes([0x00, 0x00, 0x00, 0x01, 0x41])    # non-IDR slice
    return make_packet(VIDEO_PID, p, pusi=True, random_access=keyframe and use_rai)


def make_filler():
    return make_packet(VIDEO_PID, b"\x00" * 20)


class ParserTests(unittest.TestCase):
    def test_pat_pmt_roundtrip(self):
        self.assertEqual(parse_pat(make_pat()), PMT_PID)
        video_pid, stream_type = parse_pmt(make_pmt())
        self.assertEqual(video_pid, VIDEO_PID)
        self.assertEqual(stream_type, H264)

    def test_pts_roundtrip(self):
        packet = make_video_pes(1234.5, keyframe=True)
        self.assertAlmostEqual(extract_pts(packet), 1234.5, places=3)

    def test_keyframe_detection_nal_and_rai(self):
        self.assertTrue(starts_keyframe(make_video_pes(0, keyframe=True), H264))
        self.assertFalse(starts_keyframe(make_video_pes(0, keyframe=False), H264))
        self.assertTrue(starts_keyframe(make_video_pes(0, keyframe=True, use_rai=True), H264))

    def test_pid_extraction(self):
        self.assertEqual(packet_pid(make_pat()), 0)
        self.assertEqual(packet_pid(make_pmt()), PMT_PID)


def feed_stream(segmenter, gop_seconds, gop_count, start_pts=10.0, fillers_per_gop=5):
    """Feed `gop_count` GOPs of `gop_seconds` each; returns finished segments."""
    out = []
    for i in range(gop_count):
        pts = start_pts + i * gop_seconds
        out += segmenter.feed(make_video_pes(pts, keyframe=True))
        for j in range(fillers_per_gop):
            out += segmenter.feed(make_filler())
            out += segmenter.feed(make_video_pes(pts + (j + 1) * 0.2, keyframe=False))
    return out


class SegmenterTests(unittest.TestCase):
    def make_started(self, target=4.0):
        seg = TSSegmenter(target_duration=target)
        seg.feed(make_pat())
        seg.feed(make_pmt())
        return seg

    def test_cuts_on_keyframes_at_target_duration(self):
        seg = self.make_started(target=4.0)
        # 2-second GOPs: cuts must land every 2 GOPs (4.0s)
        finished = feed_stream(seg, gop_seconds=2.0, gop_count=7)
        self.assertEqual(len(finished), 3)
        for s in finished:
            self.assertAlmostEqual(s.duration, 4.0, places=3)

    def test_segments_start_with_pat_pmt(self):
        seg = self.make_started()
        finished = feed_stream(seg, gop_seconds=4.0, gop_count=3)
        self.assertGreaterEqual(len(finished), 1)
        for s in finished:
            self.assertEqual(s.data[0], 0x47)
            self.assertEqual(packet_pid(s.data[:TS_PACKET_SIZE]), 0)  # PAT first
            second = s.data[TS_PACKET_SIZE:2 * TS_PACKET_SIZE]
            self.assertEqual(packet_pid(second), PMT_PID)             # PMT second

    def test_no_segment_before_first_keyframe(self):
        seg = self.make_started()
        out = []
        out += seg.feed(make_video_pes(5.0, keyframe=False))
        out += seg.feed(make_filler())
        self.assertEqual(out, [])
        self.assertFalse(seg._collecting)

    def test_discontinuity_flag_propagates(self):
        seg = self.make_started(target=2.0)
        finished = feed_stream(seg, gop_seconds=2.0, gop_count=2)
        seg.flag_discontinuity()
        # Timeline jumps far ahead, as after a provider failover
        finished += feed_stream(seg, gop_seconds=2.0, gop_count=3, start_pts=9000.0)
        flagged = [s for s in finished if s.discontinuity]
        self.assertEqual(len(flagged), 1)

    def test_resync_after_garbage(self):
        seg = self.make_started(target=2.0)
        seg.feed(b"\xde\xad\xbe\xef" * 33)  # garbage, not packet-aligned
        finished = feed_stream(seg, gop_seconds=2.0, gop_count=4)
        self.assertGreaterEqual(len(finished), 2)

    def test_pts_wrap_tolerated(self):
        seg = self.make_started(target=2.0)
        wrap_edge = (1 << 33) / 90000.0
        out = seg.feed(make_video_pes(wrap_edge - 1.0, keyframe=True))
        out += seg.feed(make_video_pes(1.0, keyframe=True))  # wrapped
        durations = [s.duration for s in out]
        for d in durations:
            self.assertGreater(d, 0)
            self.assertLessEqual(d, 8.0)


class PlaylistTests(unittest.TestCase):
    def test_render_basic(self):
        window = [
            {"seq": 7, "dur": 4.0, "disc": False},
            {"seq": 8, "dur": 4.2, "disc": False},
            {"seq": 9, "dur": 3.9, "disc": True},
        ]
        text = render_media_playlist(window, 4)
        self.assertIn("#EXTM3U", text)
        self.assertIn("#EXT-X-VERSION:3", text)
        self.assertIn("#EXT-X-TARGETDURATION:5", text)       # ceil(4.2)
        self.assertIn("#EXT-X-MEDIA-SEQUENCE:7", text)
        self.assertIn("#EXTINF:4.200,", text)
        self.assertIn("8.ts", text)
        self.assertNotIn("#EXT-X-ENDLIST", text)             # live
        self.assertIn("#EXT-X-INDEPENDENT-SEGMENTS", text)   # segments are IDR-aligned
        # Live-edge start pinned ~3 target-durations back, clamped to the window
        # (min(3*5, 4.0+4.2+3.9) = min(15, 12.1) = 12.1)
        self.assertIn("#EXT-X-START:TIME-OFFSET=-12.100,PRECISE=YES", text)
        # Discontinuity tag must precede its segment
        lines = text.splitlines()
        self.assertEqual(lines[lines.index("#EXT-X-DISCONTINUITY") + 2], "9.ts")

    def test_render_empty_window(self):
        text = render_media_playlist([], 4)
        self.assertIn("#EXT-X-MEDIA-SEQUENCE:0", text)
        self.assertIn("#EXT-X-INDEPENDENT-SEGMENTS", text)
        self.assertIn("#EXT-X-TARGETDURATION:4", text)       # ceil(4)
        self.assertNotIn("#EXT-X-START", text)               # no segments to offset from

    def test_render_low_latency(self):
        window = [
            {"seq": 5, "dur": 4.0, "disc": False},
            {"seq": 6, "dur": 4.1, "disc": False},
        ]
        parts_by_seq = {"6": [[0.5, True], [0.5, False]]}
        building = {"seq": 7, "parts": [[0.5, True], [0.4, False]]}
        text = render_media_playlist(
            window, 4, part_target=0.5, parts_by_seq=parts_by_seq, building=building
        )
        self.assertIn("#EXT-X-VERSION:10", text)
        self.assertIn("#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK=", text)
        self.assertIn("#EXT-X-PART-INF:PART-TARGET=", text)
        # Parts of the last completed segment, with the first marked independent.
        self.assertIn('#EXT-X-PART:DURATION=0.50000,URI="p6.0.ts",INDEPENDENT=YES', text)
        self.assertIn('#EXT-X-PART:DURATION=0.50000,URI="p6.1.ts"', text)
        # In-progress segment's parts + a preload hint for the next part.
        self.assertIn('#EXT-X-PART:DURATION=0.50000,URI="p7.0.ts",INDEPENDENT=YES', text)
        self.assertIn('#EXT-X-PRELOAD-HINT:TYPE=PART,URI="p7.2.ts"', text)
        # The completed segments are still present for non-LL clients.
        self.assertIn("6.ts", text)
        # Same call without part_target stays a plain version-3 playlist.
        self.assertIn("#EXT-X-VERSION:3", render_media_playlist(window, 4))


class PartTests(unittest.TestCase):
    def _feed(self, seg, frames):
        events = []
        seg.feed(make_pat())
        seg.feed(make_pmt())
        for pts, keyframe in frames:
            events.extend(seg.feed(make_video_pes(pts, keyframe=keyframe)))
        return events

    def test_parts_tile_the_segment(self):
        seg = TSSegmenter(target_duration=4.0, part_target=0.5)
        frames = [(0.0, True)]
        t = 0.25
        while t < 4.0:
            frames.append((round(t, 2), False))
            t += 0.25
        frames.append((4.0, True))  # keyframe at/after target cuts the segment
        events = self._feed(seg, frames)

        parts = [e for e in events if isinstance(e, Part)]
        segments = [e for e in events if isinstance(e, Segment)]
        self.assertEqual(len(segments), 1)
        # ~8 parts of ~0.5s span the 4s segment (including the final tail part).
        self.assertGreaterEqual(len(parts), 6)
        # Only the first part carries the keyframe + PAT/PMT.
        self.assertTrue(parts[0].independent)
        self.assertFalse(parts[1].independent)
        # A segment's parts concatenate exactly to the segment bytes.
        self.assertEqual(b"".join(p.data for p in parts), segments[0].data)
        for p in parts:
            self.assertLessEqual(p.duration, 2 * 0.5)  # bounded near part_target

    def test_part_target_zero_emits_only_segments(self):
        seg = TSSegmenter(target_duration=4.0)  # part_target defaults to 0
        frames = [(0.0, True), (2.0, False), (4.0, True)]
        events = self._feed(seg, frames)
        self.assertTrue(all(isinstance(e, Segment) for e in events))
        self.assertEqual(len(events), 1)

    def test_bframe_reorder_gives_no_garbage_durations(self):
        # Decode-order PTS that dips below the previous frame like B-frame
        # reordering. A naive pts-start delta goes slightly negative and, if
        # treated as a 33-bit wrap, yields a ~95443s garbage part duration that
        # makes AVPlayer reject the playlist. Durations must stay sane.
        seg = TSSegmenter(target_duration=4.0, part_target=0.5)
        frames = [(0.0, True)]
        base = 0.0
        while base < 4.0:
            base += 0.25
            frames.append((round(base + 0.1, 3), False))  # ahead in presentation
            frames.append((round(base, 3), False))         # dips back (B-frame)
        frames.append((4.2, True))  # keyframe cuts the segment
        events = self._feed(seg, frames)
        parts = [e for e in events if isinstance(e, Part)]
        self.assertTrue(parts)
        for p in parts:
            self.assertGreater(p.duration, 0)
            self.assertLess(p.duration, seg.target_duration)  # never ~95443s


class VideoCodecTests(unittest.TestCase):
    def test_codec_none_before_pmt(self):
        seg = TSSegmenter()
        self.assertIsNone(seg.video_codec)
        self.assertFalse(seg.video_detected)

    def test_codec_h264_learned_from_pmt(self):
        seg = TSSegmenter()
        seg.feed(make_pat() + make_pmt())
        self.assertTrue(seg.video_detected)
        self.assertEqual(seg.video_codec, "h264")


if __name__ == "__main__":
    unittest.main()
