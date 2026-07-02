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
        # EXT-X-START is suppressed until the window is deep enough to honor a
        # frozen 3*TARGETDURATION join point (3*5 = 15s > 12.1s here), so the
        # advertised offset never changes across reloads (RFC 8216 6.2.1).
        self.assertNotIn("#EXT-X-START", text)
        # Discontinuity tag must precede its segment
        lines = text.splitlines()
        self.assertEqual(lines[lines.index("#EXT-X-DISCONTINUITY") + 2], "9.ts")

    def test_render_non_ll_start_frozen_once_deep(self):
        # Once the window holds >= 3*adv_target of content, EXT-X-START appears
        # and equals the frozen constant, unchanged as the window shifts.
        adv_target = 8
        w1 = [{"seq": i, "dur": 5.0, "disc": False} for i in range(6)]     # 30s
        w2 = [{"seq": i, "dur": 5.0, "disc": False} for i in range(1, 7)]  # shifted
        t1 = render_media_playlist(w1, 4, adv_target=adv_target)
        t2 = render_media_playlist(w2, 4, adv_target=adv_target)
        self.assertIn("#EXT-X-START:TIME-OFFSET=-24.000,PRECISE=YES", t1)
        self.assertIn("#EXT-X-START:TIME-OFFSET=-24.000,PRECISE=YES", t2)
        # And a shallow window (< 24s) suppresses it entirely.
        shallow = [{"seq": 0, "dur": 5.0, "disc": False}]
        self.assertNotIn("#EXT-X-START", render_media_playlist(shallow, 4, adv_target=adv_target))

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
        # part_target here is the FROZEN advertised PART-TARGET (adv_part), which
        # the manager computes and carries in the descriptor; render emits it and
        # PART-HOLD-BACK = 3x it verbatim, recomputing neither.
        text = render_media_playlist(
            window, 4, part_target=0.56, parts_by_seq=parts_by_seq, building=building,
            adv_target=8,
        )
        self.assertIn("#EXT-X-VERSION:10", text)
        self.assertIn("#EXT-X-TARGETDURATION:8", text)  # frozen, not ceil(4.1)=5
        self.assertIn("#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK=1.680", text)
        self.assertIn("#EXT-X-PART-INF:PART-TARGET=0.560", text)
        # LL never carries EXT-X-START: PART-HOLD-BACK positions the client.
        self.assertNotIn("#EXT-X-START", text)
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

    def test_targetduration_constant_across_window_shift(self):
        # RFC 8216 6.2.1: EXT-X-TARGETDURATION MUST NOT change across reloads.
        # Roll a window whose EXTINF maxima cross integer ceilings (4.604 -> 6.457
        # -> back down) and assert the advertised target is byte-identical and >=
        # every rounded EXTINF in all renders (4.3.3.1).
        adv_target = 8
        windows = [
            [{"seq": 1, "dur": 4.604}, {"seq": 2, "dur": 4.421}, {"seq": 3, "dur": 5.005}],
            [{"seq": 2, "dur": 4.421}, {"seq": 3, "dur": 5.005}, {"seq": 4, "dur": 6.457}],
            [{"seq": 3, "dur": 5.005}, {"seq": 4, "dur": 6.457}, {"seq": 5, "dur": 4.100}],
        ]
        lines = []
        for w in windows:
            text = render_media_playlist(w, 4, part_target=0.56, adv_target=adv_target)
            td = [l for l in text.splitlines() if l.startswith("#EXT-X-TARGETDURATION:")]
            self.assertEqual(td, ["#EXT-X-TARGETDURATION:8"])
            for entry in w:
                self.assertLessEqual(round(entry["dur"]), adv_target)
            lines.append(td[0])
        self.assertEqual(len(set(lines)), 1)  # identical across all shifts

    def test_part_target_and_holdback_constant(self):
        # PART-TARGET / PART-HOLD-BACK are frozen constants: render must emit the
        # passed adv_part verbatim regardless of the listed part maxima (0.501 vs
        # 0.534), so AVPlayer's blocking-reload timing model never sees them flap.
        w1 = [{"seq": 6, "dur": 4.1}]
        w2 = [{"seq": 6, "dur": 4.1}]
        t1 = render_media_playlist(
            w1, 4, part_target=0.56, parts_by_seq={"6": [[0.501, True]]},
            building={"seq": 7, "parts": [[0.501, True]]}, adv_target=8)
        t2 = render_media_playlist(
            w2, 4, part_target=0.56, parts_by_seq={"6": [[0.534, True]]},
            building={"seq": 7, "parts": [[0.534, True]]}, adv_target=8)
        for text in (t1, t2):
            self.assertIn("#EXT-X-PART-INF:PART-TARGET=0.560", text)
            self.assertIn("PART-HOLD-BACK=1.680", text)

    def test_ll_program_date_time_and_building_discontinuity(self):
        # Apple's LL profile requires EXT-X-PROGRAM-DATE-TIME per segment; the
        # building segment's discontinuity must be signalled before its first part
        # (never inserted retroactively).
        window = [
            {"seq": 5, "dur": 4.0, "disc": False, "pdt": "2026-07-01T00:00:00.000+00:00"},
            {"seq": 6, "dur": 4.1, "disc": True, "pdt": "2026-07-01T00:00:04.000+00:00"},
        ]
        building = {"seq": 7, "parts": [[0.5, True]], "disc": True}
        text = render_media_playlist(
            window, 4, part_target=0.56,
            parts_by_seq={"6": [[0.5, True]]}, building=building, adv_target=8)
        lines = text.splitlines()
        # Segment 5 has no listed parts, so its PDT sits immediately before EXTINF.
        pdt5 = lines.index("#EXT-X-PROGRAM-DATE-TIME:2026-07-01T00:00:00.000+00:00")
        self.assertEqual(lines[pdt5 + 1], "#EXTINF:4.000,")
        # Segment 6's PDT precedes its EXTINF (parts intervene) and follows the
        # discontinuity tag for that segment.
        pdt6 = lines.index("#EXT-X-PROGRAM-DATE-TIME:2026-07-01T00:00:04.000+00:00")
        extinf6 = lines.index("#EXTINF:4.100,")
        self.assertLess(pdt6, extinf6)
        disc_positions = [i for i, l in enumerate(lines) if l == "#EXT-X-DISCONTINUITY"]
        self.assertTrue(any(i < pdt6 for i in disc_positions))  # seg-6 disc before its PDT
        # Building discontinuity precedes the building segment's first part line.
        first_building_part = lines.index('#EXT-X-PART:DURATION=0.50000,URI="p7.0.ts",INDEPENDENT=YES')
        self.assertTrue(any(i < first_building_part for i in disc_positions))


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

    def test_forced_cut_bounds_segment_to_frozen_target(self):
        # Keyframe drought: no keyframe appears for well past the target. The
        # segmenter must force a cut at max_segment_duration so no EXTINF can
        # exceed the frozen advertised TARGETDURATION (RFC 8216 4.3.3.1).
        seg = TSSegmenter(target_duration=4.0, max_segment_duration=8.0)
        frames = [(0.0, True)]
        t = 0.5
        while t <= 9.0:              # only one keyframe (at 0); pure drought after
            frames.append((round(t, 3), False))
            t += 0.5
        events = self._feed(seg, frames)
        segments = [e for e in events if isinstance(e, Segment)]
        self.assertTrue(segments)                       # a cut was forced
        for s in segments:
            self.assertLessEqual(s.duration, 8.0)       # never exceeds the ceiling
            self.assertLessEqual(round(s.duration), 8)  # nor its rounded EXTINF

    def test_disc_cut_reports_measured_span_not_target(self):
        # A discontinuity cut must report the CLOSING segment's measured span,
        # not the nominal target, and must not be poisoned by the post-jump
        # keyframe's PTS (RFC 8216 4.3.2.1).
        seg = TSSegmenter(target_duration=4.0)
        seg.feed(make_pat())
        seg.feed(make_pmt())
        seg.feed(make_video_pes(0.0, keyframe=True))
        seg.feed(make_video_pes(2.0, keyframe=False))   # last real frame at 2.0s
        seg.flag_discontinuity()
        events = seg.feed(make_video_pes(100.0, keyframe=True))  # timeline jump
        segments = [e for e in events if isinstance(e, Segment)]
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0].duration, 2.0, places=3)  # measured, not 4.0
        self.assertFalse(segments[0].discontinuity)     # the jump tags the NEXT segment

    def test_part_ceiling_clamps_over_target_tail(self):
        # A coarse frame cadence makes a tail longer than the advertised ceiling;
        # every emitted part duration must still be <= the ceiling so no
        # EXT-X-PART exceeds the frozen PART-TARGET (RFC 8216bis 4.4.4.9).
        ceiling = 0.56
        seg = TSSegmenter(target_duration=4.0, part_target=0.5, part_ceiling=ceiling)
        # ~1.5fps: PES boundaries 0.7s apart, so a raw tail would reach ~0.7s.
        frames = [(0.0, True), (0.7, False), (1.4, False), (2.1, False),
                  (2.8, False), (3.5, False), (4.2, True)]
        events = self._feed(seg, frames)
        parts = [e for e in events if isinstance(e, Part)]
        self.assertTrue(parts)
        for p in parts:
            self.assertLessEqual(p.duration, ceiling + 1e-9)

    def test_non_final_parts_land_in_85_percent_band(self):
        # At a realistic frame cadence the frozen adv_part (part_target * 1.12)
        # keeps every non-final part inside [0.85*adv_part, adv_part]
        # (RFC 8216bis 4.4.4.9), the last part of each segment exempt.
        part_target = 0.5
        adv_part = round(part_target * 1.12, 3)          # 0.56, as the manager freezes
        seg = TSSegmenter(target_duration=4.0, part_target=part_target)
        frames = [(0.0, True)]
        i = 1
        while i / 30.0 < 8.0:                            # 30fps, two 4s segments
            t = round(i / 30.0, 5)
            kf = abs(t - 4.0) < 1e-6                     # a keyframe at the 4s target
            frames.append((t, kf))
            i += 1
        events = self._feed(seg, frames)
        # Split parts by the Segment they precede; the last part before each
        # Segment is that segment's final (exempt) part.
        groups, cur = [], []
        for e in events:
            if isinstance(e, Part):
                cur.append(e)
            elif isinstance(e, Segment):
                groups.append(cur)
                cur = []
        self.assertTrue(any(len(g) >= 2 for g in groups))
        for g in groups:
            for p in g[:-1]:                             # non-final parts
                self.assertLessEqual(p.duration, adv_part)
                self.assertGreaterEqual(p.duration, 0.85 * adv_part)
            if g:
                self.assertLessEqual(g[-1].duration, adv_part)  # final part


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
