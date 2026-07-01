"""
MPEG-TS HLS segmenter - pure packet-copy splitting, no remux.

The live proxy's source ring already guarantees 188-byte packet alignment
(StreamBuffer.add_chunk), and TS segments are first-class HLS citizens
(RFC 8216 section 3.2), so producing HLS from the ring is a matter of
CUTTING the existing packets into keyframe-aligned segments. No bytes are
rewritten, no subprocess is spawned.

This module is intentionally dependency-free (stdlib only, no Django or
Redis imports) so the parsing logic is unit-testable in isolation.

Segmentation rules:
- A segment may only begin on a video keyframe access unit. Keyframes are
  detected via the adaptation-field random_access_indicator when the
  provider sets it, with a fallback NAL-header scan (H.264 IDR/SPS,
  H.265 IRAP/parameter sets) for providers that do not.
- Segment duration is measured from video PES PTS deltas, cut at the
  first keyframe at or after the target duration.
- Every emitted segment is prefixed with the most recently seen PAT and
  PMT packets so each segment decodes independently, as HLS requires.
"""

TS_PACKET_SIZE = 188
TS_SYNC_BYTE = 0x47

# ISO 13818-1 / ATSC stream_type values
VIDEO_STREAM_TYPES = {
    0x01: "mpeg1",
    0x02: "mpeg2",
    0x1B: "h264",
    0x24: "h265",
}

PTS_CLOCK = 90000.0
# 33-bit PTS wraps every ~26.5 hours; treat large negative deltas as a wrap.
PTS_WRAP = 1 << 33


class Segment:
    """One finished HLS media segment."""

    __slots__ = ("data", "duration", "discontinuity")

    def __init__(self, data, duration, discontinuity=False):
        self.data = data
        self.duration = duration
        self.discontinuity = discontinuity


class Part:
    """One Low-Latency HLS partial segment: a sub-second slice of the
    in-progress segment, emitted before the segment closes so a client can
    play within a part or two of the live edge. The concatenation of a
    segment's parts equals that segment's bytes. ``independent`` is True only
    for the first part of a segment (it begins on the keyframe + PAT/PMT, so a
    client may start decoding there)."""

    __slots__ = ("data", "duration", "independent")

    def __init__(self, data, duration, independent=False):
        self.data = data
        self.duration = float(duration)
        self.independent = independent


def packet_pid(packet):
    """13-bit PID of a TS packet."""
    return ((packet[1] & 0x1F) << 8) | packet[2]


def packet_pusi(packet):
    """payload_unit_start_indicator flag."""
    return bool(packet[1] & 0x40)


def packet_payload_offset(packet):
    """Byte offset of the payload within the packet, or None if no payload."""
    afc = (packet[3] >> 4) & 0x03
    if afc == 0x01:
        return 4
    if afc == 0x03:
        af_len = packet[4]
        offset = 5 + af_len
        return offset if offset < TS_PACKET_SIZE else None
    return None


def packet_random_access(packet):
    """adaptation-field random_access_indicator, when an AF is present."""
    afc = (packet[3] >> 4) & 0x03
    if afc in (0x02, 0x03) and packet[4] > 0:
        return bool(packet[5] & 0x40)
    return False


def parse_pat(packet):
    """Return the PMT PID of the first non-zero program, or None."""
    base = packet_payload_offset(packet)
    if base is None or base + 1 >= TS_PACKET_SIZE:
        return None
    pointer = packet[base]
    section = base + 1 + pointer
    # table_id(1) section_length(2) tsid(2) ver(1) sec(1) last(1) = 8 bytes,
    # then 4-byte program entries.
    offset = section + 8
    while offset + 3 < TS_PACKET_SIZE:
        program_number = (packet[offset] << 8) | packet[offset + 1]
        pid = ((packet[offset + 2] & 0x1F) << 8) | packet[offset + 3]
        if program_number != 0:
            return pid
        offset += 4
    return None


def parse_pmt(packet):
    """Return (video_pid, video_stream_type) from a PMT packet, or (None, None)."""
    base = packet_payload_offset(packet)
    if base is None or base + 1 >= TS_PACKET_SIZE:
        return None, None
    pointer = packet[base]
    section = base + 1 + pointer
    if section + 12 >= TS_PACKET_SIZE:
        return None, None
    section_length = ((packet[section + 1] & 0x0F) << 8) | packet[section + 2]
    program_info_length = ((packet[section + 10] & 0x0F) << 8) | packet[section + 11]
    offset = section + 12 + program_info_length
    section_end = min(section + 3 + section_length - 4, TS_PACKET_SIZE - 1)

    while offset + 4 < section_end:
        stream_type = packet[offset]
        es_pid = ((packet[offset + 1] & 0x1F) << 8) | packet[offset + 2]
        es_info_length = ((packet[offset + 3] & 0x0F) << 8) | packet[offset + 4]
        if stream_type in VIDEO_STREAM_TYPES:
            return es_pid, stream_type
        offset += 5 + es_info_length
    return None, None


def extract_pts(packet):
    """PTS in seconds from a PES header starting in this packet, or None."""
    base = packet_payload_offset(packet)
    if base is None or base + 13 >= TS_PACKET_SIZE:
        return None
    if packet[base] != 0x00 or packet[base + 1] != 0x00 or packet[base + 2] != 0x01:
        return None
    flags = packet[base + 7]
    if not (flags & 0x80):
        return None
    b = packet
    pts = (
        ((b[base + 9] >> 1) & 0x07) << 30
        | b[base + 10] << 22
        | ((b[base + 11] >> 1) & 0x7F) << 15
        | b[base + 12] << 7
        | (b[base + 13] >> 1)
    )
    return pts / PTS_CLOCK


def starts_keyframe(packet, video_stream_type):
    """
    Does this PUSI video packet open a keyframe access unit?

    Prefers the adaptation-field random_access_indicator; falls back to
    scanning visible NAL start codes. Encoders emit parameter sets
    immediately before IDR/IRAP frames, so SPS/VPS in the first packet is
    a reliable keyframe marker even when the keyframe NAL itself starts
    in a later packet of the same PES.
    """
    if packet_random_access(packet):
        return True

    base = packet_payload_offset(packet)
    if base is None or base + 9 >= TS_PACKET_SIZE:
        return False
    header_len = packet[base + 8]
    i = base + 9 + header_len
    end = TS_PACKET_SIZE - 4
    while i < end:
        if packet[i] == 0x00 and packet[i + 1] == 0x00:
            nal_start = -1
            if packet[i + 2] == 0x01:
                nal_start = i + 3
            elif packet[i + 2] == 0x00 and i + 3 < end and packet[i + 3] == 0x01:
                nal_start = i + 4
            if 0 < nal_start < TS_PACKET_SIZE:
                if video_stream_type == 0x24:
                    # H.265: nal_unit_type in bits 1-6 of the first byte.
                    nal_type = (packet[nal_start] >> 1) & 0x3F
                    # IRAP (16-21) or VPS/SPS/PPS (32-34)
                    if 16 <= nal_type <= 21 or 32 <= nal_type <= 34:
                        return True
                else:
                    # H.264: nal_unit_type in bits 0-4.
                    nal_type = packet[nal_start] & 0x1F
                    # IDR (5) or SPS (7)
                    if nal_type in (5, 7):
                        return True
                i = nal_start
                continue
        i += 1
    return False


class TSSegmenter:
    """
    Stateful packet-copy segmenter. Feed it raw TS bytes (any chunking);
    it returns finished Segment objects as keyframe boundaries are crossed.
    """

    def __init__(self, target_duration=4.0, part_target=0.0):
        self.target_duration = float(target_duration)
        # Low-Latency HLS: when > 0, emit Part objects every ~part_target
        # seconds of the in-progress segment. 0 disables LL-HLS (segments only),
        # which keeps the parser backward compatible for the non-LL path/tests.
        self.part_target = float(part_target)
        self._pending = bytearray()
        self._current = bytearray()
        self._pat_packet = None
        self._pmt_packet = None
        self._pmt_pid = None
        self._video_pid = None
        self._video_stream_type = None
        self._segment_start_pts = None
        self._collecting = False
        self._pending_discontinuity = False
        self._current_discontinuity = False
        # Part accounting for the in-progress segment.
        self._part_offset = 0        # byte index into _current where the open part began
        self._part_start_pts = None  # PTS at the open part's first frame
        self._part_index = 0         # 0-based part number within the current segment

    @property
    def video_detected(self):
        return self._video_pid is not None

    @property
    def video_codec(self):
        """Detected video codec family name (e.g. "h264", "h265"), or None
        until the PMT has been parsed. Used to advertise the codec to
        clients and to gate formats that a given HLS client cannot decode
        (notably HEVC-in-MPEG-TS, which AVFoundation refuses)."""
        return VIDEO_STREAM_TYPES.get(self._video_stream_type)

    def flag_discontinuity(self):
        """Mark that the NEXT emitted segment follows a stream discontinuity
        (provider failover, buffer skip-ahead)."""
        self._pending_discontinuity = True
        # PTS timeline may jump arbitrarily across the discontinuity.
        self._segment_start_pts = None

    def feed(self, data):
        """Consume raw TS bytes; return a list of events (possibly empty).

        With ``part_target == 0`` every event is a finished ``Segment`` (the
        classic non-LL contract). With ``part_target > 0`` the list interleaves
        ``Part`` events (emitted as the in-progress segment fills) with the
        ``Segment`` that closes them; a segment's ``Part`` events always precede
        its ``Segment`` in the stream. Dispatch by type at the call site."""
        events = []
        self._pending.extend(data)

        while len(self._pending) >= TS_PACKET_SIZE:
            if self._pending[0] != TS_SYNC_BYTE:
                sync = self._pending.find(bytes([TS_SYNC_BYTE]))
                if sync < 0:
                    self._pending.clear()
                    break
                del self._pending[:sync]
                continue
            # Require the next packet to also be in sync (or be the tail) so
            # a stray 0x47 in payload cannot fake an alignment point.
            if (
                len(self._pending) >= TS_PACKET_SIZE + 1
                and self._pending[TS_PACKET_SIZE] != TS_SYNC_BYTE
            ):
                del self._pending[:1]
                continue

            packet = bytes(self._pending[:TS_PACKET_SIZE])
            del self._pending[:TS_PACKET_SIZE]
            events.extend(self._handle_packet(packet))

        return events

    def _handle_packet(self, packet):
        events = []
        pid = packet_pid(packet)

        if pid == 0:
            self._pat_packet = packet
            if self._pmt_pid is None:
                self._pmt_pid = parse_pat(packet)
            return events
        if self._pmt_pid is not None and pid == self._pmt_pid:
            self._pmt_packet = packet
            video_pid, stream_type = parse_pmt(packet)
            if video_pid is not None:
                # Re-learned continuously so PID/codec changes across
                # provider failovers are tolerated.
                self._video_pid = video_pid
                self._video_stream_type = stream_type
            return events

        if self._video_pid is None:
            return events

        if pid == self._video_pid and packet_pusi(packet):
            pts = extract_pts(packet)
            keyframe = starts_keyframe(packet, self._video_stream_type)

            if not self._collecting:
                if keyframe:
                    self._begin_segment(pts)
            elif keyframe and pts is not None:
                cut = False
                duration = self.target_duration
                if self._segment_start_pts is None:
                    # Discontinuity reset the timeline: cut here.
                    cut = True
                else:
                    elapsed = pts - self._segment_start_pts
                    if elapsed < 0:
                        elapsed += PTS_WRAP / PTS_CLOCK
                    if elapsed >= self.target_duration:
                        duration = elapsed
                        cut = True
                if cut:
                    # Flush the segment's trailing bytes as its final part, then
                    # the segment itself, then open the next segment (its part 0
                    # begins on this keyframe).
                    self._emit_final_part(events, pts)
                    events.append(self._finish_segment(duration))
                    self._begin_segment(pts)
                else:
                    # Keyframe before the target: still a PES boundary a part
                    # may close on.
                    self._maybe_emit_part(events, pts)
            else:
                # Collecting, ordinary PES boundary: close the open part if due.
                self._maybe_emit_part(events, pts)

        if self._collecting:
            self._current.extend(packet)
        return events

    def _maybe_emit_part(self, events, pts):
        """Close the open part if it has reached part_target at this PES
        boundary. The part is the bytes accumulated since the previous boundary
        and does NOT include the triggering packet (which opens the next part)."""
        if self.part_target <= 0 or pts is None or self._part_start_pts is None:
            return
        elapsed = pts - self._part_start_pts
        if elapsed < 0:
            elapsed += PTS_WRAP / PTS_CLOCK
        if elapsed < self.part_target:
            return
        data = bytes(self._current[self._part_offset:])
        if not data:
            return
        events.append(Part(data, elapsed, independent=(self._part_index == 0)))
        self._part_offset = len(self._current)
        self._part_start_pts = pts
        self._part_index += 1

    def _emit_final_part(self, events, end_pts):
        """Flush the in-progress segment's trailing bytes as its last part when
        the segment is cut, so a segment's parts tile the whole segment."""
        if self.part_target <= 0:
            return
        data = bytes(self._current[self._part_offset:])
        if not data:
            return
        duration = self.part_target
        if end_pts is not None and self._part_start_pts is not None:
            d = end_pts - self._part_start_pts
            if d < 0:
                d += PTS_WRAP / PTS_CLOCK
            # Guard against a discontinuity/timeline jump producing a nonsense
            # value; a legitimate final part is at most ~part_target long.
            if 0 < d <= 2 * self.part_target:
                duration = d
        events.append(Part(data, duration, independent=(self._part_index == 0)))

    def _begin_segment(self, pts):
        self._current = bytearray()
        if self._pat_packet:
            self._current.extend(self._pat_packet)
        if self._pmt_packet:
            self._current.extend(self._pmt_packet)
        self._segment_start_pts = pts
        self._collecting = True
        self._current_discontinuity = self._pending_discontinuity
        self._pending_discontinuity = False
        # Part 0 begins here, at offset 0 so it carries the PAT+PMT and the
        # keyframe (making it the segment's INDEPENDENT part).
        self._part_offset = 0
        self._part_start_pts = pts
        self._part_index = 0

    def _finish_segment(self, duration):
        if duration <= 0 or duration > 4 * self.target_duration:
            duration = self.target_duration
        segment = Segment(
            bytes(self._current),
            float(duration),
            discontinuity=self._current_discontinuity,
        )
        self._current = bytearray()
        self._current_discontinuity = False
        return segment


def _append_part_line(lines, part_name, seq, index, part):
    """Append one #EXT-X-PART line. part is [duration, independent]."""
    duration = part[0]
    independent = len(part) > 1 and part[1]
    uri = part_name.format(seq=seq, part=index)
    line = f'#EXT-X-PART:DURATION={duration:.5f},URI="{uri}"'
    if independent:
        line += ",INDEPENDENT=YES"
    lines.append(line)


def render_media_playlist(window, target_duration, segment_name="{seq}.ts",
                          part_target=0.0, parts_by_seq=None, building=None,
                          part_name="p{seq}.{part}.ts"):
    """
    Render an HLS media playlist from a window of segment descriptors:
    [{"seq": int, "dur": float, "disc": bool}, ...]. Segment URIs are relative
    so they resolve against the playlist URL.

    Default output is RFC 8216 version 3. When ``part_target > 0`` the output
    is a Low-Latency HLS playlist (version 10) that additionally carries
    #EXT-X-SERVER-CONTROL (CAN-BLOCK-RELOAD), #EXT-X-PART-INF, #EXT-X-PART lines
    for the most recent segments and the in-progress segment, and an
    #EXT-X-PRELOAD-HINT for the next part. ``parts_by_seq`` maps a completed
    segment's seq (str) to its list of [duration, independent] parts;
    ``building`` is {"seq": int, "parts": [[duration, independent], ...]} for
    the segment currently being produced.
    """
    ll = bool(part_target and part_target > 0)
    parts_by_seq = parts_by_seq or {}
    building = building or {}
    building_parts = building.get("parts") or []
    building_seq = building.get("seq")

    # Nothing published yet: minimal valid live playlist, client retries.
    if not window and not building_parts:
        return (
            "#EXTM3U\n"
            "#EXT-X-VERSION:3\n"
            "#EXT-X-INDEPENDENT-SEGMENTS\n"
            # Ceil so a fractional target never rounds DOWN below a real EXTINF
            # (RFC 8216 4.3.3.1).
            f"#EXT-X-TARGETDURATION:{int(max(target_duration, 1) + 0.999)}\n"
            "#EXT-X-MEDIA-SEQUENCE:0\n"
        )

    if window:
        max_duration = max(entry["dur"] for entry in window)
        total_duration = sum(entry["dur"] for entry in window)
        media_sequence = window[0]["seq"]
    else:
        # Only in-progress parts exist (the first segment has not closed yet):
        # let an LL client start from a partial segment for a fast first frame.
        max_duration = target_duration
        total_duration = sum(p[0] for p in building_parts)
        media_sequence = building_seq if building_seq is not None else 0

    # TARGETDURATION must be >= every EXTINF rounded up (RFC 8216 4.3.3.1).
    advertised_target = int(max_duration + 0.999)

    version = 10 if ll else 3
    lines = [
        "#EXTM3U",
        f"#EXT-X-VERSION:{version}",
        "#EXT-X-INDEPENDENT-SEGMENTS",
        f"#EXT-X-TARGETDURATION:{advertised_target}",
        f"#EXT-X-MEDIA-SEQUENCE:{media_sequence}",
    ]

    if ll:
        # Advertise a PART-TARGET slightly above the emit threshold: a part is
        # closed on the first frame boundary at or past the threshold, so its
        # real duration can exceed the threshold by up to one frame. PART-TARGET
        # must be >= every EXT-X-PART DURATION (RFC 8216bis), and PART-HOLD-BACK
        # must be >= 3 x PART-TARGET.
        adv_part = round(part_target + 0.05, 3)
        hold_back = round(3 * adv_part, 3)
        lines.append(f"#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK={hold_back:.3f}")
        lines.append(f"#EXT-X-PART-INF:PART-TARGET={adv_part:.3f}")
        # Start near the live edge (a couple of hold-backs) so the low-latency
        # win is realised instead of joining 3 target-durations back.
        start_offset = min(2 * hold_back, total_duration)
    else:
        # Non-LL: join ~3 target durations back, the safe no-stall convention.
        start_offset = min(3 * advertised_target, total_duration)

    if total_duration:
        lines.append(f"#EXT-X-START:TIME-OFFSET:-{start_offset:.3f},PRECISE=YES")

    # Carry EXT-X-PART lines only for the last two completed segments so parts
    # stay available within PART-HOLD-BACK without bloating the playlist with
    # parts for the entire window.
    part_seqs = {entry["seq"] for entry in window[-2:]} if (ll and window) else set()

    for entry in window:
        if entry.get("disc"):
            lines.append("#EXT-X-DISCONTINUITY")
        if entry["seq"] in part_seqs:
            for i, part in enumerate(parts_by_seq.get(str(entry["seq"]), [])):
                _append_part_line(lines, part_name, entry["seq"], i, part)
        lines.append(f"#EXTINF:{entry['dur']:.3f},")
        lines.append(segment_name.format(seq=entry["seq"]))

    # In-progress segment: its parts (no EXTINF yet) followed by a preload hint
    # for the next part, which the client fetches with a blocking GET.
    if ll and building_parts and building_seq is not None:
        for i, part in enumerate(building_parts):
            _append_part_line(lines, part_name, building_seq, i, part)
        next_uri = part_name.format(seq=building_seq, part=len(building_parts))
        lines.append(f'#EXT-X-PRELOAD-HINT:TYPE=PART,URI="{next_uri}"')

    return "\n".join(lines) + "\n"
