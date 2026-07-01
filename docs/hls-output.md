# Native HLS output for live channels

Dispatcharr can serve any live channel as a real HLS media playlist so that
HLS-native clients (Apple AVPlayer on iOS/tvOS/macOS, Safari, hls.js in the
browser, VLC, ffmpeg, QuickTime) can play it directly, with no client-side
remuxing. This document is the integration contract for client developers.

## Requesting HLS

A client asks for HLS on the normal stream endpoint using a standard query
parameter (no bespoke headers, no separate auth path):

```
GET /proxy/ts/stream/<channel_uuid>?output_format=hls
```

Aliases:

- `?output=hls` (XC-style parameter name) is accepted as well.
- Xtream Codes clients may use the `.m3u8` extension on the XC stream URL,
  e.g. `/live/<user>/<pass>/<id>.m3u8`, which resolves to the same output.

You can also set HLS as the server's **Default Output Format** (System >
Settings > Stream Settings), after which plain stream requests return HLS.
Clients that want a specific format should always set `output_format`
explicitly rather than relying on the server default.

## Redirect and playlist URLs

The stream request runs the normal init/auth/client-registration path and
then returns an **HTTP 302** to a client-scoped media playlist:

```
302 Location: /proxy/hls/<channel_uuid>/<client_id>/index.m3u8
```

Clients **must follow redirects** and **must preserve the redirected base
URL**. Segment URIs in the playlist are relative (`<seq>.ts`) and resolve
against the playlist URL, i.e. `/proxy/hls/<channel_uuid>/<client_id>/<seq>.ts`.
Do not hand-construct segment URLs against the pre-redirect `/proxy/ts/...`
path; fetch the redirected playlist and let the player resolve segments.

Every playlist and segment request touches the client record, so a polling
player keeps its session alive and a stopped player is reaped by the existing
ghost-client heartbeat.

## Playlist shape

The media playlist is a standard live RFC 8216 (version 3) playlist:

- `#EXT-X-VERSION:3`
- `#EXT-X-INDEPENDENT-SEGMENTS` (every segment starts on an IDR keyframe and is
  self-contained, so a player can begin decoding or seek to any segment boundary
  without fetching a prior segment)
- `#EXT-X-TARGETDURATION:<n>` (integer, a true upper bound on every segment)
- `#EXT-X-MEDIA-SEQUENCE:<n>` (monotonically increasing as segments roll off)
- `#EXT-X-START:TIME-OFFSET:-<n>,PRECISE=YES` (server-pinned live-edge start,
  ~3 target durations back, clamped to the window; makes the join point the same
  across players. A player that sets its own start offset still overrides it, and
  players that do not understand the tag ignore it)
- `#EXT-X-DISCONTINUITY` before a segment that follows a stream discontinuity
- No `#EXT-X-ENDLIST` (the stream is live; players keep reloading)
- A rolling window of segments (default 10 x ~4s; real length is floored by the
  source GOP, so ~2s-GOP content yields ~4.5-5.5s segments)

Segments are MPEG-TS, each prefixed with the current PAT and PMT so it decodes
independently. No transcoding or remuxing is performed; the source packets are
split on keyframe boundaries.

## MIME types and CORS

- Playlist responses use `Content-Type: application/vnd.apple.mpegurl` and
  `Cache-Control: no-cache` (the playlist changes on every reload).
- Segment responses use `Content-Type: video/mp2t`. A finished segment is
  immutable (a `<seq>.ts` always maps to the same bytes, and sequence numbers are
  never reused), so it is served `Cache-Control: public, max-age=60, immutable`
  to let browsers/hls.js and any CDN reuse in-window re-requests. Expired
  segments (404) are not cached.
- All HLS responses (playlist, segments, redirect, errors) carry permissive
  CORS headers and answer `OPTIONS` preflight, so a browser hls.js / Safari
  MSE player can fetch them cross-origin. Native players ignore CORS.

## Codec support

HLS output carries the source codec untouched in MPEG-TS segments.

- **H.264 (AVC) video is supported** and is broadly playable across all HLS
  clients.
- **HEVC / H.265 video is not served over HLS.** AVFoundation (AVPlayer,
  Safari) refuses HEVC in MPEG-TS, so serving it would black-screen those
  clients with no error. For an HEVC channel the playlist endpoint returns
  **HTTP 415** with a message directing the client to the MPEG-TS or fMP4
  output format. A client should treat a 415 on the playlist as "HLS not
  available for this channel" and fall back accordingly. HEVC-in-HLS would
  require fMP4/CMAF segments and is future work.
- Audio (AAC, AC-3, E-AC-3 including Atmos-as-JOC) is passed through and plays
  on clients that support it; Dolby audio bitstreams to a receiver on tvOS via
  the native player.

## Server settings

- `HLS_SEGMENT_DURATION` (default 4 seconds) - target segment length.
- `HLS_WINDOW_SIZE` (default 10) - number of segments retained in the rolling
  live playlist. A longer window adds no latency (players start near the live
  edge) and gives a client that briefly falls behind more room to catch up
  before a segment rolls off.
