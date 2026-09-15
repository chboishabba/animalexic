#!/usr/bin/env python3
"""Candidate-only YouTube birdsong/video source adapter.

Reuses Animalexic's existing transient YouTube machinery (`FrameStreamer` in
`run_stereo_dispatch.py`) and its yt-dlp `download=False` resolution discipline.
This module does not promote source identity, species identity, behaviour,
physiology, geometry, or causality.  It only turns a URL/ID + resolved metadata
into an append-only source receipt that later audiovisual producers can consume.

The three initial seed videos come from the birdsong/situated-performance lane.
Known labels are deliberately source-role labels, not authoritative biological
identification.  Unknown metadata remains unresolved rather than guessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")


@dataclass(frozen=True)
class YouTubeSeed:
    url: str
    video_id: str
    source_role: str
    external_label: str | None = None
    external_label_basis: str | None = None


SEEDS: tuple[YouTubeSeed, ...] = (
    YouTubeSeed(
        "https://www.youtube.com/watch?v=Kz4SvP0c_VE",
        "Kz4SvP0c_VE",
        "candidate audiovisual example for magpie chatter / movement coupling",
        "magpie chatter",
        "external contextual page supplied the label; YouTube metadata still resolved separately",
    ),
    YouTubeSeed(
        "https://www.youtube.com/watch?v=8u_7lFB5iLg",
        "8u_7lFB5iLg",
        "candidate audiovisual birdsong/performance example; title/species unresolved until source metadata is acquired",
    ),
    YouTubeSeed(
        "https://www.youtube.com/watch?v=oYEYc8Ge3nw",
        "oYEYc8Ge3nw",
        "candidate audiovisual example for Australian-magpie singing / visible-body coupling",
        "Australian magpie singing",
        "external institutional/reference pages supplied the label; YouTube metadata still resolved separately",
    ),
)


def canonical_video_id(url: str) -> str:
    match = YOUTUBE_ID_RE.search(url)
    if not match:
        raise ValueError(f"unrecognized YouTube URL: {url}")
    return match.group(1)


def resolve_metadata(url: str) -> dict[str, Any]:
    """Resolve public source metadata without downloading media bytes."""
    import yt_dlp

    with yt_dlp.YoutubeDL(
        {
            "quiet": True,
            "skip_download": True,
            "noplaylist": True,
            "socket_timeout": 15,
            "extractor_args": {"youtube": {"player_client": ["android"]}},
        }
    ) as ydl:
        info = ydl.extract_info(url, download=False)

    return {
        "id": info.get("id"),
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "uploader_id": info.get("uploader_id"),
        "channel": info.get("channel"),
        "channel_id": info.get("channel_id"),
        "duration_s": info.get("duration"),
        "timestamp": info.get("timestamp"),
        "upload_date": info.get("upload_date"),
        "webpage_url": info.get("webpage_url") or url,
        "extractor": info.get("extractor"),
        "format_id": info.get("format_id"),
        "ext": info.get("ext"),
        "fps": info.get("fps"),
        "width": info.get("width"),
        "height": info.get("height"),
        "audio_codec": info.get("acodec"),
        "video_codec": info.get("vcodec"),
    }


def source_receipt(seed: YouTubeSeed, metadata: dict[str, Any] | None) -> dict[str, Any]:
    if canonical_video_id(seed.url) != seed.video_id:
        raise ValueError(f"seed URL/id mismatch for {seed.url}")

    payload = {
        "schema": "animalexic-youtube-birdsong-source-v1",
        "source": asdict(seed),
        "resolved_metadata": metadata,
        "decision": "candidate",
        "source_identity": f"youtube:{seed.video_id}",
        "acquisition": {
            "runtime_owner": "scripts/run_stereo_dispatch.py:FrameStreamer",
            "youtube_resolution": "yt_dlp.extract_info(download=False)",
            "media_decode": "ffmpeg transient stream; no full-download requirement in FrameStreamer",
            "source_pts_time_retained": True,
        },
        "usable_modalities_if_present": [
            "audio waveform / spectral features",
            "visible body/head/beak/movement features",
            "source-relative audio/video time",
        ],
        "boundary": {
            "url_identity_equals_media_byte_identity": False,
            "external_label_equals_verified_species_identity": False,
            "video_only_reveals_heart_rate": False,
            "video_only_reveals_metabolic_power": False,
            "audio_amplitude_equals_calibrated_SPL": False,
            "audio_energy_equals_metabolic_energy": False,
            "visible_motion_equals_biomechanical_work": False,
            "temporal_association_equals_causality": False,
            "youtube_publication_equals_animalexic_promotion": False,
            "candidate_observations_may_be_used_for_diagnostic_analysis": True,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def seed_by_id(video_id: str) -> YouTubeSeed:
    for seed in SEEDS:
        if seed.video_id == video_id:
            return seed
    raise KeyError(video_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-id", choices=[seed.video_id for seed in SEEDS])
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--resolve", action="store_true", help="resolve current YouTube metadata with yt-dlp; does not download media")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.all_seeds == bool(args.seed_id):
        raise SystemExit("choose exactly one of --seed-id or --all-seeds")

    seeds = list(SEEDS if args.all_seeds else (seed_by_id(args.seed_id),))
    receipts = []
    for seed in seeds:
        metadata = resolve_metadata(seed.url) if args.resolve else None
        receipts.append(source_receipt(seed, metadata))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipts, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
