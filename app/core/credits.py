"""Builds a human-readable attribution/credits file for any Wikimedia
Commons media used in a render.

Why this exists: Pexels and Pixabay's free-tier license terms don't
require attribution (see their respective license pages), so the app has
never needed a credits mechanism before. Wikimedia Commons is different
-- essentially all of its media is published under a Creative Commons
license (typically CC BY-SA 4.0 or CC BY 4.0, confirmed via the real
`LicenseShortName`/`LicenseUrl` extmetadata fields returned by
`wikimedia_fetcher.search_wikimedia`) that legally *requires* crediting
the original author and linking back to the license, wherever the work
is used/redistributed. A finished MP4 that used Wikimedia footage without
surfacing that attribution somewhere would put the end user (who owns/
publishes the video) out of compliance with those licenses -- through no
fault of their own, since they have no way to know which pixels in the
video came from where.

This module turns `PipelineResult.wikimedia_assets_used` (populated by
`pipeline._make_shot_media_resolver`, deduplicated by URL) into:
  - a plain-text credits block (`build_credits_text`), suitable for
    pasting into a video description on YouTube/social platforms where
    the video eventually gets uploaded, which is exactly where this kind
    of attribution is conventionally expected to live; and
  - a small credits *file* written next to the output video
    (`write_credits_file`), so the requirement doesn't rely on the user
    remembering something they were told in a progress log that has
    since scrolled away.

If no Wikimedia assets were used in a render (the common case for tracks
without a specific named deity detected), both functions are a no-op --
this module adds nothing to the output for a purely Pexels/Pixabay/
procedural render, matching prior behavior exactly.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .wikimedia_fetcher import WikimediaAsset

log = logging.getLogger(__name__)


def build_credits_text(wikimedia_assets_used: list[WikimediaAsset]) -> str:
    """Return a plain-text attribution block for `wikimedia_assets_used`,
    or an empty string if the list is empty (nothing to credit).

    One entry per distinct asset (already deduplicated by the caller --
    see `pipeline._make_shot_media_resolver`'s `wikimedia_assets_used`
    tracking), in first-use order, each with title/author/license/source
    link -- the standard fields most CC license attribution guidance
    (including Wikimedia Commons' own "Reusing content outside Wikimedia"
    guide) asks for: title, author, license (with a link to its terms),
    and a link back to the original source page.
    """
    if not wikimedia_assets_used:
        return ""

    lines = [
        "Media credits (Wikimedia Commons)",
        "=" * 34,
        "",
        "This video includes media from Wikimedia Commons, used under the",
        "Creative Commons licenses below. Please keep this credits list with",
        "the video (e.g. in its description) to comply with those licenses.",
        "",
    ]
    for i, asset in enumerate(wikimedia_assets_used, start=1):
        lines.append(f"{i}. \"{asset.title}\" by {asset.author}")
        license_part = asset.license_name
        if asset.license_url:
            license_part += f" ({asset.license_url})"
        lines.append(f"   License: {license_part}")
        if asset.source_url:
            lines.append(f"   Source: {asset.source_url}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_credits_file(wikimedia_assets_used: list[WikimediaAsset], output_video_path: str) -> str | None:
    """Write a `<video_name>_credits.txt` file next to the rendered video,
    if there's anything to credit. Returns the written path, or None if
    `wikimedia_assets_used` was empty (no file is created in that case --
    a render with no Wikimedia media produces no extra output, same as
    before this feature existed).

    Placed next to the video (not just logged) so the requirement
    survives independently of the GUI/log output the user may not have
    kept -- the two travel together on disk and are easy to find.
    """
    text = build_credits_text(wikimedia_assets_used)
    if not text:
        return None

    video_path = Path(output_video_path)
    credits_path = video_path.with_name(f"{video_path.stem}_credits.txt")
    try:
        credits_path.write_text(text, encoding="utf-8")
        log.info("Wrote Wikimedia attribution credits to %s", credits_path)
        return str(credits_path)
    except Exception:
        log.warning("Failed to write credits file to %s", credits_path, exc_info=True)
        return None
