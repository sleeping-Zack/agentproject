"""Validate the tracked official-source catalog without crawling vendor sites."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

import yaml


VENDOR_HOST_SUFFIXES = {
    "roborock": ("roborock.com",),
    "irobot": ("irobot.com", "irobotapi.com"),
    "ecovacs": ("ecovacs.com", "ecovacs.cn"),
    "dreame": ("dreametech.com", "dreame.tech", "shopify.com"),
}


def _is_allowed_host(vendor: str, url: str) -> bool:
    host = (urlparse(url).hostname or "").casefold()
    suffixes = VENDOR_HOST_SUFFIXES.get(vendor.casefold(), ())
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in suffixes)


def validate_manifest(manifest_path: Path) -> dict[str, int]:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    sources = payload.get("sources") or []
    if not isinstance(sources, list) or not sources:
        raise ValueError("manifest must contain a non-empty sources list")

    seen_ids: set[str] = set()
    local_paths: set[str] = set()
    for source in sources:
        source_id = str(source.get("id") or "").strip()
        vendor = str(source.get("vendor") or "").strip()
        source_type = str(source.get("document_type") or "").strip()
        official_url = str(source.get("official_page_url") or "").strip()
        if not source_id or not vendor or not source_type or not official_url:
            raise ValueError("every source needs id, vendor, document_type and official_page_url")
        if source_id in seen_ids:
            raise ValueError(f"duplicate source id: {source_id}")
        seen_ids.add(source_id)

        for field in ("official_page_url", "download_url"):
            url = str(source.get(field) or "").strip()
            if not url:
                continue
            if urlparse(url).scheme != "https":
                raise ValueError(f"{source_id}: {field} must use https")
            if not _is_allowed_host(vendor, url):
                raise ValueError(f"{source_id}: {field} is not on an approved vendor host")

        local_path = str(source.get("local_path") or "").strip()
        if local_path:
            candidate = Path(local_path)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"{source_id}: local_path must stay below data/")
            normalized = candidate.as_posix().casefold()
            if normalized in local_paths:
                raise ValueError(f"duplicate local_path: {local_path}")
            local_paths.add(normalized)
            if source.get("redistribute") is not False:
                raise ValueError(f"{source_id}: cached vendor manuals must not be redistributed")

        if not source.get("rights_status") or not source.get("usage_scope"):
            raise ValueError(f"{source_id}: rights_status and usage_scope are required")
        if not source.get("verified_at"):
            raise ValueError(f"{source_id}: verified_at is required")

    return {"sources": len(sources), "local_manual_paths": len(local_paths)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path("data/knowledge_sources.yml"),
    )
    args = parser.parse_args()
    summary = validate_manifest(args.manifest)
    print(
        f"validated {summary['sources']} official sources; "
        f"{summary['local_manual_paths']} optional local manual paths"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
