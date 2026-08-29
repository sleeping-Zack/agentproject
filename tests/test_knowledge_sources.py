from pathlib import Path

import pytest

from rag.rag_utils import load_knowledge_source_metadata
from scripts.validate_knowledge_sources import validate_manifest
from utils.file_handler import listdir_with_allowed_type


def test_knowledge_files_are_discovered_recursively(tmp_path):
    nested = tmp_path / "vendor_manuals" / "roborock"
    nested.mkdir(parents=True)
    top_level = tmp_path / "guide.txt"
    manual = nested / "manual.PDF"
    ignored = nested / "notes.csv"
    top_level.write_text("guide", encoding="utf-8")
    manual.write_bytes(b"%PDF-1.4\n")
    ignored.write_text("ignored", encoding="utf-8")

    discovered = listdir_with_allowed_type(str(tmp_path), ("txt", "pdf"))

    assert discovered == tuple(sorted((str(top_level), str(manual))))


def test_official_source_metadata_is_attached_by_local_path(tmp_path):
    data_root = tmp_path / "data"
    manual = data_root / "vendor_manuals" / "irobot" / "j7-zh-cn.pdf"
    manual.parent.mkdir(parents=True)
    manual.write_bytes(b"%PDF-1.4\n")
    manifest = data_root / "knowledge_sources.yml"
    manifest.write_text(
        """
version: v1
sources:
  - id: irobot-j7-zh-cn
    vendor: iRobot
    models: [Roomba j7, Roomba j7+]
    document_type: user_manual
    language: zh-CN
    region: CN
    official_page_url: https://homesupport.irobot.com/articles/en_US/Knowledge/843
    download_url: https://prod-help-content.care.irobotapi.com/files/j_series/og/robot_only_CHINA.pdf
    local_path: vendor_manuals/irobot/j7-zh-cn.pdf
    redistribute: false
""".strip(),
        encoding="utf-8",
    )

    metadata = load_knowledge_source_metadata(str(manifest), str(data_root))
    item = metadata[str(manual.resolve()).casefold()]

    assert item["source_id"] == "irobot-j7-zh-cn"
    assert item["vendor"] == "iRobot"
    assert item["models"] == "Roomba j7|Roomba j7+"
    assert item["source_url"].startswith("https://homesupport.irobot.com/")
    assert item["redistribute"] is False


def test_source_manifest_rejects_paths_outside_data_root(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    manifest = data_root / "knowledge_sources.yml"
    manifest.write_text(
        """
version: v1
sources:
  - id: invalid
    vendor: Example
    local_path: ../outside.pdf
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside data root"):
        load_knowledge_source_metadata(str(manifest), str(data_root))


def test_repository_official_source_catalog_is_valid():
    summary = validate_manifest(Path("data/knowledge_sources.yml"))

    assert summary["sources"] >= 15
    assert summary["local_manual_paths"] >= 4
