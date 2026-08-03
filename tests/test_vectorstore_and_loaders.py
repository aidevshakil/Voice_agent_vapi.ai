"""Vector store contract and document loaders."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.core.exceptions import IngestionError, UnsupportedDocumentError
from app.rag.loaders import load_document
from app.rag.vectorstores.memory_store import MemoryVectorStore


@pytest.fixture
def populated_store(tmp_path: Path) -> MemoryVectorStore:
    store = MemoryVectorStore(collection="t", path=tmp_path)
    store.upsert(
        [
            Document(page_content="alpha", metadata={"source": "a.md"}),
            Document(page_content="beta", metadata={"source": "b.md"}),
        ],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ["id-a", "id-b"],
    )
    return store


# ------------------------------------------------------------------ the store
def test_search_ranks_by_similarity(populated_store: MemoryVectorStore):
    results = populated_store.search([1.0, 0.0, 0.0], k=2)
    assert results[0].text == "alpha"
    assert results[0].score > results[1].score
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_upsert_replaces_rather_than_duplicates(populated_store: MemoryVectorStore):
    populated_store.upsert(
        [Document(page_content="alpha revised", metadata={"source": "a.md"})],
        [[1.0, 0.0, 0.0]],
        ["id-a"],
    )
    assert populated_store.count() == 2
    assert populated_store.search([1.0, 0.0, 0.0], k=1)[0].text == "alpha revised"


def test_metadata_filter(populated_store: MemoryVectorStore):
    results = populated_store.search([1.0, 1.0, 0.0], k=2, where={"source": "b.md"})
    assert [r.source for r in results] == ["b.md"]


def test_delete_by_id_and_by_filter(populated_store: MemoryVectorStore):
    assert populated_store.delete(ids=["id-a"]) == 1
    assert populated_store.count() == 1
    assert populated_store.delete(where={"source": "b.md"}) == 1
    assert populated_store.count() == 0


def test_search_on_empty_store_is_safe(tmp_path: Path):
    assert MemoryVectorStore("empty", tmp_path).search([1.0, 0.0], k=3) == []


def test_dimension_change_is_rejected(populated_store: MemoryVectorStore):
    with pytest.raises(ValueError, match="dimension changed"):
        populated_store.upsert(
            [Document(page_content="x", metadata={})], [[1.0, 0.0]], ["id-x"]
        )


def test_state_survives_a_restart(tmp_path: Path):
    store = MemoryVectorStore("persist", tmp_path)
    store.upsert([Document(page_content="kept", metadata={"source": "s"})], [[1.0, 0.0]], ["k"])

    reopened = MemoryVectorStore("persist", tmp_path)
    assert reopened.count() == 1
    assert reopened.search([1.0, 0.0], k=1)[0].text == "kept"


def test_reset_clears_disk_and_memory(tmp_path: Path):
    store = MemoryVectorStore("r", tmp_path)
    store.upsert([Document(page_content="x", metadata={})], [[1.0]], ["x"])
    store.reset()

    assert store.count() == 0
    assert MemoryVectorStore("r", tmp_path).count() == 0


def test_stats_lists_sources(populated_store: MemoryVectorStore):
    stats = populated_store.stats()
    assert stats.vectors == 2
    assert stats.dimensions == 3
    assert stats.sources == ["a.md", "b.md"]


# ---------------------------------------------------------------- the loaders
def test_load_markdown(tmp_path: Path):
    path = tmp_path / "doc.md"
    path.write_text("# Title\n\nSome body text here.", encoding="utf-8")

    documents = load_document(path)
    assert len(documents) == 1
    assert "body text" in documents[0].page_content
    assert documents[0].metadata["source"] == "doc.md"
    assert documents[0].metadata["file_type"] == "md"


def test_load_csv_yields_one_document_per_row(tmp_path: Path):
    path = tmp_path / "rows.csv"
    path.write_text("name,role\nAda,engineer\nGrace,admiral\n", encoding="utf-8")

    documents = load_document(path)
    assert len(documents) == 2
    assert "Ada" in documents[0].page_content


def test_load_json_object_and_array(tmp_path: Path):
    array = tmp_path / "list.json"
    array.write_text('[{"q": "a"}, {"q": "b"}]', encoding="utf-8")
    assert len(load_document(array)) == 2

    obj = tmp_path / "one.json"
    obj.write_text('{"q": "only"}', encoding="utf-8")
    assert len(load_document(obj)) == 1


def test_load_html_drops_scripts(tmp_path: Path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>T</title><script>evil()</script></head>"
        "<body><p>Visible copy.</p></body></html>",
        encoding="utf-8",
    )
    content = load_document(path)[0].page_content
    assert "Visible copy." in content
    assert "evil" not in content


def test_unsupported_extension_raises(tmp_path: Path):
    path = tmp_path / "thing.xyz"
    path.write_text("data", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentError):
        load_document(path)


def test_invalid_json_raises_ingestion_error(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(IngestionError):
        load_document(path)


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(IngestionError):
        load_document(tmp_path / "nope.md")


def test_non_utf8_file_still_loads(tmp_path: Path):
    path = tmp_path / "latin.txt"
    path.write_bytes("Café façade naïve".encode("cp1252"))
    assert load_document(path)[0].page_content
