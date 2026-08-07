from pathlib import Path

from app.ingestion import (
    DEFAULT_PDF_DIR,
    PageText,
    batched,
    chunk_pages,
    clean_text,
    discover_pdfs,
    infer_category,
    split_text,
    stable_chunk_id,
)


def test_default_pdf_dir_points_to_project_knowledge_base():
    assert DEFAULT_PDF_DIR == Path("shujuku")


# ── clean_text ───────────────────────────────────────────────────────────────

def test_clean_text_removes_nulls_and_collapses_whitespace():
    raw = "  \x00 高血压\x00指南\n\n\n\n第二节\n\t\t饮食建议  "
    cleaned = clean_text(raw)
    assert "\x00" not in cleaned
    assert "\n\n\n\n" not in cleaned
    assert cleaned == cleaned.strip()


# ── split_text ───────────────────────────────────────────────────────────────

def test_split_text_respects_overlap():
    text = "abcde" * 100
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert chunks[0][-20:] == chunks[1][:20]


def test_split_text_rejects_bad_overlap():
    try:
        split_text("hello", chunk_size=10, overlap=10)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_split_text_rejects_negative_overlap():
    try:
        split_text("hello", chunk_size=10, overlap=-1)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_split_text_rejects_non_positive_chunk_size():
    try:
        split_text("hello", chunk_size=0)
    except ValueError as exc:
        assert "chunk_size" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_split_text_empty_input_returns_no_chunks():
    assert split_text("   \n\n  ") == []
    assert split_text("") == []


def test_split_text_keeps_single_chunk_when_short():
    assert split_text("简短文本", chunk_size=900) == ["简短文本"]


# ── infer_category ───────────────────────────────────────────────────────────

def test_infer_category_from_parent_folder():
    root = Path("raw_pdfs")
    path = root / "guidelines" / "gaoxueya.pdf"
    assert infer_category(path, root) == "guidelines"


def test_infer_category_uncategorized_for_root_files():
    root = Path("raw_pdfs")
    assert infer_category(root / "notes.pdf", root) == "uncategorized"


def test_infer_category_uncategorized_for_foreign_path():
    assert infer_category(Path("elsewhere/a.pdf"), Path("raw_pdfs")) == "uncategorized"


# ── discover_pdfs ────────────────────────────────────────────────────────────

def test_discover_pdfs_finds_only_pdfs_recursively(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.pdf").write_bytes(b"x")
    (tmp_path / "b.txt").write_bytes(b"x")
    (tmp_path / "sub" / "c.pdf").write_bytes(b"x")

    found = discover_pdfs(tmp_path)
    assert [p.name for p in found] == ["a.pdf", "c.pdf"]


def test_discover_pdfs_empty_dir(tmp_path):
    assert discover_pdfs(tmp_path) == []


# ── stable_chunk_id ──────────────────────────────────────────────────────────

def test_stable_chunk_id_is_deterministic():
    assert stable_chunk_id("a/b.pdf", 3, 0) == stable_chunk_id("a/b.pdf", 3, 0)
    assert stable_chunk_id("a/b.pdf", 3, 0) != stable_chunk_id("a/b.pdf", 3, 1)


# ── chunk_pages ──────────────────────────────────────────────────────────────

def test_chunk_pages_preserves_metadata():
    pages = [
        PageText(
            source_path="raw_pdfs/guidelines/example.pdf",
            pdf_name="example.pdf",
            category="guidelines",
            page=3,
            text="a" * 120,
        )
    ]
    chunks = chunk_pages(pages, chunk_size=50, overlap=10)
    assert chunks
    assert chunks[0].source_path == "raw_pdfs/guidelines/example.pdf"
    assert chunks[0].pdf_name == "example.pdf"
    assert chunks[0].category == "guidelines"
    assert chunks[0].page == 3
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata()["page"] == 3


def test_chunk_pages_multiple_pages_reset_chunk_index():
    pages = [
        PageText("src.pdf", "src.pdf", "g", 1, "b" * 2000),
        PageText("src.pdf", "src.pdf", "g", 2, "c" * 2000),
    ]
    chunks = chunk_pages(pages, chunk_size=500, overlap=50)
    assert chunks[0].page == 1
    assert chunks[0].chunk_index == 0
    page_two_chunks = [c for c in chunks if c.page == 2]
    assert page_two_chunks[0].chunk_index == 0


# ── batched ──────────────────────────────────────────────────────────────────

def test_batched_yields_batches_of_given_size():
    items = [PageText("s", "n", "c", 1, "x")] * 10
    batches = list(batched(items, batch_size=4))
    assert [len(b) for b in batches] == [4, 4, 2]
