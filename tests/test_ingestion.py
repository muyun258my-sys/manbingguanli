from app.ingestion import DEFAULT_PDF_DIR, PageText, chunk_pages, infer_category, split_text


def test_default_pdf_dir_points_to_project_knowledge_base():
    assert DEFAULT_PDF_DIR == Path("shujuku")
from pathlib import Path


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


def test_infer_category_from_parent_folder():
    root = Path("raw_pdfs")
    path = root / "guidelines" / "gaoxueya.pdf"
    assert infer_category(path, root) == "guidelines"


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
