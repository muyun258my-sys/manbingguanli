from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PDF_DIR,
    DEFAULT_VECTOR_DB_DIR,
    build_vector_store,
    retrieve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vectorize PDF files into a local Chroma database.")
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--vector-db-dir", type=Path, default=DEFAULT_VECTOR_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--test-query", default=None, help="Run a sample retrieval after vectorization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_vector_store(
        pdf_dir=args.pdf_dir,
        vector_db_dir=args.vector_db_dir,
        collection_name=args.collection,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        reset=args.reset,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    if args.test_query:
        hits = retrieve(
            args.test_query,
            vector_db_dir=args.vector_db_dir,
            collection_name=args.collection,
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            top_k=5,
        )
        print("\nTop retrieval results:")
        for index, hit in enumerate(hits, start=1):
            metadata = hit["metadata"]
            text = str(hit["text"]).replace("\n", " ")
            print(f"{index}. {metadata} distance={hit['distance']}")
            print(text[:300])


if __name__ == "__main__":
    main()
