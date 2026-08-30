"""Command-line entry point for pipeline, validation, and search."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .config import DEFAULT_EMBEDDING_MODEL
from .pipeline import run_pipeline, validate_outputs
from .search import ArticleSearchIndex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YipitData technology-news pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run and publish the complete pipeline")
    run_parser.add_argument("--articles", type=Path, default=Path("tech_news.csv"))
    run_parser.add_argument("--companies", type=Path, default=Path("company_metadata.json"))
    run_parser.add_argument("--output-dir", type=Path, default=Path("data/output"))
    run_parser.add_argument("--embedding-backend", choices=["sentence-transformers", "tfidf"], default="sentence-transformers")
    run_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    run_parser.add_argument("--show-progress", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="validate generated outputs")
    validate_parser.add_argument("--output-dir", type=Path, default=Path("data/output"))

    search_parser = subparsers.add_parser("search", help="perform semantic or hybrid search")
    search_parser.add_argument("--output-dir", type=Path, default=Path("data/output"))
    query_group = search_parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query")
    query_group.add_argument("--article-id", help="rank against an existing article instead of query text")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--start-date")
    search_parser.add_argument("--end-date")
    search_parser.add_argument("--category", action="append", dest="categories")
    search_parser.add_argument("--industry", action="append", dest="industries")
    search_parser.add_argument("--min-arr-usd", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        result = run_pipeline(
            args.articles,
            args.companies,
            args.output_dir,
            embedding_backend=args.embedding_backend,
            embedding_model=args.embedding_model,
            show_progress=args.show_progress,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "validate":
        print(json.dumps(validate_outputs(args.output_dir), indent=2, sort_keys=True))
        return 0
    if args.command == "search":
        index = ArticleSearchIndex(args.output_dir)
        if args.article_id:
            results = index.find_articles_similar_to_article(args.article_id, args.top_k)
        else:
            results = index.hybrid_search(
                args.query,
                top_k=args.top_k,
                start_date=args.start_date,
                end_date=args.end_date,
                categories=args.categories,
                industries=args.industries,
                min_arr_usd=args.min_arr_usd,
            )
        print(json.dumps(results, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
