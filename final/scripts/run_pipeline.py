"""Convenience script for running the complete pipeline from the repository root."""

from pathlib import Path

from yipit_pipeline.pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline(
        Path("tech_news.csv"),
        Path("company_metadata.json"),
        Path("data/output"),
        embedding_backend="sentence-transformers",
        show_progress=True,
    )

