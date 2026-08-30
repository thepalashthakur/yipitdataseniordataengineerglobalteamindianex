"""Public modeling helpers.

The end-to-end materialization lives in :mod:`yipit_pipeline.pipeline`; this
module exposes the stable modeling entry points expected by consumers.
"""

from pathlib import Path
from typing import Any, Dict

from .pipeline import run_pipeline, validate_outputs


def materialize_models(articles_path: Path, companies_path: Path, output_dir: Path, **options: Any) -> Dict[str, Any]:
    return run_pipeline(articles_path, companies_path, output_dir, **options)


__all__ = ["materialize_models", "run_pipeline", "validate_outputs"]

