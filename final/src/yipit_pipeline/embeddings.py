"""Embedding generation and article-to-article similarity utilities."""

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np

from .config import DEFAULT_EMBEDDING_MODEL


def build_article_text(title: object, summary: object) -> str:
    return "{}\n\n{}".format(str(title or "").strip(), str(summary or "").strip()).strip()


def generate_embeddings(
    texts: Sequence[str],
    artifact_dir: Path,
    backend: str = "sentence-transformers",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    show_progress: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Generate normalized embeddings and persist backend-specific artifacts."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    if backend == "sentence-transformers":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        matrix = model.encode(
            list(texts),
            batch_size=64,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        manifest = {
            "backend": backend,
            "model_name": model_name,
            "dimension": int(matrix.shape[1]),
            "normalized": True,
        }
    elif backend == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=384,
            norm="l2",
        )
        matrix = vectorizer.fit_transform(list(texts)).toarray().astype(np.float32)
        joblib.dump(vectorizer, artifact_dir / "tfidf_vectorizer.joblib")
        manifest = {
            "backend": backend,
            "model_name": "sklearn-tfidf-word-bigram",
            "dimension": int(matrix.shape[1]),
            "normalized": True,
        }
    else:
        raise ValueError("unsupported embedding backend: {}".format(backend))

    if matrix.ndim != 2 or matrix.shape[0] != len(texts):
        raise ValueError("embedding output has an unexpected shape")
    if not np.isfinite(matrix).all():
        raise ValueError("embedding output contains non-finite values")

    np.save(artifact_dir / "article_embeddings.npy", matrix)
    (artifact_dir / "embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return matrix, manifest


def embedding_json_rows(matrix: np.ndarray) -> List[str]:
    return [json.dumps([round(float(value), 8) for value in row], separators=(",", ":")) for row in matrix]


def top_similar_articles(
    article_ids: Sequence[str], matrix: np.ndarray, top_n: int = 3
) -> Tuple[List[str], List[Dict[str, object]]]:
    """Return per-article JSON IDs and a normalized similarity bridge."""

    if len(article_ids) != matrix.shape[0]:
        raise ValueError("article ID and embedding counts differ")
    scores = np.matmul(matrix, matrix.T)
    np.fill_diagonal(scores, -np.inf)
    json_rows: List[str] = []
    bridge_rows: List[Dict[str, object]] = []
    limit = min(top_n, max(0, len(article_ids) - 1))
    for source_index, article_id in enumerate(article_ids):
        order = np.argsort(-scores[source_index], kind="mergesort")[:limit]
        similar_ids = [str(article_ids[index]) for index in order]
        json_rows.append(json.dumps(similar_ids, separators=(",", ":")))
        for rank, target_index in enumerate(order, start=1):
            bridge_rows.append(
                {
                    "source_article_id": str(article_id),
                    "similar_article_id": str(article_ids[target_index]),
                    "similarity_rank": rank,
                    "similarity_score": round(float(scores[source_index, target_index]), 8),
                }
            )
    return json_rows, bridge_rows

