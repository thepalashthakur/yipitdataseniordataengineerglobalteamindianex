"""Reusable vector and hybrid-search interface over generated artifacts."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import duckdb
import numpy as np
import pandas as pd

from .embeddings import build_article_text


class ArticleSearchIndex:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.articles = pd.read_csv(self.output_dir / "fact_article.csv")
        self.matrix = np.load(self.output_dir / "article_embeddings.npy")
        self.article_index = pd.read_csv(self.output_dir / "article_embedding_index.csv")
        self.manifest = json.loads((self.output_dir / "embedding_manifest.json").read_text(encoding="utf-8"))
        if len(self.articles) != self.matrix.shape[0]:
            raise ValueError("article table and embedding matrix are not aligned")
        if self.article_index["article_id"].astype(str).tolist() != self.articles["article_id"].astype(str).tolist():
            raise ValueError("article embedding index and fact_article are not aligned")
        self._embedding_row_by_article_id = {
            str(row.article_id): int(row.embedding_row)
            for row in self.article_index.itertuples(index=False)
        }
        self._encoder = None

    def _encode_query(self, query_text: str) -> np.ndarray:
        if not str(query_text).strip():
            raise ValueError("query_text cannot be blank")
        backend = self.manifest["backend"]
        if backend == "sentence-transformers":
            if self._encoder is None:
                from sentence_transformers import SentenceTransformer

                # The completed pipeline run has already downloaded the model.
                # Avoid slow network metadata checks during local/offline search.
                self._encoder = SentenceTransformer(
                    self.manifest["model_name"], local_files_only=True
                )
            vector = self._encoder.encode(
                [str(query_text)], convert_to_numpy=True, normalize_embeddings=True
            )[0]
        elif backend == "tfidf":
            if self._encoder is None:
                self._encoder = joblib.load(self.output_dir / "tfidf_vectorizer.joblib")
            vector = self._encoder.transform([str(query_text)]).toarray()[0]
        else:
            raise ValueError("unsupported embedding backend: {}".format(backend))
        return np.asarray(vector, dtype=np.float32)

    def _rank(self, query_vector: np.ndarray, candidate_indices: Sequence[int], top_k: int) -> List[Dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        candidates = np.asarray(list(candidate_indices), dtype=int)
        if not len(candidates):
            return []
        scores = np.matmul(self.matrix[candidates], query_vector)
        order = np.argsort(-scores, kind="mergesort")[: min(top_k, len(candidates))]
        results: List[Dict[str, Any]] = []
        for position in order:
            row_index = int(candidates[position])
            row = self.articles.iloc[row_index]
            results.append(
                {
                    "article_id": str(row["article_id"]),
                    "similarity_score": round(float(scores[position]), 8),
                    "title": row["title"],
                    "company_name": row["company_name"],
                    "published_date": None if pd.isna(row["published_date"]) else str(row["published_date"]),
                    "arr_usd": None if pd.isna(row["arr_usd"]) else int(row["arr_usd"]),
                }
            )
        return results

    def find_similar_articles(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self._encode_query(query_text)
        return self._rank(query_vector, range(len(self.articles)), top_k)

    def find_articles_similar_to_article(self, article_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        matches = self.articles.index[self.articles["article_id"].astype(str).eq(str(article_id))].tolist()
        if not matches:
            raise KeyError("unknown article_id: {}".format(article_id))
        source_index = matches[0]
        candidates = [index for index in range(len(self.articles)) if index != source_index]
        return self._rank(self.matrix[source_index], candidates, top_k)

    def hybrid_search(
        self,
        query_text: str,
        top_k: int = 5,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        categories: Optional[Sequence[str]] = None,
        industries: Optional[Sequence[str]] = None,
        min_arr_usd: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        conditions = ["1 = 1"]
        parameters: List[Any] = []
        if start_date:
            conditions.append("published_date >= CAST(? AS DATE)")
            parameters.append(start_date)
        if end_date:
            conditions.append("published_date <= CAST(? AS DATE)")
            parameters.append(end_date)
        if categories:
            category_values = list(categories)
            conditions.append(
                "category_standardized IN ({})".format(
                    ", ".join("?" for _ in category_values)
                )
            )
            parameters.extend(category_values)
        if industries:
            industry_values = list(industries)
            conditions.append(
                "industry_standardized IN ({})".format(
                    ", ".join("?" for _ in industry_values)
                )
            )
            parameters.extend(industry_values)
        if min_arr_usd is not None:
            conditions.append("arr_usd > ?")
            parameters.append(int(min_arr_usd))

        sql = """
            SELECT article_id
            FROM fact_article
            WHERE {}
            ORDER BY source_row_number
        """.format(" AND ".join(conditions))
        database_path = self.output_dir / "analytics.duckdb"
        with duckdb.connect(str(database_path), read_only=True) as connection:
            candidate_ids = [
                str(row[0]) for row in connection.execute(sql, parameters).fetchall()
            ]
        candidate_indices = [
            self._embedding_row_by_article_id[article_id]
            for article_id in candidate_ids
        ]
        query_vector = self._encode_query(query_text)
        return self._rank(query_vector, candidate_indices, top_k)


def find_similar_articles(query_text: str, top_k: int = 5, output_dir: Path = Path("data/output")) -> List[Dict[str, Any]]:
    return ArticleSearchIndex(output_dir).find_similar_articles(query_text, top_k)


def hybrid_search(query_text: str, top_k: int = 5, output_dir: Path = Path("data/output"), **filters: Any) -> List[Dict[str, Any]]:
    return ArticleSearchIndex(output_dir).hybrid_search(query_text, top_k=top_k, **filters)
