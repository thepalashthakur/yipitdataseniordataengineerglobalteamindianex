import json
from pathlib import Path

import duckdb
import pandas as pd

from yipit_pipeline.pipeline import run_pipeline, validate_outputs
from yipit_pipeline.search import ArticleSearchIndex


ROOT = Path(__file__).resolve().parents[1]


def test_full_pipeline_is_queryable_and_idempotent(tmp_path):
    output = tmp_path / "output"
    first = run_pipeline(
        ROOT / "tech_news.csv",
        ROOT / "company_metadata.json",
        output,
        embedding_backend="tfidf",
    )
    counts = validate_outputs(output)
    assert counts["fact_article"] == 750
    assert counts["fact_arr_observation"] > 0
    assert counts["bridge_article_similarity"] == 2_250
    assert counts["ai_articles_enriched"] > 0

    articles = pd.read_csv(output / "fact_article.csv")
    arr = pd.read_csv(output / "fact_arr_observation.csv")
    ai = pd.read_csv(output / "ai_articles_enriched.csv")
    required_ai_columns = [
        "article_id", "title", "company_name", "published_date", "category", "arr_usd",
        "summary", "url", "industry", "founded_year", "headquarters", "employee_count",
        "is_public", "stock_ticker", "company_age", "company_size_category", "embedding",
    ]
    assert ai.columns.tolist() == required_ai_columns
    assert set(arr["article_id"]) == set(articles.loc[articles["arr_parse_status"].eq("PARSED"), "article_id"])
    unresolved_arr = arr["company_match_status"].isin(["UNRESOLVED", "AMBIGUOUS"])
    assert unresolved_arr.any()
    assert articles["top_similar_articles"].map(lambda value: len(json.loads(value)) == 3).all()

    connection = duckdb.connect(str(output / "analytics.duckdb"), read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM fact_article").fetchone()[0] == 750
        assert connection.execute("SELECT COUNT(*) FROM latest_company_arr").fetchone()[0] > 0
        assert connection.execute("SELECT COUNT(*) FROM company_arr_by_quarter").fetchone()[0] > 0
    finally:
        connection.close()

    index = ArticleSearchIndex(output)
    results = index.hybrid_search(
        "artificial intelligence model growth",
        top_k=5,
        start_date="2022-01-01",
        end_date="2024-12-31",
        min_arr_usd=50_000_000,
    )
    assert results
    assert all("article_id" in result and "similarity_score" in result for result in results)

    observation_ids_before = set(arr["arr_observation_id"])
    second = run_pipeline(
        ROOT / "tech_news.csv",
        ROOT / "company_metadata.json",
        output,
        embedding_backend="tfidf",
    )
    arr_after = pd.read_csv(output / "fact_arr_observation.csv")
    assert set(arr_after["arr_observation_id"]) == observation_ids_before
    assert not arr_after["arr_observation_id"].duplicated().any()
    assert first["inputs"] == second["inputs"]

