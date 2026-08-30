"""End-to-end local pipeline orchestration."""

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from uuid import uuid4, uuid5

import duckdb
import numpy as np
import pandas as pd

from .cleaning import (
    calculate_company_age,
    company_size_category,
    parse_arr,
    parse_published_date,
    standardize_category,
    standardize_industry,
)
from .config import (
    ARR_NAMESPACE,
    ARR_PARSER_VERSION,
    CATEGORY_MAPPING_VERSION,
    COMPANY_ALIAS_VERSION,
    DEFAULT_EMBEDDING_MODEL,
    PIPELINE_VERSION,
)
from .embeddings import build_article_text, embedding_json_rows, generate_embeddings, top_similar_articles
from .matching import company_id_for, resolve_company
from .quality import quality_issue


ARTICLE_COLUMNS = [
    "article_id",
    "title",
    "company_name",
    "published_date",
    "category",
    "revenue",
    "summary",
    "url",
    "author",
    "word_count",
]

METADATA_FIELDS = [
    "founded_year",
    "headquarters",
    "employee_count",
    "industry",
    "is_public",
    "stock_ticker",
]

MODELED_TABLES = {
    "dim_company": "dim_company.csv",
    "fact_article": "fact_article.csv",
    "fact_arr_observation": "fact_arr_observation.csv",
    "data_quality_issue": "data_quality_issue.csv",
    "bridge_article_similarity": "bridge_article_similarity.csv",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_record_hash(values: Iterable[Any]) -> str:
    payload = json.dumps([None if pd.isna(value) else str(value) for value in values], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_inputs(articles: pd.DataFrame, metadata: Mapping[str, Any]) -> None:
    missing_columns = sorted(set(ARTICLE_COLUMNS) - set(articles.columns))
    if missing_columns:
        raise ValueError("article input is missing columns: {}".format(", ".join(missing_columns)))
    if articles["article_id"].astype(str).str.strip().eq("").any():
        raise ValueError("article_id cannot be blank")
    if articles["article_id"].duplicated().any():
        duplicates = articles.loc[articles["article_id"].duplicated(), "article_id"].tolist()
        raise ValueError("article_id must be unique; duplicates: {}".format(duplicates[:5]))
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("company metadata must be a non-empty JSON object")
    for company, values in metadata.items():
        if not isinstance(values, dict):
            raise ValueError("metadata for {} must be an object".format(company))
        missing_fields = sorted(set(METADATA_FIELDS) - set(values))
        if missing_fields:
            raise ValueError("metadata for {} is missing fields: {}".format(company, ", ".join(missing_fields)))


def _build_company_dimension(
    metadata: Mapping[str, Mapping[str, Any]], resolutions: List[Dict[str, Any]], run_id: str
) -> pd.DataFrame:
    resolution_by_id = {item["company_id"]: item for item in resolutions}
    rows: List[Dict[str, Any]] = []
    for canonical_name, values in metadata.items():
        company_id = company_id_for(canonical_name)
        representative = resolution_by_id.get(company_id)
        row = {
            "company_id": company_id,
            "company_name": canonical_name,
            "company_name_raw": representative["company_name_raw"] if representative else canonical_name,
            "company_match_status": "MATCHED" if representative else "METADATA_ONLY",
            "company_match_method": representative["company_match_method"] if representative else "METADATA_KEY",
            "company_match_confidence": representative["company_match_confidence"] if representative else 1.0,
            "has_company_metadata": True,
            "industry_raw": values.get("industry"),
            "industry_standardized": standardize_industry(values.get("industry")),
            "founded_year": values.get("founded_year"),
            "headquarters": values.get("headquarters"),
            "employee_count": values.get("employee_count"),
            "company_size_category": company_size_category(values.get("employee_count")),
            "is_public": values.get("is_public"),
            "stock_ticker": values.get("stock_ticker"),
            "source_file": "company_metadata.json",
            "pipeline_run_id": run_id,
        }
        row["record_hash"] = _stable_record_hash(row.get(field) for field in METADATA_FIELDS)
        rows.append(row)

    known_ids = {row["company_id"] for row in rows}
    for resolution in resolutions:
        if resolution["company_id"] in known_ids:
            continue
        row = {
            "company_id": resolution["company_id"],
            "company_name": resolution["company_entity_name"],
            "company_name_raw": resolution["company_name_raw"],
            "company_match_status": resolution["company_match_status"],
            "company_match_method": resolution["company_match_method"],
            "company_match_confidence": resolution["company_match_confidence"],
            "has_company_metadata": False,
            "industry_raw": None,
            "industry_standardized": "UNMAPPED",
            "founded_year": None,
            "headquarters": None,
            "employee_count": None,
            "company_size_category": "UNKNOWN",
            "is_public": None,
            "stock_ticker": None,
            "source_file": "tech_news.csv",
            "pipeline_run_id": run_id,
        }
        row["record_hash"] = _stable_record_hash([resolution["company_entity_name"], resolution["company_match_status"]])
        rows.append(row)
        known_ids.add(resolution["company_id"])
    return pd.DataFrame(rows).sort_values(["company_name", "company_id"]).reset_index(drop=True)


def _build_quality_issues(
    fact_article: pd.DataFrame,
    dim_company: pd.DataFrame,
    run_id: str,
    created_at: str,
) -> pd.DataFrame:
    issues: List[Dict[str, Any]] = []
    for row in fact_article.to_dict("records"):
        article_id = str(row["article_id"])
        arr_status = row["arr_parse_status"]
        if arr_status != "PARSED":
            severity = "ERROR" if arr_status == "INVALID_FORMAT" else "INFO"
            issues.append(quality_issue(run_id, created_at, "fact_article", article_id, "revenue_raw", "ARR_{}".format(arr_status), severity, row.get("revenue_raw"), row.get("arr_parse_reason")))
        if row["date_parse_status"] != "PARSED":
            issues.append(quality_issue(run_id, created_at, "fact_article", article_id, "published_date_raw", "DATE_{}".format(row["date_parse_status"]), "ERROR", row.get("published_date_raw"), row.get("date_parse_reason")))
        elif row.get("date_parse_reason"):
            issues.append(quality_issue(run_id, created_at, "fact_article", article_id, "published_date_raw", "DATE_AMBIGUOUS_POLICY_APPLIED", "INFO", row.get("published_date_raw"), row.get("date_parse_reason")))
        if row["company_match_status"] in {"UNRESOLVED", "AMBIGUOUS"}:
            issues.append(quality_issue(run_id, created_at, "fact_article", article_id, "company_name_raw", "COMPANY_{}".format(row["company_match_status"]), "WARNING", row.get("company_name_raw"), "company metadata was not joined"))
        if row["category_standardized"] == "UNMAPPED":
            issues.append(quality_issue(run_id, created_at, "fact_article", article_id, "category_raw", "CATEGORY_UNMAPPED", "WARNING", row.get("category_raw"), "category is not in the configured taxonomy"))
        if row.get("published_year") is not None and row.get("founded_year") is not None and row.get("company_age") is None:
            issues.append(quality_issue(run_id, created_at, "fact_article", article_id, "founded_year", "COMPANY_AGE_NEGATIVE", "WARNING", row.get("founded_year"), "founded year is after publication year"))

    for company in dim_company.to_dict("records"):
        if not company["has_company_metadata"]:
            continue
        company_id = str(company["company_id"])
        is_public = bool(company["is_public"])
        has_ticker = bool(company.get("stock_ticker")) and not pd.isna(company.get("stock_ticker"))
        if is_public and not has_ticker:
            issues.append(quality_issue(run_id, created_at, "dim_company", company_id, "stock_ticker", "PUBLIC_COMPANY_WITHOUT_TICKER", "WARNING", company.get("stock_ticker"), "public company has no ticker in source metadata"))
        if not is_public and has_ticker:
            issues.append(quality_issue(run_id, created_at, "dim_company", company_id, "stock_ticker", "PRIVATE_COMPANY_WITH_TICKER", "WARNING", company.get("stock_ticker"), "private company has a ticker in source metadata"))
    columns = ["issue_id", "pipeline_run_id", "source_table", "source_record_id", "field_name", "rule_code", "severity", "raw_value", "message", "created_at"]
    return pd.DataFrame(issues, columns=columns).sort_values(["source_table", "source_record_id", "rule_code"]).reset_index(drop=True)


def _write_duckdb(path: Path, tables: Mapping[str, pd.DataFrame]) -> None:
    connection = duckdb.connect(str(path))
    try:
        for table_name, frame in tables.items():
            connection.register("source_frame", frame)
            connection.execute("CREATE OR REPLACE TABLE {} AS SELECT * FROM source_frame".format(table_name))
            connection.unregister("source_frame")
        connection.execute(
            """
            CREATE OR REPLACE VIEW latest_company_arr AS
            SELECT * EXCLUDE (row_num)
            FROM (
                SELECT a.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY company_id
                           ORDER BY observation_date DESC, article_id DESC
                       ) AS row_num
                FROM fact_arr_observation a
                WHERE observation_status = 'DATED'
            )
            WHERE row_num = 1
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE VIEW company_arr_by_quarter AS
            SELECT company_id,
                   observation_year,
                   observation_quarter,
                   COUNT(*) AS observation_count,
                   MIN(arr_usd) AS min_reported_arr_usd,
                   MAX(arr_usd) AS max_reported_arr_usd,
                   ARG_MAX(arr_usd, observation_date) AS latest_reported_arr_usd,
                   MAX(observation_date) AS latest_observation_date
            FROM fact_arr_observation
            WHERE observation_status = 'DATED'
            GROUP BY 1, 2, 3
            """
        )
        connection.execute("CHECKPOINT")
    finally:
        connection.close()


def validate_outputs(output_dir: Path) -> Dict[str, int]:
    output_dir = Path(output_dir)
    required = list(MODELED_TABLES.values()) + ["ai_articles_enriched.csv", "article_embeddings.npy", "article_embedding_index.csv", "analytics.duckdb", "pipeline_run.json"]
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise ValueError("missing required outputs: {}".format(", ".join(missing)))
    dim = pd.read_csv(output_dir / "dim_company.csv")
    articles = pd.read_csv(output_dir / "fact_article.csv")
    arr = pd.read_csv(output_dir / "fact_arr_observation.csv")
    bridge = pd.read_csv(output_dir / "bridge_article_similarity.csv")
    if dim["company_id"].duplicated().any():
        raise ValueError("dim_company company_id is not unique")
    if articles["article_id"].duplicated().any():
        raise ValueError("fact_article article_id is not unique")
    if arr["arr_observation_id"].duplicated().any():
        raise ValueError("fact_arr_observation arr_observation_id is not unique")
    if not set(arr["article_id"]).issubset(set(articles["article_id"])):
        raise ValueError("ARR fact contains an unknown article_id")
    if not set(arr["company_id"]).issubset(set(dim["company_id"])):
        raise ValueError("ARR fact contains an unknown company_id")
    if not (arr["arr_usd"] > 0).all():
        raise ValueError("ARR fact contains a non-positive value")
    expected_bridge_rows = len(articles) * min(3, max(0, len(articles) - 1))
    if len(bridge) != expected_bridge_rows:
        raise ValueError("similarity bridge row count is incorrect")
    embeddings = np.load(output_dir / "article_embeddings.npy")
    if embeddings.shape[0] != len(articles) or not np.isfinite(embeddings).all():
        raise ValueError("embedding matrix does not align with fact_article")
    return {
        "dim_company": len(dim),
        "fact_article": len(articles),
        "fact_arr_observation": len(arr),
        "data_quality_issue": len(pd.read_csv(output_dir / "data_quality_issue.csv")),
        "bridge_article_similarity": len(bridge),
        "ai_articles_enriched": len(pd.read_csv(output_dir / "ai_articles_enriched.csv")),
    }


def run_pipeline(
    articles_path: Path,
    companies_path: Path,
    output_dir: Path,
    embedding_backend: str = "sentence-transformers",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """Run all transformations and atomically publish required outputs."""

    articles_path = Path(articles_path)
    companies_path = Path(companies_path)
    output_dir = Path(output_dir)
    run_id = str(uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    raw_articles = pd.read_csv(articles_path, dtype=str, keep_default_na=False)
    metadata = json.loads(companies_path.read_text(encoding="utf-8"))
    _validate_inputs(raw_articles, metadata)

    fact = raw_articles[ARTICLE_COLUMNS].copy()
    fact.insert(0, "source_row_number", range(2, len(fact) + 2))
    fact["source_file"] = articles_path.name
    fact["source_record_hash"] = raw_articles[ARTICLE_COLUMNS].apply(lambda row: _stable_record_hash(row.tolist()), axis=1)
    fact = fact.rename(
        columns={
            "company_name": "company_name_raw",
            "published_date": "published_date_raw",
            "category": "category_raw",
            "revenue": "revenue_raw",
        }
    )

    arr_frame = pd.DataFrame([asdict(parse_arr(value)) for value in fact["revenue_raw"]]).drop(columns=["revenue_raw"])
    date_frame = pd.DataFrame([asdict(parse_published_date(value)) for value in fact["published_date_raw"]]).drop(columns=["published_date_raw"])
    resolutions = [asdict(resolve_company(value, metadata)) for value in fact["company_name_raw"]]
    resolution_frame = pd.DataFrame(resolutions).drop(columns=["company_name_raw"])
    fact = pd.concat([fact.reset_index(drop=True), arr_frame, date_frame, resolution_frame], axis=1)
    fact["category_standardized"] = fact["category_raw"].map(standardize_category)
    fact["word_count"] = pd.to_numeric(fact["word_count"], errors="coerce").astype("Int64")
    fact["arr_usd"] = pd.array(fact["arr_usd"], dtype="Int64")
    fact["published_year"] = pd.array(fact["published_year"], dtype="Int64")
    fact["published_quarter"] = pd.array(fact["published_quarter"], dtype="Int64")
    fact["published_month"] = pd.array(fact["published_month"], dtype="Int64")

    dim_company = _build_company_dimension(metadata, resolutions, run_id)
    enrichment_columns = [
        "company_id",
        "industry_raw",
        "industry_standardized",
        "founded_year",
        "headquarters",
        "employee_count",
        "company_size_category",
        "is_public",
        "stock_ticker",
    ]
    fact = fact.merge(dim_company[enrichment_columns], on="company_id", how="left", validate="many_to_one")
    fact["company_age"] = [calculate_company_age(founded, year) for founded, year in zip(fact["founded_year"], fact["published_year"])]
    fact["company_name"] = fact["company_name_canonical"].fillna(fact["company_name_raw"])
    fact["published_date"] = pd.to_datetime(fact["published_date"], errors="coerce").dt.date
    fact["pipeline_run_id"] = run_id

    staging_parent = output_dir.parent if output_dir.parent.exists() else Path(".")
    staging_dir = Path(tempfile.mkdtemp(prefix=".yipit-pipeline-", dir=str(staging_parent)))
    try:
        texts = [build_article_text(title, summary) for title, summary in zip(fact["title"], fact["summary"])]
        matrix, embedding_manifest = generate_embeddings(texts, staging_dir, embedding_backend, embedding_model, show_progress)
        embedding_rows = embedding_json_rows(matrix)
        fact["embedding"] = embedding_rows
        top_rows, bridge_records = top_similar_articles(fact["article_id"].astype(str).tolist(), matrix, top_n=3)
        fact["top_similar_articles"] = top_rows

        article_index = pd.DataFrame({"embedding_row": range(len(fact)), "article_id": fact["article_id"].astype(str)})
        article_index.to_csv(staging_dir / "article_embedding_index.csv", index=False)
        bridge = pd.DataFrame(bridge_records)
        bridge["embedding_model"] = embedding_manifest["model_name"]
        bridge["pipeline_run_id"] = run_id

        arr_fact = fact.loc[fact["arr_parse_status"].eq("PARSED"), [
            "article_id", "company_id", "published_date", "date_parse_status", "published_year",
            "published_quarter", "published_month", "arr_usd", "revenue_raw", "source_currency",
            "company_match_status", "source_file",
        ]].copy()
        arr_fact.insert(0, "arr_observation_id", [str(uuid5(ARR_NAMESPACE, "{}:reported_arr".format(article_id))) for article_id in arr_fact["article_id"]])
        arr_fact = arr_fact.rename(columns={
            "published_date": "observation_date",
            "published_year": "observation_year",
            "published_quarter": "observation_quarter",
            "published_month": "observation_month",
        })
        arr_fact.insert(4, "observation_status", np.where(arr_fact["observation_date"].notna(), "DATED", "UNDATED"))
        arr_fact["parse_method_version"] = ARR_PARSER_VERSION
        arr_fact["pipeline_run_id"] = run_id

        quality = _build_quality_issues(fact, dim_company, run_id, started_at)

        ai_mask = (
            (fact["category_standardized"].eq("AI_ML") | fact["industry_standardized"].eq("AI_ML"))
            & fact["published_year"].between(2022, 2024, inclusive="both")
            & fact["arr_parse_status"].eq("PARSED")
            & fact["arr_usd"].gt(50_000_000)
        )
        ai_export = fact.loc[ai_mask, [
            "article_id", "title", "company_name", "published_date", "category_standardized",
            "arr_usd", "summary", "url", "industry_raw", "founded_year", "headquarters",
            "employee_count", "is_public", "stock_ticker", "company_age",
            "company_size_category", "embedding",
        ]].rename(columns={"category_standardized": "category", "industry_raw": "industry"})

        fact_columns = [
            "article_id", "title", "company_name_raw", "company_name_canonical", "company_name",
            "company_id", "company_match_status", "company_match_method", "company_match_confidence",
            "has_company_metadata", "published_date_raw", "published_date", "published_year",
            "published_quarter", "published_month", "date_parse_status", "date_parse_reason",
            "category_raw", "category_standardized", "revenue_raw", "arr_usd", "source_currency",
            "lower_bound_source", "upper_bound_source", "arr_parse_status", "arr_parse_reason",
            "summary", "url", "author", "word_count", "industry_raw", "industry_standardized",
            "founded_year", "headquarters", "employee_count", "is_public", "stock_ticker",
            "company_age", "company_size_category", "embedding", "top_similar_articles",
            "source_file", "source_row_number", "source_record_hash", "pipeline_run_id",
        ]
        fact = fact[fact_columns]
        tables = {
            "dim_company": dim_company,
            "fact_article": fact,
            "fact_arr_observation": arr_fact,
            "data_quality_issue": quality,
            "bridge_article_similarity": bridge,
        }
        for table_name, frame in tables.items():
            frame.to_csv(staging_dir / MODELED_TABLES[table_name], index=False, na_rep="")
            frame.to_parquet(staging_dir / "{}.parquet".format(table_name), index=False)
        ai_export.to_csv(staging_dir / "ai_articles_enriched.csv", index=False, na_rep="")
        ai_export.to_parquet(staging_dir / "ai_articles_enriched.parquet", index=False)
        _write_duckdb(staging_dir / "analytics.duckdb", dict(tables, ai_articles_enriched=ai_export))

        completed_at = datetime.now(timezone.utc).isoformat()
        run_manifest = {
            "pipeline_run_id": run_id,
            "pipeline_version": PIPELINE_VERSION,
            "started_at": started_at,
            "completed_at": completed_at,
            "status": "SUCCEEDED",
            "arr_parser_version": ARR_PARSER_VERSION,
            "category_mapping_version": CATEGORY_MAPPING_VERSION,
            "company_alias_version": COMPANY_ALIAS_VERSION,
            "embedding": embedding_manifest,
            "inputs": {
                str(articles_path): {"sha256": _sha256_file(articles_path), "rows": len(raw_articles)},
                str(companies_path): {"sha256": _sha256_file(companies_path), "rows": len(metadata)},
            },
            "output_row_counts": {
                **{name: len(frame) for name, frame in tables.items()},
                "ai_articles_enriched": len(ai_export),
            },
        }
        (staging_dir / "pipeline_run.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")

        output_dir.mkdir(parents=True, exist_ok=True)
        for artifact in staging_dir.iterdir():
            destination = output_dir / artifact.name
            if destination.exists() and destination.is_dir():
                shutil.rmtree(destination)
            os.replace(str(artifact), str(destination))
        validation_counts = validate_outputs(output_dir)
        run_manifest["validated_row_counts"] = validation_counts
        (output_dir / "pipeline_run.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
        return run_manifest
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
