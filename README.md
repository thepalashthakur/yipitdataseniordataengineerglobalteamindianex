# YipitData Data Engineering Assignment

This repository implements a local, reproducible pipeline for cleaning synthetic technology-news data, modeling reported company ARR observations, exporting an enriched AI article dataset, and supporting semantic and hybrid search.

The design and operational reasoning are documented in [SOLUTION_APPROACH.md](SOLUTION_APPROACH.md), which serves as the required Data Architecture Document.

## System requirements

- macOS or Linux
- Python 3.9 or newer; Python 3.11 is recommended
- Approximately 4 GB of available memory
- Approximately 2 GB of free disk space for Python packages and the embedding model
- Internet access during initial dependency and `all-MiniLM-L6-v2` model installation

After the first model download, sentence-transformers uses its local Hugging Face cache. An offline deterministic TF-IDF backend is also available for development and constrained environments.

## Installation

Create an isolated environment and install the pinned dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

The commands below use `PYTHONPATH=src`, so an editable package install is not required. Optionally install the package and CLI with:

```bash
.venv/bin/python -m pip install .
```

## Run the complete pipeline

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m yipit_pipeline.cli run \
  --articles tech_news.csv \
  --companies company_metadata.json \
  --output-dir data/output \
  --embedding-backend sentence-transformers \
  --show-progress
```

This command regenerates every required modeled CSV, the AI article export, embeddings, Parquet mirrors, the DuckDB database, and the run manifest.

For an offline or fast development run:

```bash
PYTHONPATH=src .venv/bin/python -m yipit_pipeline.cli run \
  --articles tech_news.csv \
  --companies company_metadata.json \
  --output-dir data/output \
  --embedding-backend tfidf
```

The TF-IDF backend is a reasonable local semantic-search fallback, but the submission output should normally be generated with `sentence-transformers/all-MiniLM-L6-v2`.

## Validate generated outputs

```bash
PYTHONPATH=src .venv/bin/python -m yipit_pipeline.cli validate \
  --output-dir data/output
```

Validation checks required files, primary-key uniqueness, foreign-key relationships, positive ARR values, embedding alignment, and the expected top-three similarity cardinality.

## Generated outputs

All outputs are written to `data/output/`.

| Output | Grain and purpose |
|---|---|
| `dim_company.csv` | One canonical or unresolved source-company entity |
| `fact_article.csv` | One source article, including cleaned fields, metadata, parse statuses, embeddings, and `top_similar_articles` |
| `fact_arr_observation.csv` | One successfully parsed reported ARR observation linked to its source article |
| `data_quality_issue.csv` | One quality rule violation or informational source condition |
| `bridge_article_similarity.csv` | One ranked article-to-article similarity relationship |
| `ai_articles_enriched.csv` | Required AI-related, 2022–2024, ARR greater than $50M export |
| `article_embeddings.npy` | Reusable normalized numeric embedding matrix |
| `article_embedding_index.csv` | Matrix-row-to-article mapping |
| `embedding_manifest.json` | Embedding backend, model, dimension, and normalization metadata |
| `analytics.duckdb` | Typed local analytical database and convenience views |
| `pipeline_run.json` | Input checksums, versions, timestamps, backend, and output counts |

Parquet mirrors are produced for warehouse tables and the AI export. CSV remains the required exchange format.

## Data-model behavior

- `article_id` is the article business key and idempotency key.
- Every successfully parsed revenue value becomes a reported ARR observation.
- ARR is a sourced observation, not company master data.
- Missing, `N/A`, and undisclosed revenue never become zero-value observations.
- Unmatched companies receive stable dimension entities with null metadata, so valid ARR is retained.
- Invalid publication dates produce `UNDATED` observations rather than data loss.
- Latest and quarterly views use only `DATED` observations.
- Raw revenue, date, category, and company values remain available for audit.

Detailed business rules, including currency conversion, date ambiguity, company aliases, taxonomy mappings, idempotency, backfills, and schema evolution, are in [SOLUTION_APPROACH.md](SOLUTION_APPROACH.md).

## DuckDB example queries

Open the database with the DuckDB CLI or Python client.

### ARR observations for a company over time

```sql
SELECT
    c.company_name,
    a.observation_date,
    a.arr_usd,
    a.article_id,
    a.revenue_raw
FROM fact_arr_observation a
JOIN dim_company c USING (company_id)
WHERE c.company_name = 'Snowflake'
ORDER BY a.observation_date, a.article_id;
```

### Trace an observation to its source article

```sql
SELECT
    a.arr_observation_id,
    a.arr_usd,
    f.article_id,
    f.title,
    f.url,
    f.revenue_raw
FROM fact_arr_observation a
JOIN fact_article f USING (article_id)
WHERE a.arr_observation_id = '<observation-id>';
```

### Filter articles by date, category, industry, and ARR

```sql
SELECT article_id, title, company_name, published_date, arr_usd
FROM fact_article
WHERE published_date BETWEEN DATE '2022-01-01' AND DATE '2024-12-31'
  AND category_standardized = 'AI_ML'
  AND industry_standardized = 'AI_ML'
  AND arr_usd > 50000000
ORDER BY arr_usd DESC;
```

### Latest and quarterly reported ARR

```sql
SELECT * FROM latest_company_arr;

SELECT *
FROM company_arr_by_quarter
ORDER BY company_id, observation_year, observation_quarter;
```

## Semantic and hybrid search

Semantic search with SQL-style filters:

```bash
PYTHONPATH=src .venv/bin/python -m yipit_pipeline.cli search \
  --output-dir data/output \
  --query "enterprise adoption of generative AI" \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --category AI_ML \
  --min-arr-usd 50000000 \
  --top-k 5
```

Find articles similar to an existing article:

```bash
PYTHONPATH=src .venv/bin/python -m yipit_pipeline.cli search \
  --output-dir data/output \
  --query ignored \
  --article-id ART0001 \
  --top-k 5
```

Python usage:

```python
from pathlib import Path
from yipit_pipeline.search import ArticleSearchIndex

index = ArticleSearchIndex(Path("data/output"))

results = index.find_similar_articles(
    "companies reporting strong AI revenue growth",
    top_k=5,
)

hybrid_results = index.hybrid_search(
    "large language model adoption",
    top_k=5,
    start_date="2022-01-01",
    end_date="2024-12-31",
    categories=["AI_ML"],
    min_arr_usd=50_000_000,
)
```

## Tests

Run all unit and integration tests:

```bash
PYTHONPATH=src .venv/bin/pytest
```

The test suite covers revenue formats and conversion, date formats and ambiguity, category and industry taxonomies, company aliases and unresolved entities, size boundaries, semantic similarity, full-output validation, DuckDB views, hybrid search, lineage, and idempotent reruns.

## Reliability and reruns

The pipeline builds outputs in a temporary staging directory, validates the complete dataset, and publishes files only after successful materialization. Stable UUID5 identifiers are derived from business keys. Re-running identical inputs therefore updates the complete published state without creating duplicate articles, company entities, ARR observations, or similarity relationships.

`pipeline_run.json` records source checksums and transformation/configuration versions. Production backfills and schema-evolution handling are described in [SOLUTION_APPROACH.md](SOLUTION_APPROACH.md).

