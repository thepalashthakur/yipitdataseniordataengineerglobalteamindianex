# YipitData Data Engineering Assignment — Solution Approach

## 1. Overview

This document serves as the required **Data Architecture Document** as well as the implementation solution approach.

The goal is to build a small, reproducible local Python pipeline that converts the supplied synthetic technology-news data into:

1. Queryable warehouse-style tables for analyzing company ARR observations over time.
2. An enriched AI-article export for the requested date and ARR filters.
3. A reusable semantic-search index and hybrid search interface.

The implementation should remain easy to run locally while demonstrating patterns that can scale to recurring batches: deterministic transformations, explicit schemas, source lineage, data-quality reporting, idempotent writes, and separation between raw, cleaned, and analytical data.

The proposed implementation uses Python, pandas, PyArrow/Parquet, DuckDB, and sentence-transformers. CSV files are produced for every modeled table because they are required deliverables. Parquet files and a DuckDB database may also be generated for efficient typed querying.

## 2. Design Principles

- **Preserve source truth:** Retain raw revenue, company name, date, and other source fields alongside normalized values.
- **Never invent ARR:** Only successfully parsed revenue values become ARR observations. Missing, undisclosed, or invalid values remain visible in article and quality tables but are excluded from the ARR fact table.
- **Make lineage explicit:** Every ARR observation links to exactly one source article through `article_id`.
- **Be deterministic:** The same inputs and configuration produce the same identifiers and outputs.
- **Be idempotent:** Re-running the pipeline replaces or upserts records by stable keys rather than appending duplicates.
- **Separate concerns:** Extraction, validation, normalization, modeling, embedding generation, and exports are distinct stages with reusable functions.
- **Expose uncertainty:** Ambiguous dates, unresolved companies, and failed revenue parses receive status and reason fields rather than being silently coerced.
- **Keep business rules configurable:** Currency rates, category mappings, aliases, thresholds, and date policies live in version-controlled configuration or constants.

## 3. Proposed Project Structure

```text
.
├── company_metadata.json
├── tech_news.csv
├── README.md
├── SOLUTION_APPROACH.md                # required Data Architecture Document
├── requirements.txt
├── pyproject.toml                     # optional packaging/tool configuration
├── src/
│   └── yipit_pipeline/
│       ├── __init__.py
│       ├── config.py
│       ├── schemas.py
│       ├── cleaning.py
│       ├── matching.py
│       ├── modeling.py
│       ├── embeddings.py
│       ├── search.py
│       ├── quality.py
│       ├── pipeline.py
│       └── cli.py
├── tests/
│   ├── test_revenue_cleaning.py
│   ├── test_date_cleaning.py
│   ├── test_category_mapping.py
│   ├── test_company_matching.py
│   ├── test_modeling.py
│   ├── test_search.py
│   └── test_pipeline_integration.py
├── data/
│   ├── raw/                           # optional immutable snapshots
│   ├── interim/
│   └── output/
│       ├── dim_company.csv
│       ├── fact_article.csv
│       ├── fact_arr_observation.csv
│       ├── data_quality_issue.csv
│       ├── bridge_article_similarity.csv
│       ├── ai_articles_enriched.csv
│       ├── article_embeddings.npy
│       ├── article_embedding_index.csv
│       ├── embedding_manifest.json
│       └── analytics.duckdb            # optional
└── scripts/
    └── run_pipeline.py
```

The top-level source files may be read directly for the assignment. In a production-style run, they would first be copied into an immutable, batch-stamped raw directory.

## 4. Pipeline Flow

```text
tech_news.csv ───────────┐
                        ├─> validate schema ─> clean articles ─┐
company_metadata.json ──┘                                      │
                                                               ├─> warehouse tables
aliases + mappings ────────────────────────────────────────────┤
                                                               ├─> AI article export
title + summary ─> embedding model ─> vector index ────────────┤
                                                               └─> DuckDB hybrid search
```

The run is divided into the following stages:

1. Load source files and assign batch metadata.
2. Validate required columns and basic types.
3. Normalize dates, revenue/ARR, and categories.
4. Resolve article company names to canonical company identities.
5. Enrich articles with company metadata and derived attributes.
6. Build warehouse-style dimensions and facts.
7. Generate article embeddings and nearest-neighbor relationships.
8. Export each modeled table and the required AI article dataset.
9. Load typed outputs into DuckDB and create useful views.
10. Run data-quality assertions and write a run summary.

## 5. Input Validation

### 5.1 Article schema

Expected source columns:

```text
article_id, title, company_name, published_date, category, revenue,
summary, url, author, word_count
```

Validation rules:

- Reject the run if a required column is absent.
- Require non-null `article_id`, title, company name, summary, and URL.
- Require `article_id` to be unique within the batch.
- Validate `word_count` as a nullable non-negative integer.
- Record duplicate URLs as quality issues, if any.
- Preserve unexpected columns in the bronze/raw representation so schema additions are not silently lost.

### 5.2 Company metadata schema

Expected fields per company:

```text
founded_year, headquarters, employee_count, industry,
is_public, stock_ticker
```

Validation should check field presence and type without attempting to correct implausible synthetic values. For example, a private company with a ticker should be flagged as a metadata consistency warning but preserved exactly as provided.

## 6. Cleaning and Standardization

### 6.1 ARR parsing

Implement a pure function with an explicit result object:

```python
parse_arr(raw_value: object) -> ArrParseResult
```

Suggested result fields:

```text
raw_value
arr_usd
source_currency
lower_bound_source
upper_bound_source
parse_status
parse_reason
```

`arr_usd` is a nullable integer. `parse_status` should distinguish at least:

- `PARSED`
- `MISSING`
- `UNDISCLOSED`
- `INVALID_FORMAT`
- `UNSUPPORTED_CURRENCY`

#### Parsing algorithm

1. Treat null, blank, `N/A`, `NA`, `null`, `none`, and similar markers as missing.
2. Treat `Not disclosed` as undisclosed, not as zero.
3. Normalize case and whitespace while retaining the original string.
4. Detect currency from symbols or codes:
   - `$` or `USD` → USD
   - `€` or `EUR` → EUR
   - `£` or `GBP` → GBP
   - `¥` or `JPY` → JPY
   - no currency symbol or code → USD for this exercise
5. Detect a range and parse both endpoints in the same currency and scale.
6. Normalize numeric separators and suffixes:
   - `K` or `thousand` → `1_000`
   - `M` or `million` → `1_000_000`
   - `B` or `billion` → `1_000_000_000`
7. For a range, compute the midpoint in source currency before conversion.
8. Convert to USD using decimal arithmetic:
   - EUR × `1.10`
   - GBP × `1.27`
   - JPY ÷ `150`
   - USD × `1.00`
9. Round once at the end using a documented rule, preferably half-up, and return an integer.
10. Reject negative values, contradictory currency markers, and partially parsed strings.

Examples:

| Raw value | Normalized result |
|---|---:|
| `$5,200,000,000` | `5,200,000,000` |
| `5.2B` | `5,200,000,000` |
| `5.2 billion` | `5,200,000,000` |
| `500M USD` | `500,000,000` |
| `$10M - $20M` | `15,000,000` |
| `€2.1B` | `2,310,000,000` |
| `£100M` | `127,000,000` |
| `¥15B` | `100,000,000` |
| `Not disclosed` | null; excluded from ARR fact |

The parser should use anchored regular expressions and verify that the entire normalized input was consumed. This prevents strings with valid-looking substrings from being silently accepted.

### 6.2 Date normalization

Implement:

```python
parse_published_date(raw_value: object) -> DateParseResult
```

The source contains six observed shapes:

- `YYYY-MM-DD`
- `YYYY-MM-DDT00:00:00Z`
- `MM/DD/YYYY`
- `DD-MM-YYYY`
- `DD Mon YYYY`
- `Month DD, YYYY`

Use an ordered, explicit format list instead of unrestricted inference. The documented ambiguity policy is:

- Slash-separated numeric dates are interpreted as US `MM/DD/YYYY`.
- Dash-separated dates with the year last are interpreted as `DD-MM-YYYY`.
- ISO timestamps are normalized to their UTC calendar date.
- Invalid or missing values produce a null date and an explanatory status.

Derived columns:

```text
published_date       DATE
published_year       INTEGER
published_quarter    INTEGER (1–4)
published_month      INTEGER (1–12)
date_parse_status    STRING
date_parse_reason    STRING
```

Rows with an invalid date remain in the article fact table. If their ARR parses successfully, they also remain in the ARR fact with a null `observation_date` and `observation_status = 'UNDATED'`. They are excluded from latest, quarterly, and date-filtered outputs until the date is corrected.

### 6.3 Category standardization

Use a version-controlled mapping rather than scattered conditional logic. A reasonable taxonomy is:

| Raw values | Standard category |
|---|---|
| `AI/ML`, `AI & ML`, `Artificial Intelligence`, `Machine Learning` | `AI_ML` |
| `Cloud`, `Cloud Computing`, `Cloud Services` | `CLOUD` |
| `FinTech`, `Financial Technology`, `Finance` | `FINTECH` |
| `Security`, `InfoSec`, `Cybersecurity` | `CYBERSECURITY` |
| `Analytics`, `Data Analytics`, `Big Data` | `DATA_ANALYTICS` |
| `SaaS`, `Software`, `Enterprise Software` | `SOFTWARE_SAAS` |

Retain both `category_raw` and `category_standardized`. Unknown new categories should map to `OTHER` or `UNMAPPED`, be counted in the quality report, and not crash the pipeline.

### 6.4 Industry standardization

Normalize company industry independently from article category so both fields can be filtered consistently. Retain the supplied value as `industry_raw` and expose the normalized value as `industry_standardized`.

| Raw industry | Standard industry |
|---|---|
| `AI/ML` | `AI_ML` |
| `Cloud Computing` | `CLOUD` |
| `FinTech` | `FINTECH` |
| `Cybersecurity` | `CYBERSECURITY` |
| `Data Analytics` | `DATA_ANALYTICS` |
| `SaaS` | `SOFTWARE_SAAS` |

Unknown values map to `OTHER` or `UNMAPPED` and produce a quality metric. The AI article filter qualifies a record when either `category_standardized = 'AI_ML'` or `industry_standardized = 'AI_ML'`.

### 6.5 Company identity resolution

Company matching should be conservative and auditable. Use this sequence:

1. Exact match against metadata keys.
2. Exact match after safe normalization such as trimming, case folding, and whitespace normalization.
3. Explicit alias mapping maintained in configuration.
4. Optional fuzzy-match suggestion for review, never an automatic production match below a strict threshold.
5. For an unmatched source company, create a stable source-derived company entity, mark it unresolved, and preserve its source name. This allows valid ARR observations to remain in the warehouse even when enrichment metadata is unavailable.

Examples of deterministic aliases:

```yaml
AWS: Amazon Web Services
Amazon Web Services (AWS): Amazon Web Services
Open AI: OpenAI
OpenAI Inc.: OpenAI
DeepMind: Google DeepMind
Google Deepmind: Google DeepMind
Nvidia: NVIDIA
NVIDIA Corporation: NVIDIA
Databricks Inc.: Databricks
Snowflake Inc.: Snowflake
CloudFlare: Cloudflare
Data Robot: DataRobot
Mongo DB: MongoDB
Palantir Technologies: Palantir
Stripe Inc.: Stripe
```

Aliases such as `Azure` → `Microsoft` and `Facebook AI Research` → `Meta AI` may be included, but the relationship should be documented as a product/subsidiary-to-parent resolution rather than a spelling correction.

Companies absent from metadata—such as Cohere, xAI, Mistral AI, Perplexity AI, and Hugging Face—receive stable unresolved company entities with null enrichment fields unless metadata is explicitly added. The compound source value `The Boring Company / SpaceX` receives its own stable unresolved entity and is flagged as ambiguous rather than silently assigned to SpaceX. Neither condition causes an otherwise valid ARR observation to be discarded.

Suggested resolution columns:

```text
company_name_raw
company_id
company_name_canonical
company_match_status
company_match_method
company_match_confidence
has_company_metadata
```

### 6.6 Metadata enrichment

For companies resolved to supplied metadata, add:

- industry
- founded year
- headquarters
- employee count
- public/private status
- stock ticker
- company age at publication
- company size category

Company age is calculated as:

```text
published_year - founded_year
```

It is null when the date or founding year is unavailable. Unresolved source-derived companies have null metadata fields, including company age. A negative result is retained as null and logged as a data-quality issue.

Company size rules:

- `SMALL`: fewer than 10,000 employees
- `MEDIUM`: 10,000 through 30,000 employees, inclusive
- `LARGE`: more than 30,000 employees
- `UNKNOWN`: missing or invalid employee count

## 7. Data Model

### 7.1 `dim_company`

**Grain:** one row per resolved canonical company or stable unresolved source-company entity.

| Column | Purpose |
|---|---|
| `company_id` | Stable deterministic company identifier |
| `company_name` | Canonical metadata name, or preserved source name for an unresolved entity |
| `company_name_raw` | Representative source name where applicable |
| `company_match_status` | Exact, normalized, alias, unresolved, or ambiguous |
| `has_company_metadata` | Whether supplied metadata was joined successfully |
| `industry_raw` | Supplied company industry |
| `industry_standardized` | Normalized industry used for filtering |
| `founded_year` | Founding year |
| `headquarters` | Headquarters text |
| `employee_count` | Supplied employee count |
| `company_size_category` | Derived size band |
| `is_public` | Supplied public/private flag |
| `stock_ticker` | Nullable ticker |
| `source_file` | Metadata source filename |
| `record_hash` | Hash of business attributes for change detection |
| `pipeline_run_id` | Run that last materialized the record |

For this assignment, `company_id` can be a UUID5 derived from a namespace plus the resolved canonical name. Unresolved entities use the same namespace with their normalized source-company name, so IDs remain deterministic across runs. In a larger system, this dimension could become slowly changing if metadata history is required.

### 7.2 `fact_article`

**Grain:** one row per source article (`article_id`).

This table contains the cleaned article, canonical company relationship, date attributes, standardized category, parsed ARR result, metadata enrichment, and all relevant lineage/status fields. It includes articles with missing ARR, invalid dates, or unresolved companies so failed records remain inspectable.

Important columns include:

```text
article_id
title
company_name_raw
company_id
company_name_canonical
company_match_status
published_date_raw
published_date
published_year
published_quarter
published_month
date_parse_status
category_raw
category_standardized
revenue_raw
arr_usd
arr_parse_status
arr_parse_reason
summary
url
author
word_count
industry
industry_raw
industry_standardized
founded_year
headquarters
employee_count
is_public
stock_ticker
company_age
company_size_category
top_similar_articles
source_file
source_row_number
source_record_hash
pipeline_run_id
```

### 7.3 `fact_arr_observation`

**Grain:** one valid ARR observation reported by one source article.

Only records with all of the following enter this fact:

- valid parsed ARR
- stable company entity, whether metadata-resolved or unresolved
- non-null source article ID

A valid date is not an admission requirement. Records with invalid or missing dates remain as undated ARR observations so valid revenue claims are not silently discarded.

Recommended columns:

```text
arr_observation_id
company_id
article_id
observation_date
observation_status
date_parse_status
observation_year
observation_quarter
observation_month
arr_usd
revenue_raw
source_currency
company_match_status
parse_method_version
source_file
pipeline_run_id
```

`arr_observation_id` should be deterministic, for example UUID5 of `article_id + observation_type`. It must not be based on ARR value because a corrected reprocessing of the same article should update the observation rather than create a duplicate.

This table represents reported observations, not authoritative company master ARR. Valid observations are not discarded merely because company metadata or a usable publication date is missing. Time-based views must filter to `observation_status = 'DATED'`. Multiple articles on the same date may produce multiple observations and should remain separate unless an explicit deduplication business rule is introduced.

### 7.4 `data_quality_issue`

**Grain:** one detected issue for one source record and rule.

```text
issue_id
pipeline_run_id
source_table
source_record_id
field_name
rule_code
severity
raw_value
message
created_at
```

Typical rule codes include:

- `ARR_MISSING`
- `ARR_UNDISCLOSED`
- `ARR_INVALID_FORMAT`
- `DATE_INVALID`
- `DATE_AMBIGUOUS_POLICY_APPLIED`
- `COMPANY_UNRESOLVED`
- `COMPANY_AMBIGUOUS`
- `CATEGORY_UNMAPPED`
- `METADATA_PUBLIC_TICKER_INCONSISTENT`

Expected missing ARR values may be recorded as informational metrics rather than warnings to avoid overwhelming the issue table.

## 8. Analytical Views

DuckDB views make recurring query patterns explicit while keeping exported facts atomic.

### Latest reported ARR per company

```sql
CREATE OR REPLACE VIEW latest_company_arr AS
SELECT * EXCLUDE (row_num)
FROM (
    SELECT
        a.*,
        ROW_NUMBER() OVER (
            PARTITION BY company_id
            ORDER BY observation_date DESC, article_id DESC
        ) AS row_num
    FROM fact_arr_observation a
    WHERE observation_status = 'DATED'
)
WHERE row_num = 1;
```

The tie-breaker is deterministic but does not claim that one same-day article is more authoritative than another.

### Quarterly observations

The base quarterly view should expose every observation rather than automatically summing ARR, because ARR is a point-in-time metric and summing reports is usually incorrect.

```sql
CREATE OR REPLACE VIEW company_arr_by_quarter AS
SELECT
    company_id,
    observation_year,
    observation_quarter,
    COUNT(*) AS observation_count,
    MIN(arr_usd) AS min_reported_arr_usd,
    MAX(arr_usd) AS max_reported_arr_usd,
    ARG_MAX(arr_usd, observation_date) AS latest_reported_arr_usd,
    MAX(observation_date) AS latest_observation_date
FROM fact_arr_observation
WHERE observation_status = 'DATED'
GROUP BY 1, 2, 3;
```

If multiple observations share the latest date, consumers should inspect the source articles rather than assuming they are equivalent.

## 9. Required AI Article Export

Create `ai_articles_enriched.csv` from `fact_article` using these rules:

```text
(
  category_standardized = 'AI_ML'
  OR industry_standardized = 'AI_ML'
)
AND published_date BETWEEN 2022-01-01 AND 2024-12-31
AND arr_parse_status = 'PARSED'
AND arr_usd > 50,000,000
```

The threshold is strictly greater than $50M, not greater than or equal to it.

Required columns, in assignment order:

```text
article_id
title
company_name
published_date
category
arr_usd
summary
url
industry
founded_year
headquarters
employee_count
is_public
stock_ticker
company_age
company_size_category
embedding
```

Set export `company_name` to `COALESCE(company_name_canonical, company_name_raw)` so qualifying unresolved-company articles remain usable. Retain the raw and canonical values separately in `fact_article`. Use the standardized category in the export; document these choices in the README.

Because embedding generation is labeled bonus while `embedding` appears in the mandatory export schema, the safest interpretation is to implement embeddings as part of the standard pipeline. Store the CSV embedding as compact JSON, such as `[0.0123,-0.0456,...]`, while keeping the reusable numeric matrix in a `.npy` file.

## 10. Embeddings and Semantic Search

### 10.1 Embedding generation

- Model: `sentence-transformers/all-MiniLM-L6-v2`.
- Input text: normalized `title + "\n\n" + summary`.
- Batch encode all articles.
- Normalize vectors to unit length so cosine similarity becomes a dot product.
- Persist:
  - `article_embeddings.npy` as a float32 matrix.
  - `article_embedding_index.csv` mapping matrix row to `article_id`.
  - JSON embeddings in DuckDB and the required AI export.
- Record model name, model revision if available, embedding dimension, input-text hash, and generation time.

Cache embeddings by a hash of:

```text
article_id + normalized_text + model_name + model_revision
```

Only new or changed articles need to be re-embedded.

### 10.2 Similarity search

```python
find_similar_articles(query_text: str, top_k: int = 5) -> list[SearchResult]
```

Algorithm:

1. Embed and normalize the query with the same model.
2. Calculate matrix-vector dot products against normalized article embeddings.
3. Select the highest `top_k` scores using `numpy.argpartition` followed by exact sorting.
4. Return article IDs, scores, and useful metadata.

For article-to-article recommendations, exclude the current `article_id`, then take the top three. Every row in `fact_article` must contain `top_similar_articles` as a JSON array of exactly three article IDs when at least three alternatives exist. The field must contain no current-article ID and no duplicate IDs. A normalized bridge table should also be generated for queryability and auditability:

```text
bridge_article_similarity
  source_article_id
  similar_article_id
  similarity_rank
  similarity_score
  embedding_model
```

Export this modeled table as `bridge_article_similarity.csv`. The bridge complements rather than replaces the required `top_similar_articles` field.

### 10.3 Hybrid search

The hybrid function first applies SQL filters in DuckDB, then ranks only the candidate rows by vector similarity:

```python
hybrid_search(
    query_text: str,
    top_k: int = 5,
    start_date: date | None = None,
    end_date: date | None = None,
    categories: list[str] | None = None,
    industries: list[str] | None = None,
    min_arr_usd: int | None = None,
) -> list[SearchResult]
```

This is simple and efficient for the supplied dataset. At much larger scale, embeddings could move to a vector database or a database extension supporting approximate nearest-neighbor indexes.

## 11. Idempotency and Incremental Processing

Each run receives a `pipeline_run_id`, input checksums, start/end timestamps, code/config version, and status.

For the local assignment, the safest publishing method is:

1. Read inputs and build outputs in a temporary run directory.
2. Validate row counts, uniqueness, foreign keys, and required exports.
3. Write each file completely.
4. Atomically replace the previous published output only after all checks pass.

This guarantees that a failed run cannot leave a mixture of old and new tables.

For incremental batches:

- Treat `article_id` as the source business key.
- Calculate a stable source-record hash from normalized source columns.
- Skip unchanged articles.
- Update changed articles using the same deterministic keys.
- Insert new articles.
- Optionally retain batch history in a bronze table.
- Recompute downstream ARR and embeddings only for affected articles.

CSV is not a transactional update format, so each CSV should be regenerated from the complete modeled state. DuckDB tables can use delete-and-insert or merge-like logic keyed by stable IDs.

## 12. Schema Evolution and Backfills

- Define expected schemas centrally and validate them before transformation.
- Additive source columns should be retained in raw storage and logged for review.
- Missing or type-changed required columns should fail fast with a useful error.
- Include `transformation_version`, `arr_parser_version`, `category_mapping_version`, and `company_alias_version` in run metadata.
- A backfill runs the same pipeline with an explicit input batch or date range and publishes through the same validation gates.
- Business-rule changes should trigger a full deterministic rebuild or a targeted backfill of affected records.
- Never overwrite immutable raw input snapshots.

## 13. Data-Quality Checks

The pipeline should fail on structural integrity problems and warn on expected source imperfections.

### Blocking checks

- Required input files and columns exist.
- `article_id` is non-null and unique.
- Modeled primary keys are unique.
- Every ARR fact references an existing article and company.
- Every ARR fact has a positive integer `arr_usd` and valid observation date.
- Published output files were written and can be read back.
- Embedding count equals embedded article count.
- Embeddings have the expected dimension and finite numeric values.

### Warning/reporting checks

- Revenue parse success, missing, undisclosed, and failure counts.
- Date parse success and failure counts by source format.
- Company match rate by method and unresolved name.
- Unmapped category counts.
- Metadata consistency warnings.
- AI export row count and filter-boundary checks.
- Changes in source or output counts relative to the previous run.

Quality thresholds should be configurable. For example, a sudden company match-rate drop may fail production even though unresolved records are technically supported.

## 14. Testing Strategy

### Unit tests

Use parameterized tests for every documented revenue and date format, including:

- currency symbols and codes
- suffix and word multipliers
- comma-separated amounts
- ranges and midpoint calculations
- null and undisclosed markers
- invalid strings and mixed currencies
- conversion rounding
- leap days and invalid calendar dates
- all six observed date shapes
- ambiguous slash-date policy
- category mappings and unknown categories
- industry mappings and unknown industries
- exact, normalized, alias, unresolved, and ambiguous company matching
- retention of valid ARR observations for unresolved and ambiguous company entities
- retention and explicit status of valid but undated ARR observations
- currency-less revenue values defaulting to USD
- company-size boundaries at 9,999, 10,000, 30,000, and 30,001

### Model tests

- Article grain remains one row per `article_id`.
- ARR fact contains only valid observations.
- ARR-to-article and ARR-to-company foreign keys resolve.
- Deterministic identifiers remain stable across runs.
- Latest and quarterly views return expected records for controlled fixtures.
- AI export satisfies every filter and required-column rule.

### Search tests

- Embeddings are deterministic within the pinned model/runtime tolerance.
- Query results are ordered by decreasing score.
- Article-to-article search excludes the source article.
- Every article has three unique `top_similar_articles` when at least three alternatives exist.
- `top_k` handles zero, oversized, and invalid values.
- SQL filters are applied before similarity ranking.

### Integration tests

Run the pipeline against a small fixture and verify:

- all expected files exist
- schemas and dtypes are correct
- output can be loaded into DuckDB
- a second identical run creates no duplicate records
- a changed source article updates rather than duplicates its ARR observation
- a forced failure does not replace the prior successful outputs

## 15. Dependency and Runtime Choices

Suggested dependencies:

```text
pandas
pyarrow
duckdb
numpy
sentence-transformers
PyYAML
pydantic
pytest
```

Pin compatible versions in `requirements.txt` or a lock file. Python 3.11 is a reasonable target. The embedding model is downloaded on first use, so the README should state the network and disk requirements and explain how to use a pre-populated local model cache for offline runs.

## 16. Command-Line Interface

Suggested commands:

```bash
python -m yipit_pipeline.cli run \
  --articles tech_news.csv \
  --companies company_metadata.json \
  --output-dir data/output

python -m yipit_pipeline.cli validate --output-dir data/output

python -m yipit_pipeline.cli search \
  --query "enterprise adoption of generative AI" \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --min-arr-usd 50000000 \
  --top-k 5
```

The standard `run` command should regenerate every required modeled CSV and `ai_articles_enriched.csv`. Optional flags may skip embedding generation during rapid development, but the submission’s documented full command should include it.

The README must explicitly document:

- supported Python version, memory/disk expectations, network needs, and embedding-model download behavior
- installation and dependency setup
- the full pipeline command
- how to regenerate every required CSV output from a clean checkout
- example ARR-over-time, source-lineage, filtered article, semantic-search, and hybrid-search queries
- the location of this Data Architecture Document

## 17. Observability and Run Metadata

Write a small `pipeline_run` table or JSON manifest containing:

```text
pipeline_run_id
started_at
completed_at
status
code_version
config_versions
input_paths
input_checksums
input_row_counts
output_row_counts
quality_metrics
embedding_model
error_message
```

Use structured logging with the run ID on every message. Do not log full embeddings or unnecessarily duplicate sensitive source text. A production deployment would publish these metrics to an orchestration and monitoring system.

## 18. Production Evolution

Beyond the local batch, the same logical stages could run under Airflow, Dagster, or another orchestrator with object storage for immutable raw batches and a warehouse for modeled tables.

Likely production improvements:

- schema registry or data contracts
- partitioned Parquet/Iceberg/Delta storage
- merge-based incremental models
- slowly changing company metadata
- reviewed company-identity mapping workflow
- effective-date currency conversion rather than fixed exercise rates
- model registry and embedding-version migration
- vector database or approximate nearest-neighbor index
- automated quality thresholds, alerts, retries, and lineage publication
- access control, retention, and privacy policies

The most important invariant should remain unchanged: a reported ARR observation is an article-derived claim with a date and source, not a timeless company attribute.

## 19. Assumptions and Decisions to Document

1. Every successfully parsed source revenue value is treated as reported ARR because the assignment explicitly instructs this interpretation; a missing metadata match or unusable publication date does not remove the observation.
2. Missing or undisclosed revenue does not create an ARR observation.
3. Slash dates are `MM/DD/YYYY`; year-last dash dates are `DD-MM-YYYY`.
4. Fixed exercise currency rates are used without attempting historical FX lookup.
5. Currency conversion is rounded once, after range midpoint calculation and conversion.
6. A valid currency-less value such as `5.2B` is interpreted as USD for this exercise.
7. Article category and company industry are normalized independently; either may qualify an article as AI-related.
8. The AI export date window is inclusive of 2022-01-01 through 2024-12-31.
9. The ARR threshold is strictly greater than $50,000,000.
10. Unresolved or ambiguous source company names receive stable company entities with null metadata, allowing their valid ARR observations to remain queryable and auditable.
11. Embeddings are implemented despite their bonus label because the mandatory export schema includes an embedding field.
12. Repeated article titles are not duplicates; `article_id` is the source key.
13. Synthetic metadata is enriched as supplied, with inconsistencies flagged rather than externally corrected.

## 20. Definition of Done

The solution is complete when:

- one documented command runs the full pipeline locally
- tests pass
- all required input formats are handled
- raw values and source lineage are preserved
- company mismatches and failed parses are inspectable
- `dim_company.csv`, `fact_article.csv`, `fact_arr_observation.csv`, `data_quality_issue.csv`, and `bridge_article_similarity.csv` are generated
- `ai_articles_enriched.csv` contains the required columns and only qualifying rows
- every valid ARR value is retained even if company metadata cannot be matched or the publication date is unusable; undated records are explicitly flagged
- every article contains a compliant `top_similar_articles` field, and embeddings and similarity relationships are reusable and versioned
- DuckDB can answer ARR-over-time, source-lineage, filtered article, latest ARR, quarterly ARR, semantic, and hybrid queries
- an identical rerun produces no duplicate modeled records
- README explicitly covers system requirements, installation, pipeline execution, full CSV regeneration, example queries/usage, and the location of this architecture document
- this solution approach is delivered as the Data Architecture Document and explains the model, reliability beyond a local batch, assumptions, trade-offs, backfills, and schema evolution
