"""Versioned business rules used by the pipeline."""

from decimal import Decimal
from uuid import UUID


PIPELINE_VERSION = "1.0.0"
ARR_PARSER_VERSION = "1.0.0"
CATEGORY_MAPPING_VERSION = "1.0.0"
COMPANY_ALIAS_VERSION = "1.0.0"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COMPANY_NAMESPACE = UUID("f34c9717-0844-42e7-b1f3-22d29d332dd1")
ARR_NAMESPACE = UUID("0f7f50c4-ddb2-437f-976b-6cfa447d42fc")
ISSUE_NAMESPACE = UUID("82116fcb-14e3-4a62-9a9e-d15451690cab")

CURRENCY_TO_USD = {
    "USD": Decimal("1"),
    "EUR": Decimal("1.1"),
    "GBP": Decimal("1.27"),
    "JPY": Decimal("0.006666666666666666666666666667"),
}

MISSING_MARKERS = {"", "n/a", "na", "null", "none", "nan", "missing"}
UNDISCLOSED_MARKERS = {"not disclosed", "undisclosed"}

CATEGORY_MAPPING = {
    "ai/ml": "AI_ML",
    "ai & ml": "AI_ML",
    "artificial intelligence": "AI_ML",
    "machine learning": "AI_ML",
    "cloud": "CLOUD",
    "cloud computing": "CLOUD",
    "cloud services": "CLOUD",
    "fintech": "FINTECH",
    "financial technology": "FINTECH",
    "finance": "FINTECH",
    "security": "CYBERSECURITY",
    "infosec": "CYBERSECURITY",
    "cybersecurity": "CYBERSECURITY",
    "analytics": "DATA_ANALYTICS",
    "data analytics": "DATA_ANALYTICS",
    "big data": "DATA_ANALYTICS",
    "saas": "SOFTWARE_SAAS",
    "software": "SOFTWARE_SAAS",
    "enterprise software": "SOFTWARE_SAAS",
}

INDUSTRY_MAPPING = {
    "ai/ml": "AI_ML",
    "cloud computing": "CLOUD",
    "fintech": "FINTECH",
    "cybersecurity": "CYBERSECURITY",
    "data analytics": "DATA_ANALYTICS",
    "saas": "SOFTWARE_SAAS",
}

# Explicit mappings are intentionally reviewed rather than inferred at run time.
COMPANY_ALIASES = {
    "AWS": "Amazon Web Services",
    "Amazon Web Services (AWS)": "Amazon Web Services",
    "Microsoft Azure": "Microsoft",
    "Azure": "Microsoft",
    "DeepMind": "Google DeepMind",
    "Google Deepmind": "Google DeepMind",
    "Meta AI Research": "Meta AI",
    "Facebook AI Research": "Meta AI",
    "Nvidia": "NVIDIA",
    "NVIDIA Corporation": "NVIDIA",
    "Open AI": "OpenAI",
    "OpenAI Inc.": "OpenAI",
    "Databricks Inc.": "Databricks",
    "Snowflake Inc.": "Snowflake",
    "Stripe Inc.": "Stripe",
    "CloudFlare": "Cloudflare",
    "Data Robot": "DataRobot",
    "Mongo DB": "MongoDB",
    "Palantir Technologies": "Palantir",
}

AMBIGUOUS_COMPANY_NAMES = {"The Boring Company / SpaceX"}

