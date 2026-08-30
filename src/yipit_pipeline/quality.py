"""Data-quality issue creation and output assertions."""

from typing import Any, Dict, Optional
from uuid import uuid5

from .config import ISSUE_NAMESPACE


def quality_issue(
    pipeline_run_id: str,
    created_at: str,
    source_table: str,
    source_record_id: str,
    field_name: str,
    rule_code: str,
    severity: str,
    raw_value: Any,
    message: Optional[str],
) -> Dict[str, Any]:
    stable_key = "{}:{}:{}:{}".format(source_table, source_record_id, field_name, rule_code)
    return {
        "issue_id": str(uuid5(ISSUE_NAMESPACE, stable_key)),
        "pipeline_run_id": pipeline_run_id,
        "source_table": source_table,
        "source_record_id": source_record_id,
        "field_name": field_name,
        "rule_code": rule_code,
        "severity": severity,
        "raw_value": raw_value,
        "message": message or "",
        "created_at": created_at,
    }

