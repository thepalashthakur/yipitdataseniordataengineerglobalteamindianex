from yipit_pipeline.matching import resolve_company


METADATA = {
    "OpenAI": {},
    "Amazon Web Services": {},
    "SpaceX": {},
}


def test_exact_and_alias_matches_share_canonical_identity():
    exact = resolve_company("Amazon Web Services", METADATA)
    alias = resolve_company("AWS", METADATA)
    assert exact.company_id == alias.company_id
    assert alias.company_name_canonical == "Amazon Web Services"
    assert alias.company_match_method == "ALIAS"
    assert alias.has_company_metadata is True


def test_unresolved_company_gets_stable_entity():
    first = resolve_company("Cohere", METADATA)
    second = resolve_company("Cohere", METADATA)
    assert first.company_id == second.company_id
    assert first.company_match_status == "UNRESOLVED"
    assert first.company_entity_name == "Cohere"
    assert first.has_company_metadata is False


def test_ambiguous_company_is_not_silently_mapped():
    result = resolve_company("The Boring Company / SpaceX", METADATA)
    assert result.company_match_status == "AMBIGUOUS"
    assert result.company_name_canonical is None
    assert result.has_company_metadata is False

