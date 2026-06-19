"""Offline collect: normalize -> gates -> scrub -> dedup on in-repo fixtures.

NO network: every input is a hand-built GitHub-API-shaped dict. We prove the
collector's pure transforms (``docs/tasks.md`` §5) work end-to-end on the base
install:

* :func:`normalize_repo` maps a REST/GraphQL payload onto a ``raw_harvest`` record
  that validates against ``schemas/raw_harvest.schema.json``;
* :func:`passes_quality_gates` accepts good repos and rejects bad ones with reasons;
* :func:`scrub_record` redacts emails/secrets from stored excerpts;
* :func:`dedup_records` collapses exact + near-duplicate repos (datasketch-free
  fallback path).

The whole ``collect`` package imports with core deps only (datasketch is lazy).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from collect.filters import (
    GateConfig,
    dedup_records,
    is_redistributable,
    minhash_available,
    passes_quality_gates,
    scrub_record,
    scrub_text,
)
from collect.normalize import guess_domain, normalize_repo, tier_size_proxy

ROOT = Path(__file__).resolve().parents[2]
RAW_HARVEST_SCHEMA = json.loads((ROOT / "schemas" / "raw_harvest.schema.json").read_text())


# --------------------------------------------------------------------------- #
# In-repo fixture payloads (no network)                                        #
# --------------------------------------------------------------------------- #


def _good_repo(name: str = "ical-stats", *, stars: int = 420) -> dict[str, Any]:
    """A REST-shaped repo payload that should pass the gates."""
    return {
        "full_name": f"acme/{name}",
        "owner": {"login": "acme"},
        "name": name,
        "html_url": f"https://github.com/acme/{name}",
        "default_branch": "main",
        "description": "A small CLI that summarizes your calendar into a weekly report.",
        "topics": ["cli", "calendar", "ics", "report"],
        "language": "Python",
        "size": 800,
        "stargazers_count": stars,
        "pushed_at": "2026-05-01T00:00:00Z",
        "archived": False,
        "fork": False,
        "license": {"spdx_id": "MIT"},
    }


def _bad_repo() -> dict[str, Any]:
    """A repo that should FAIL multiple gates (few stars, GPL, archived, no desc)."""
    return {
        "full_name": "old/abandoned",
        "owner": {"login": "old"},
        "name": "abandoned",
        "html_url": "https://github.com/old/abandoned",
        "default_branch": "master",
        "description": "",
        "topics": [],
        "language": "Python",
        "size": 50,
        "stargazers_count": 3,
        "pushed_at": "2019-01-01T00:00:00Z",
        "archived": True,
        "fork": False,
        "license": {"spdx_id": "GPL-3.0"},
    }


# --------------------------------------------------------------------------- #
# normalize                                                                    #
# --------------------------------------------------------------------------- #


def test_normalize_repo_produces_schema_valid_record() -> None:
    record = normalize_repo(_good_repo(), head_sha="a" * 40, has_tests=True, has_ci=True)
    jsonschema.validate(instance=record, schema=RAW_HARVEST_SCHEMA)
    assert record["owner"] == "acme"
    assert record["repo"] == "ical-stats"
    assert record["head_sha"] == "a" * 40
    assert record["license_spdx"] == "MIT"
    assert record["redistributable"] is True
    assert record["domain_guess"] in {"cli-util", "data-analysis"}


def test_normalize_repo_handles_graphql_shape() -> None:
    """The normalizer also accepts the GraphQL-style nested fields."""
    gql = {
        "name": "tool",
        "owner": "octo",
        "url": "https://github.com/octo/tool",
        "description": "an api client sdk",
        "repositoryTopics": {"nodes": [{"topic": {"name": "api"}}, {"topic": {"name": "sdk"}}]},
        "primaryLanguage": {"name": "Python"},
        "stargazerCount": 99,
        "licenseInfo": {"spdxId": "Apache-2.0"},
    }
    record = normalize_repo(gql, head_sha="b" * 40)
    jsonschema.validate(instance=record, schema=RAW_HARVEST_SCHEMA)
    assert record["topics"] == ["api", "sdk"]
    assert record["primary_language"] == "Python"
    assert record["stars"] == 99
    assert record["license_spdx"] == "Apache-2.0"


def test_guess_domain_and_tier_proxy() -> None:
    assert guess_domain(_good_repo()) is not None
    assert tier_size_proxy(100) == "T1"
    assert tier_size_proxy(1000) == "T2"
    assert tier_size_proxy(50_000) == "T4"
    assert tier_size_proxy(None) is None


# --------------------------------------------------------------------------- #
# gates                                                                        #
# --------------------------------------------------------------------------- #


def test_good_repo_passes_gates() -> None:
    record = normalize_repo(_good_repo(), head_sha="a" * 40)
    passed, reasons = passes_quality_gates(record)
    assert passed is True, reasons
    assert reasons == []


def test_bad_repo_fails_with_reasons() -> None:
    record = normalize_repo(_bad_repo(), head_sha="c" * 40)
    passed, reasons = passes_quality_gates(record)
    assert passed is False
    joined = " ".join(reasons)
    assert "stars<" in joined
    assert "archived" in joined
    assert "non-permissive license" in joined
    assert "no description" in joined


def test_redistributable_allowlist() -> None:
    assert is_redistributable("MIT")
    assert is_redistributable("apache-2.0")  # case-insensitive
    assert not is_redistributable("GPL-3.0")
    assert not is_redistributable(None)


def test_custom_gate_config_threshold() -> None:
    record = normalize_repo(_good_repo(stars=120), head_sha="a" * 40)
    strict = GateConfig(min_stars=500)
    passed, reasons = passes_quality_gates(record, strict)
    assert passed is False
    assert any("stars<500" in r for r in reasons)


# --------------------------------------------------------------------------- #
# scrub (PII / secrets)                                                        #
# --------------------------------------------------------------------------- #


def test_scrub_text_redacts_email_and_token() -> None:
    text = "contact me at dev@example.com or use token ghp_abcdefghijklmnopqrstuvwxyz0123"
    out = scrub_text(text) or ""
    assert "dev@example.com" not in out
    assert "[REDACTED-EMAIL]" in out
    assert "ghp_abcdefghijklmnopqrstuvwxyz0123" not in out
    assert "[REDACTED-SECRET]" in out


def test_scrub_record_cleans_excerpt_and_issues() -> None:
    record = normalize_repo(_good_repo(), head_sha="a" * 40, readme="Reach maintainer@acme.io for help.")
    record["candidate_issues"] = [
        {"number": 1, "title": "bug from user@x.com", "labels": [], "url": "u", "body_excerpt": "email a@b.co"}
    ]
    scrubbed = scrub_record(record)
    assert "maintainer@acme.io" not in (scrubbed["readme_excerpt"] or "")
    assert "user@x.com" not in scrubbed["candidate_issues"][0]["title"]
    assert "a@b.co" not in scrubbed["candidate_issues"][0]["body_excerpt"]
    # The original record was not mutated.
    assert "maintainer@acme.io" in record["readme_excerpt"]


# --------------------------------------------------------------------------- #
# dedup (fallback path, no datasketch needed)                                  #
# --------------------------------------------------------------------------- #


def test_dedup_drops_exact_duplicates_and_clusters() -> None:
    base = normalize_repo(_good_repo(name="ical-stats", stars=500), head_sha="a" * 40,
                          readme="Summarize your calendar into a weekly time report. Parses ICS files.")
    exact_dup = normalize_repo(_good_repo(name="ical-stats", stars=10), head_sha="a" * 40,
                               readme="Summarize your calendar into a weekly time report. Parses ICS files.")
    near_dup = normalize_repo(_good_repo(name="ical-stats-fork", stars=20), head_sha="d" * 40,
                              readme="Summarize your calendar into a weekly time report. Parses ICS files.")
    unrelated = normalize_repo(
        {**_good_repo(name="weather-bot", stars=300), "description": "A weather forecast Slack bot",
         "topics": ["slack", "weather", "bot"]},
        head_sha="e" * 40,
        readme="A Slack bot that posts the daily weather forecast for your city each morning.",
    )

    deduped = dedup_records([base, exact_dup, near_dup, unrelated], threshold=0.6)
    # Exact owner/repo duplicate ('ical-stats' twice) is dropped first.
    keys = {f"{r['owner']}/{r['repo']}" for r in deduped}
    assert keys == {"acme/ical-stats", "acme/ical-stats-fork", "acme/weather-bot"}

    # Every survivor is annotated with a cluster id + representative flag.
    for r in deduped:
        assert r["dedup_cluster_id"] is not None
        assert isinstance(r["dedup_representative"], bool)

    # The two near-identical calendar repos share a cluster; the weather bot does not.
    by_key = {f"{r['owner']}/{r['repo']}": r for r in deduped}
    cal_cluster = by_key["acme/ical-stats"]["dedup_cluster_id"]
    fork_cluster = by_key["acme/ical-stats-fork"]["dedup_cluster_id"]
    weather_cluster = by_key["acme/weather-bot"]["dedup_cluster_id"]
    assert cal_cluster == fork_cluster
    assert weather_cluster != cal_cluster

    # Representative of the calendar cluster is the highest-starred member (base, 500).
    assert by_key["acme/ical-stats"]["dedup_representative"] is True
    assert by_key["acme/ical-stats-fork"]["dedup_representative"] is False


def test_minhash_available_is_boolean() -> None:
    """The lazy datasketch probe never raises (works with or without the dep)."""
    assert isinstance(minhash_available(), bool)
