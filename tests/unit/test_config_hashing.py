"""Config hashing + run_id determinism (``docs/infra.md`` §6.1).

Two numbers compare iff their manifests agree, so the ``config_hash`` -> ``run_id``
pipeline must be:

* **deterministic** -- same inputs -> same id, byte-for-byte;
* **order-insensitive** in dict keys (canonical JSON);
* **secret-invariant** -- redacting a rotated key must not change the hash;
* **sensitive** to anything outcome-affecting (config, task, seed, git sha).
"""

from __future__ import annotations

from usabench.config.hashing import canonical, config_hash, redact_secrets, to_hashable
from usabench.core import ids

# --------------------------------------------------------------------------- #
# canonical_json / config_hash                                                 #
# --------------------------------------------------------------------------- #


def test_config_hash_is_key_order_insensitive() -> None:
    a = {"model": "x", "temperature": 0.0, "budgets": {"turns": 80, "cost": 3.0}}
    b = {"budgets": {"cost": 3.0, "turns": 80}, "temperature": 0.0, "model": "x"}
    assert config_hash(a) == config_hash(b)


def test_config_hash_is_deterministic() -> None:
    cfg = {"model": "claude", "seed_matrix": [1, 2, 3]}
    assert config_hash(cfg) == config_hash(cfg)
    assert len(config_hash(cfg)) == 64


def test_config_hash_changes_with_content() -> None:
    base = {"model": "a", "temperature": 0.0}
    other = {"model": "b", "temperature": 0.0}
    assert config_hash(base) != config_hash(other)


def test_config_hash_is_secret_invariant() -> None:
    """Rotating an api_key must not change the hash (redaction-before-hash)."""
    with_key1 = {"model": "x", "api_key": "sk-aaaaaaaaaaaaaaaaaaaa"}
    with_key2 = {"model": "x", "api_key": "sk-bbbbbbbbbbbbbbbbbbbb"}
    assert config_hash(with_key1, redact=True) == config_hash(with_key2, redact=True)
    # The unredacted hashes differ (proving redaction is what makes them equal).
    assert config_hash(with_key1, redact=False) != config_hash(with_key2, redact=False)


def test_redact_secrets_masks_keys_and_values() -> None:
    cfg = to_hashable({"api_key": "sk-zzzzzzzzzzzzzzzzzzzz", "nested": {"token": "t"}})
    red = redact_secrets(cfg)
    assert red["api_key"] == "***REDACTED***"
    assert red["nested"]["token"] == "***REDACTED***"


def test_canonical_string_is_stable_and_sorted() -> None:
    s = canonical({"b": 1, "a": 2}, redact=False)
    # Canonical form sorts keys; 'a' precedes 'b'.
    assert s.index('"a"') < s.index('"b"')


# --------------------------------------------------------------------------- #
# run_id determinism                                                           #
# --------------------------------------------------------------------------- #


def test_run_id_is_deterministic_and_seed_sensitive() -> None:
    ch = ids.config_hash({"x": 1})
    r1 = ids.run_id(ch, "ub-cal-0007", 7, "abc123")
    r2 = ids.run_id(ch, "ub-cal-0007", 7, "abc123")
    r3 = ids.run_id(ch, "ub-cal-0007", 8, "abc123")  # different seed
    assert r1 == r2
    assert r1 != r3
    assert len(r1) == 64


def test_run_id_is_task_and_gitsha_sensitive() -> None:
    ch = ids.config_hash({"x": 1})
    base = ids.run_id(ch, "ub-cal-0007", 1, "sha-a")
    diff_task = ids.run_id(ch, "ub-cal-0008", 1, "sha-a")
    diff_sha = ids.run_id(ch, "ub-cal-0007", 1, "sha-b")
    diff_cfg = ids.run_id(ids.config_hash({"x": 2}), "ub-cal-0007", 1, "sha-a")
    assert len({base, diff_task, diff_sha, diff_cfg}) == 4


def test_build_manifest_run_id_matches_manual_pipeline() -> None:
    """A built manifest's run_id equals the manual config_hash -> run_id pipeline."""
    from usabench.harness.manifest import build_manifest

    cfg = {"model": "fake", "api_key": "sk-shouldnotaffecthash000"}
    manifest = build_manifest(
        task_id="ub-cal-0007",
        seed=3,
        config=cfg,
        package_version="0.1.0",
        git_sha_value="deadbeef",
    )
    expected_cfg_hash = config_hash(cfg, redact=True)
    expected_run_id = ids.run_id(expected_cfg_hash, "ub-cal-0007", 3, "deadbeef")
    assert manifest.config_hash == expected_cfg_hash
    assert manifest.run_id == expected_run_id


def test_build_manifest_redacts_secret_in_descriptors() -> None:
    """A raw API key in a descriptor never survives into the serialized manifest."""
    from usabench.harness.manifest import build_manifest

    manifest = build_manifest(
        task_id="t",
        seed=1,
        config={"run": "smoke"},
        package_version="0.1.0",
        git_sha_value="abc",
        agent={"model": "x", "api_key": "sk-shouldberedacted00000000"},
    )
    blob = manifest.model_dump_json()
    assert "sk-shouldberedacted" not in blob
