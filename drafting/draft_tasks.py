"""LLM-assisted task drafter: harvested candidate -> draft ``task.json`` (no gold).

Stage 2 of the dataset pipeline (``docs/tasks.md`` §4, §6.1). Given a normalized
``raw_harvest``/``candidates`` record, a *drafter LLM* is prompted with the README
excerpt + feature-list + enhancement issues and asked to emit a **draft** task: the
lay-phrased ``user_goal`` (an outcome, not an implementation), a tentative domain /
difficulty / capability set, and -- critically -- the ``ambiguity_points`` the agent
*should* surface by asking (``docs/tasks.md`` §3.4). The drafter is explicitly told
to omit >=2 high-severity decisions so the task is intervention-bearing (§1.3).

What the drafter does NOT author is the *gold*: acceptance criteria with passing
thresholds, the hidden-spec summary, reference-repo pins, and the frozen ``env/``
are written by a human-in-the-loop in stage 3 (``docs/tasks.md`` §6.2). To keep
the produced object schema-valid against ``schemas/task.schema.json`` (which
requires a ``hidden`` object with at least a ``summary``) while withholding gold,
the draft writes a *placeholder* ``hidden`` block tagged ``draft: true`` and the
ambiguity points (which are oracle-private but structurally part of ``hidden``)
with their ``gold`` left as an explicit ``TODO`` for the human author.

The module degrades gracefully: with no LLM client it falls back to a deterministic
heuristic drafter so the pipeline runs (and tests pass) without API keys. The
:class:`~usabench.llm.client.LLMClient` is the only model interface used; the
concrete provider clients are constructed elsewhere and injected.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from usabench.core.enums import DeliverableType, Difficulty
from usabench.core.errors import ProviderError, SchemaViolation
from usabench.llm.client import LLMClient, Message

__all__ = [
    "DraftConfig",
    "DraftResult",
    "DRAFTER_SYSTEM_PROMPT",
    "build_drafter_messages",
    "draft_from_candidate",
    "heuristic_draft",
    "draft_file",
    "iter_jsonl",
]

log = structlog.get_logger(__name__)

#: The six task domains (mirrors ``schemas/task.schema.json`` ``domain`` enum).
_DOMAINS: tuple[str, ...] = (
    "cli-util",
    "data-analysis",
    "web-dashboard",
    "api-integration",
    "automation",
    "dev-tooling",
)

#: Coarse domain -> default deliverable type for the heuristic fallback.
_DOMAIN_DELIVERABLE: dict[str, str] = {
    "cli-util": DeliverableType.CLI_TOOL.value,
    "data-analysis": DeliverableType.SCRIPT.value,
    "web-dashboard": DeliverableType.WEB_APP.value,
    "api-integration": DeliverableType.LIBRARY.value,
    "automation": DeliverableType.SCRIPT.value,
    "dev-tooling": DeliverableType.CLI_TOOL.value,
}

#: The instruction the drafter LLM obeys (data-side; the oracle persona is separate).
DRAFTER_SYSTEM_PROMPT = """\
You are a benchmark task author for a *usability* benchmark. You turn a real open-
source project into an UNDER-SPECIFIED, interactive coding task as a non-expert
user would phrase it. Rules:

1. Write `user_goal` as an OUTCOME a lay user wants ("I want a tool that tells
   me ..."), never an implementation. Do not name libraries, flags, or formats.
2. Deliberately OMIT at least TWO decisions a competent developer must make. Each
   omission becomes an `ambiguity_point` the agent should resolve by ASKING.
3. At least two ambiguity points MUST be severity "high" (load-bearing: guessing
   wrong should break the result).
4. Do NOT write gold answers, acceptance thresholds, or any reference to the
   source repo's exact API. You only propose the goal, the domain, a tentative
   difficulty (T1..T4), required capabilities, and the ambiguity points.
5. Output STRICT JSON only, matching the schema you are given. No prose outside
   the JSON object.
"""

#: The JSON shape requested from the drafter LLM (a subset of the task draft).
_DRAFTER_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["title", "user_goal", "domain", "difficulty", "ambiguity_points"],
    "properties": {
        "title": {"type": "string"},
        "user_goal": {"type": "string"},
        "user_goal_persona_note": {"type": "string"},
        "domain": {"type": "string", "enum": list(_DOMAINS)},
        "difficulty": {"type": "string", "enum": ["T1", "T2", "T3", "T4"]},
        "deliverable_type": {"type": "string"},
        "required_capabilities": {"type": "array", "items": {"type": "string"}},
        "ambiguity_points": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "question", "severity"],
                "properties": {
                    "id": {"type": "string"},
                    "question": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                },
            },
        },
        "hidden_summary_hint": {"type": "string"},
    },
}


@dataclass(slots=True)
class DraftConfig:
    """Tunables for the drafter.

    Attributes:
        model: Model id passed to the LLM client (recorded in provenance).
        temperature: Decoding temperature for the drafter.
        max_tokens: Max completion tokens for one draft.
        readme_chars: Max README characters fed into the prompt.
        max_issues: Max enhancement/feature issues fed into the prompt.
        min_high_severity: Minimum high-severity ambiguity points required.
        id_prefix: Slug prefix for generated task ids (``ub`` -> ``ub-<dom>-NNNN``).
    """

    model: str = "claude-3-5-sonnet"
    temperature: float = 0.4
    max_tokens: int = 1500
    readme_chars: int = 4096
    max_issues: int = 8
    min_high_severity: int = 2
    id_prefix: str = "ub"


@dataclass(slots=True)
class DraftResult:
    """The outcome of drafting one candidate.

    Attributes:
        task: The draft ``task.json`` dict (schema-valid; gold withheld/placeholder).
        used_llm: True if an LLM produced the draft; False for the heuristic fallback.
        warnings: Non-fatal issues (e.g. too few high-severity ambiguities).
        provenance: The drafter model id + source ``harvest_provenance_id``.
    """

    task: dict[str, Any]
    used_llm: bool = False
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Prompt construction                                                          #
# --------------------------------------------------------------------------- #


def _feature_lines(readme: str, *, limit: int = 12) -> list[str]:
    """Extract bullet/feature lines from a README to seed acceptance ideas."""
    lines: list[str] = []
    for raw in readme.splitlines():
        line = raw.strip()
        if re.match(r"^([-*+]|\d+[.)])\s+\S", line):
            cleaned = re.sub(r"^([-*+]|\d+[.)])\s+", "", line)
            if 3 <= len(cleaned) <= 200:
                lines.append(cleaned)
        if len(lines) >= limit:
            break
    return lines


def build_drafter_messages(
    candidate: dict[str, Any], config: DraftConfig | None = None
) -> list[Message]:
    """Build the chat messages for the drafter LLM from a candidate record.

    Args:
        candidate: A normalized ``raw_harvest``/``candidates`` record.
        config: Drafting tunables.

    Returns:
        A two-message list (system + user) ready for :meth:`LLMClient.chat`.
    """
    cfg = config or DraftConfig()
    readme = (candidate.get("readme_excerpt") or "")[: cfg.readme_chars]
    issues = candidate.get("candidate_issues") or []
    issue_lines = [
        f"- #{it.get('number')} {it.get('title')}"
        for it in issues[: cfg.max_issues]
        if it.get("title")
    ]
    features = _feature_lines(readme)

    payload = {
        "repo": f"{candidate.get('owner')}/{candidate.get('repo')}",
        "description": candidate.get("description"),
        "topics": candidate.get("topics") or [],
        "domain_guess": candidate.get("domain_guess"),
        "tier_size_proxy": candidate.get("tier_size_proxy"),
        "feature_lines": features,
        "enhancement_issues": issue_lines,
        "output_schema": _DRAFTER_OUTPUT_SCHEMA,
    }
    user = (
        "Draft an under-specified usability task from this open-source project. "
        "Use the README features and enhancement issues only as INSPIRATION for "
        "what a non-expert might ask for; do not reference the repo by name in the "
        "goal. Emit STRICT JSON matching `output_schema`.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return [
        Message(role="system", content=DRAFTER_SYSTEM_PROMPT),
        Message(role="user", content=user),
    ]


# --------------------------------------------------------------------------- #
# Draft assembly                                                               #
# --------------------------------------------------------------------------- #


def _slugify(text: str) -> str:
    """Lowercase, hyphenate, and strip a string into an id-safe slug fragment."""
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug or "task"


def _make_task_id(candidate: dict[str, Any], domain: str, *, prefix: str, seq: int) -> str:
    """Build a stable ``<prefix>-<domain-tag>-NNNN`` task id."""
    dom_tag = {
        "cli-util": "cli",
        "data-analysis": "data",
        "web-dashboard": "web",
        "api-integration": "api",
        "automation": "auto",
        "dev-tooling": "dev",
    }.get(domain, "gen")
    return f"{prefix}-{dom_tag}-{seq % 10000:04d}"


def _normalize_domain(value: str | None, fallback: str | None) -> str:
    """Coerce a free-form domain string onto the closed enum, with a fallback."""
    if value in _DOMAINS:
        return str(value)
    if fallback in _DOMAINS:
        return str(fallback)
    return "cli-util"


def _coerce_difficulty(value: str | None, fallback: str | None) -> str:
    """Coerce a difficulty string onto T1..T4."""
    for v in (value, fallback):
        if isinstance(v, str) and v.upper() in {d.value for d in Difficulty}:
            return v.upper()
    return Difficulty.T2.value


def _assemble_task(
    candidate: dict[str, Any],
    drafted: dict[str, Any],
    *,
    config: DraftConfig,
    seq: int,
) -> tuple[dict[str, Any], list[str]]:
    """Assemble a schema-valid draft ``task.json`` dict from drafter output.

    Gold fields are withheld: the ``hidden`` block is a placeholder carrying only a
    summary hint and the (gold-less) ambiguity points, tagged for human authoring.

    Returns:
        ``(task_dict, warnings)``.
    """
    warnings: list[str] = []
    domain = _normalize_domain(drafted.get("domain"), candidate.get("domain_guess"))
    difficulty = _coerce_difficulty(drafted.get("difficulty"), candidate.get("tier_size_proxy"))
    deliverable = drafted.get("deliverable_type") or _DOMAIN_DELIVERABLE.get(domain, "script")
    if deliverable not in {d.value for d in DeliverableType}:
        deliverable = _DOMAIN_DELIVERABLE.get(domain, "script")

    aps_in = drafted.get("ambiguity_points") or []
    ambiguity_points: list[dict[str, Any]] = []
    for i, ap in enumerate(aps_in):
        if not isinstance(ap, dict) or not ap.get("question"):
            continue
        sev = ap.get("severity") if ap.get("severity") in {"low", "medium", "high"} else "medium"
        ambiguity_points.append(
            {
                "id": str(ap.get("id") or f"AP{i + 1}"),
                "question": str(ap["question"]),
                # Gold is authored by a human in stage 3; keep an explicit marker.
                "gold": "TODO(author): resolve from reference repo behavior",
                "reveal": "on_ask",
                "severity": sev,
            }
        )

    n_high = sum(1 for ap in ambiguity_points if ap["severity"] == "high")
    if len(ambiguity_points) < 2:
        warnings.append(f"only {len(ambiguity_points)} ambiguity points (<2)")
    if n_high < config.min_high_severity:
        warnings.append(f"only {n_high} high-severity ambiguity points (<{config.min_high_severity})")

    task: dict[str, Any] = {
        "id": _make_task_id(candidate, domain, prefix=config.id_prefix, seq=seq),
        "schema_version": "1.0.0",
        "title": str(drafted.get("title") or candidate.get("repo") or "Untitled task"),
        "user_goal": str(drafted.get("user_goal") or "").strip(),
        "user_goal_persona_note": drafted.get("user_goal_persona_note")
        or "Non-expert; describes outcomes, not implementation.",
        "domain": domain,
        "difficulty": difficulty,
        "deliverable_type": deliverable,
        "required_capabilities": [
            str(c) for c in (drafted.get("required_capabilities") or []) if c
        ],
        "accept_threshold": 0.80,
        "contamination_label": None,
        "harvest_provenance_id": candidate.get("harvest_provenance_id"),
        # Reference-repo detail is oracle-private; the drafter records the *source*
        # link only (the human author pins the real reference set in stage 3).
        "reference_repos": [],
        "env": {
            "base_image": "python:3.11-slim",
            "network": "deny",
            "allowlist": [],
            "fixtures": [],
            "allowed_reqs": [],
            "setup": [],
            "entrypoint_hint": None,
        },
        "expected_interventions": None,
        # Placeholder hidden block: NOT gold. Tagged so QC/calibration treat it as a
        # draft. The human author fills acceptance_criteria + a real summary.
        "hidden": {
            "summary": str(drafted.get("hidden_summary_hint") or "TODO(author): write hidden spec"),
            "acceptance_criteria": [],
            "ambiguity_points": ambiguity_points,
            "info_units": [],
            "reveal_rules": {},
            "oracle_persona": "non_expert_user",
            "known_pitfalls": [],
            "out_of_scope": [],
            "constraints": [],
        },
    }
    if not task["user_goal"]:
        warnings.append("empty user_goal")
    return task, warnings


# --------------------------------------------------------------------------- #
# Heuristic fallback (no LLM)                                                   #
# --------------------------------------------------------------------------- #


def heuristic_draft(candidate: dict[str, Any], config: DraftConfig | None = None) -> dict[str, Any]:
    """Produce a deterministic draft *without* an LLM (offline fallback).

    Generates a plausible goal from the description plus two generic high-severity
    ambiguity points so the pipeline and tests run without API access. The output is
    intentionally conservative and flagged for human rewriting.

    Args:
        candidate: A normalized harvest/candidate record.
        config: Drafting tunables.

    Returns:
        A drafter-output dict (the same shape an LLM would return).
    """
    config or DraftConfig()
    domain = _normalize_domain(candidate.get("domain_guess"), None)
    desc = (candidate.get("description") or candidate.get("repo") or "a small tool").strip()
    desc = re.sub(r"\s+", " ", desc)[:160]
    readme = candidate.get("readme_excerpt") or ""
    caps_by_domain = {
        "cli-util": ["arg-parsing", "file-io", "error-handling/retries"],
        "data-analysis": ["data-parsing(csv/json/ics/xml)", "aggregation-stats", "file-io"],
        "web-dashboard": ["web-server", "frontend-render", "http-client"],
        "api-integration": ["http-client", "auth-handling", "error-handling/retries"],
        "automation": ["file-io", "error-handling/retries", "packaging/entrypoint"],
        "dev-tooling": ["arg-parsing", "file-io", "packaging/entrypoint"],
    }
    return {
        "title": f"{candidate.get('repo') or 'tool'} (drafted)",
        "user_goal": (
            f"I want a little tool that helps me with {desc.lower()}. "
            "Can you build something I can just run?"
        ),
        "user_goal_persona_note": "Non-expert; describes outcomes, not implementation.",
        "domain": domain,
        "difficulty": _coerce_difficulty(candidate.get("tier_size_proxy"), None),
        "deliverable_type": _DOMAIN_DELIVERABLE.get(domain, "script"),
        "required_capabilities": caps_by_domain.get(domain, ["file-io", "arg-parsing"]),
        "ambiguity_points": [
            {
                "id": "AP1",
                "question": "What exactly should the output contain and in what format?",
                "severity": "high",
            },
            {
                "id": "AP2",
                "question": "What is the expected input shape / where does the data come from?",
                "severity": "high",
            },
            {
                "id": "AP3",
                "question": "Are there edge cases (empty input, errors) it must handle?",
                "severity": "medium",
            },
        ],
        "hidden_summary_hint": (
            "TODO(author): the tool should " + (_feature_lines(readme)[:1] or ["do its core job"])[0]
        ),
    }


# --------------------------------------------------------------------------- #
# Top-level drafting                                                           #
# --------------------------------------------------------------------------- #


def _parse_llm_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response, tolerating fences."""
    stripped = text.strip()
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1)
    else:
        brace = stripped.find("{")
        if brace > 0:
            stripped = stripped[brace:]
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise SchemaViolation(f"drafter did not return valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise SchemaViolation("drafter JSON root is not an object")
    return obj


def draft_from_candidate(
    candidate: dict[str, Any],
    *,
    client: LLMClient | None = None,
    config: DraftConfig | None = None,
    seq: int = 0,
) -> DraftResult:
    """Draft one ``task.json`` from a harvested candidate record.

    Uses ``client`` (any :class:`LLMClient`) when provided; otherwise falls back to
    the deterministic :func:`heuristic_draft` so the pipeline runs offline. The
    returned task withholds gold (placeholder ``hidden`` block) but is structurally
    valid against ``schemas/task.schema.json``.

    Args:
        candidate: A normalized ``raw_harvest``/``candidates`` record.
        client: Optional LLM client for the drafter; ``None`` -> heuristic fallback.
        config: Drafting tunables.
        seq: A sequence number used to mint a stable task id.

    Returns:
        A :class:`DraftResult`.
    """
    cfg = config or DraftConfig()
    used_llm = False
    if client is not None:
        try:
            messages = build_drafter_messages(candidate, cfg)
            completion = client.chat(
                messages, temperature=cfg.temperature, max_tokens=cfg.max_tokens
            )
            drafted = _parse_llm_json(completion.text)
            used_llm = True
        except (ProviderError, SchemaViolation) as exc:
            log.warning(
                "drafting.llm_failed_fallback_heuristic",
                repo=f"{candidate.get('owner')}/{candidate.get('repo')}",
                error=str(exc),
            )
            drafted = heuristic_draft(candidate, cfg)
    else:
        drafted = heuristic_draft(candidate, cfg)

    task, warnings = _assemble_task(candidate, drafted, config=cfg, seq=seq)
    return DraftResult(
        task=task,
        used_llm=used_llm,
        warnings=warnings,
        provenance={
            "drafter_model": cfg.model if used_llm else "heuristic",
            "source_repo": f"{candidate.get('owner')}/{candidate.get('repo')}",
            "harvest_provenance_id": candidate.get("harvest_provenance_id"),
        },
    )


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield decoded JSON objects from a JSONL file (skipping blanks).

    Args:
        path: Path to a ``.jsonl`` file.

    Yields:
        One dict per non-empty line.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def draft_file(
    candidates_path: str | Path,
    out_path: str | Path,
    *,
    client: LLMClient | None = None,
    config: DraftConfig | None = None,
    limit: int | None = None,
) -> int:
    """Draft tasks for every candidate in a JSONL file, writing ``task_drafts.jsonl``.

    Args:
        candidates_path: Input ``candidates.jsonl`` (or ``raw_harvest.jsonl``).
        out_path: Output ``task_drafts.jsonl`` path.
        client: Optional LLM client (heuristic fallback when ``None``).
        config: Drafting tunables.
        limit: Optional cap on the number of candidates drafted.

    Returns:
        The number of drafts written.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as fh:
        for seq, candidate in enumerate(iter_jsonl(candidates_path)):
            if limit is not None and written >= limit:
                break
            # Skip candidates already marked rejected by the curator.
            if candidate.get("draft_status") == "rejected":
                continue
            result = draft_from_candidate(candidate, client=client, config=config, seq=seq)
            line = {
                "task": result.task,
                "_used_llm": result.used_llm,
                "_warnings": result.warnings,
                "_provenance": result.provenance,
            }
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            written += 1
    log.info("drafting.done", written=written, out=str(out))
    return written
