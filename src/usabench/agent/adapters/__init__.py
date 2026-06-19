"""Agent adapters: thin wrappers conforming any agent to the :class:`Agent` API.

The harness drives *one* uniform interface (``reset`` / ``step`` / ``run``,
``docs/protocol.md`` §5.2). An adapter is the only framework-specific code: it
maps some agent implementation onto that interface and, crucially, onto the
one-action-per-step contract.

:class:`RawScaffoldAdapter` is the default, dependency-free adapter wrapping the
in-repo :class:`~usabench.agent.scaffold.ReActScaffold`. External-framework
adapters (SWE-agent-style, OpenHands-style, bare tool-calling) would live here
too; see :mod:`usabench.agent.adapters.raw_scaffold` for the unroll contract they
must honor.
"""

from __future__ import annotations

from usabench.agent.adapters.raw_scaffold import RawScaffoldAdapter, build_default_agent

__all__ = ["RawScaffoldAdapter", "build_default_agent"]
