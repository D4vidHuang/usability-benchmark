"""Agent-under-test protocol, the typed action it emits, and the reference scaffold.

The :class:`Agent` protocol and :class:`AgentAction` are the harness-facing
contract. The reference ReAct scaffold (:class:`ReActScaffold`), its sandboxed
toolset (:class:`SandboxToolset`), and the default adapter
(:class:`RawScaffoldAdapter`) are the in-repo, dependency-free agent-under-test.
"""

from __future__ import annotations

from usabench.agent.base import (
    ActionKind,
    Agent,
    AgentAction,
    Observation,
    OracleChannel,
    ToolHandle,
)
from usabench.agent.scaffold import (
    DEFAULT_SYSTEM_PROMPT,
    ReActScaffold,
    ScaffoldConfig,
    render_task_prompt,
)
from usabench.agent.tools import (
    DEFAULT_TOOL_NAMES,
    SandboxHandle,
    SandboxToolset,
    UserChannel,
    build_tool_specs,
)

__all__ = [
    # protocol surface
    "ActionKind",
    "Agent",
    "AgentAction",
    "Observation",
    "OracleChannel",
    "ToolHandle",
    # scaffold
    "ReActScaffold",
    "ScaffoldConfig",
    "DEFAULT_SYSTEM_PROMPT",
    "render_task_prompt",
    # tools
    "SandboxToolset",
    "SandboxHandle",
    "UserChannel",
    "build_tool_specs",
    "DEFAULT_TOOL_NAMES",
]
