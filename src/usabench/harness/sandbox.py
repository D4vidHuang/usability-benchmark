"""Sandbox backends: the agent's isolated file/exec workspace.

A sandbox is one per run. It exposes a small surface -- ``write``, ``read``,
``exec``, ``snapshot``, ``diff`` -- behind the :class:`SandboxBackend` ABC. The
harness *derives* ``file_edit`` events by snapshotting and diffing the workspace
(it never trusts the agent's claim of what it wrote -- ``docs/protocol.md`` §5.3).

Two backends ship here:

* :class:`LocalSubprocessSandbox` -- a real, working backend for the Mac / DAIC
  login node: a per-run temp workspace, subprocess exec with timeouts, a
  best-effort network-deny environment, and path-escape protection. This is what
  makes the smoke path run end-to-end today.
* :class:`ApptainerSandbox` -- a stub for DAIC compute nodes (Apptainer 1.5.0).
  It builds the right ``apptainer exec`` command line but, until validated on the
  cluster, raises a clear :class:`SandboxError` on ``exec`` so it never silently
  pretends to isolate. File ops fall back to the host workspace so snapshots work.

Heavy/optional backends (docker) are intentionally NOT imported at module load.
"""

from __future__ import annotations

import abc
import difflib
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from usabench.core.enums import NetworkPolicy
from usabench.core.errors import SandboxError
from usabench.core.ids import sha256_hex
from usabench.core.schema import FileEdit
from usabench.logging_setup import get_logger

__all__ = [
    "ExecResult",
    "WorkspaceSnapshot",
    "SandboxBackend",
    "LocalSubprocessSandbox",
    "ApptainerSandbox",
]

_log = get_logger(__name__)

#: Files larger than this are hashed but not loaded for line-diff stats (bytes).
_MAX_DIFF_BYTES = 1_000_000
#: stdout/stderr captured beyond this is truncated in the returned result (bytes).
_MAX_CAPTURE_BYTES = 64_000


@dataclass(frozen=True)
class ExecResult:
    """The outcome of one sandboxed command execution.

    Attributes:
        exit_code: Process exit code (124 reserved for timeout).
        stdout: Captured stdout (already truncated to a safe size).
        stderr: Captured stderr (already truncated to a safe size).
        wall_ms: Wall-clock duration in milliseconds.
        timed_out: True if the command hit its timeout.
        truncated: True if stdout/stderr was truncated.
    """

    exit_code: int
    stdout: str
    stderr: str
    wall_ms: int
    timed_out: bool = False
    truncated: bool = False


@dataclass
class WorkspaceSnapshot:
    """A content snapshot of the workspace: relative path -> sha256 hex.

    Also retains small file contents so the harness can compute added/removed line
    counts for ``file_edit`` events without re-reading the disk after a mutation.

    Attributes:
        hashes: Mapping of POSIX-relative path -> sha256 hex of file bytes.
        contents: Mapping of path -> decoded text for small files (best effort).
        loc: Mapping of path -> line count for small files.
    """

    hashes: dict[str, str] = field(default_factory=dict)
    contents: dict[str, str] = field(default_factory=dict)
    loc: dict[str, int] = field(default_factory=dict)


class SandboxBackend(abc.ABC):
    """Abstract base for sandbox backends (``docs/protocol.md`` §5.2).

    Subclasses provide isolation; this base provides the snapshot/diff machinery
    that derives :class:`FileEdit` events, which is identical across backends as
    long as the workspace is a real directory on a shared filesystem.
    """

    #: Absolute path to the workspace root on the host filesystem.
    workspace: Path
    #: The network policy this sandbox enforces (recorded in the trace).
    network: NetworkPolicy

    # --- lifecycle ---------------------------------------------------------- #

    @abc.abstractmethod
    def setup(self) -> None:
        """Create the workspace and apply any per-task setup. Idempotent."""

    @abc.abstractmethod
    def teardown(self) -> None:
        """Remove the workspace and release resources. Idempotent."""

    # --- file ops ----------------------------------------------------------- #

    @abc.abstractmethod
    def exec(self, cmd: str, *, timeout_s: int = 120) -> ExecResult:
        """Execute ``cmd`` (a shell string) inside the sandbox under a timeout."""

    def _resolve(self, path: str) -> Path:
        """Resolve ``path`` inside the workspace, denying escapes via ``..``.

        Args:
            path: A workspace-relative (or absolute-within-workspace) path.

        Returns:
            The absolute resolved path, guaranteed under :attr:`workspace`.

        Raises:
            SandboxError: If the resolved path escapes the workspace root.
        """
        root = self.workspace.resolve()
        candidate = (root / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise SandboxError(f"path escapes workspace: {path!r}") from exc
        return candidate

    def write(self, path: str, content: str) -> None:
        """Write ``content`` (UTF-8) to ``path`` inside the workspace.

        Args:
            path: Workspace-relative target path.
            content: Text to write (parent dirs are created).

        Raises:
            SandboxError: On path escape or write failure.
        """
        target = self._resolve(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise SandboxError(f"write failed for {path!r}: {exc}") from exc

    def read(self, path: str) -> str:
        """Read and return the text of ``path`` inside the workspace.

        Args:
            path: Workspace-relative source path.

        Returns:
            The decoded file content (errors replaced).

        Raises:
            SandboxError: On path escape or if the file is missing.
        """
        target = self._resolve(path)
        if not target.is_file():
            raise SandboxError(f"no such file: {path!r}")
        return target.read_text(encoding="utf-8", errors="replace")

    # --- snapshot / diff ---------------------------------------------------- #

    def snapshot(self) -> WorkspaceSnapshot:
        """Snapshot the workspace into a content-addressed :class:`WorkspaceSnapshot`.

        Walks the workspace, hashing every regular file (skipping VCS / pycache /
        venv noise). Small text files also have their content + line count cached so
        :meth:`diff` can compute added/removed lines.

        Returns:
            A :class:`WorkspaceSnapshot` of the current workspace state.
        """
        snap = WorkspaceSnapshot()
        root = self.workspace.resolve()
        if not root.is_dir():
            return snap
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for name in filenames:
                abs_path = Path(dirpath) / name
                if abs_path.is_symlink() or not abs_path.is_file():
                    continue
                rel = abs_path.relative_to(root).as_posix()
                try:
                    data = abs_path.read_bytes()
                except OSError:
                    continue
                snap.hashes[rel] = sha256_hex(data)
                if len(data) <= _MAX_DIFF_BYTES:
                    text = data.decode("utf-8", errors="replace")
                    snap.contents[rel] = text
                    snap.loc[rel] = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        return snap

    def diff(self, before: WorkspaceSnapshot, after: WorkspaceSnapshot) -> list[FileEdit]:
        """Derive :class:`FileEdit` events from two snapshots.

        This is how the harness produces authoritative file-mutation events: by
        comparing content hashes, never by trusting the agent. Renames are not
        inferred (a moved file shows as a delete + create) to keep the derivation
        deterministic and side-effect-free.

        Args:
            before: Snapshot taken before the action.
            after: Snapshot taken after the action.

        Returns:
            One :class:`FileEdit` per created/modified/deleted path, sorted by path.
        """
        edits: list[FileEdit] = []
        before_keys = set(before.hashes)
        after_keys = set(after.hashes)

        for path in sorted(after_keys - before_keys):
            edits.append(self._make_edit(path, "create", before, after))
        for path in sorted(before_keys - after_keys):
            edits.append(self._make_edit(path, "delete", before, after))
        for path in sorted(before_keys & after_keys):
            if before.hashes[path] != after.hashes[path]:
                edits.append(self._make_edit(path, "modify", before, after))
        return edits

    @staticmethod
    def _make_edit(
        path: str, op: str, before: WorkspaceSnapshot, after: WorkspaceSnapshot
    ) -> FileEdit:
        """Build one :class:`FileEdit`, computing added/removed lines + diff hash."""
        pre_text = before.contents.get(path, "")
        post_text = after.contents.get(path, "")
        added = removed = 0
        diff_hash: str | None = None
        if pre_text != post_text:
            diff_lines = list(
                difflib.unified_diff(
                    pre_text.splitlines(keepends=True),
                    post_text.splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                    n=0,
                )
            )
            for line in diff_lines:
                if line.startswith("+") and not line.startswith("+++"):
                    added += 1
                elif line.startswith("-") and not line.startswith("---"):
                    removed += 1
            diff_hash = sha256_hex("".join(diff_lines)) if diff_lines else None
        return FileEdit(
            path=path,
            op=op,
            pre_sha256=before.hashes.get(path),
            post_sha256=after.hashes.get(path),
            added=added,
            removed=removed,
            unified_diff_sha256=diff_hash,
            loc_after=after.loc.get(path, 0),
        )

    # --- context manager sugar --------------------------------------------- #

    def __enter__(self) -> SandboxBackend:
        self.setup()
        return self

    def __exit__(self, *exc: object) -> None:
        self.teardown()


#: Directories never walked when snapshotting (VCS / build / env noise).
_SKIP_DIRS: frozenset[str] = frozenset(
    {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache", ".tox"}
)


class LocalSubprocessSandbox(SandboxBackend):
    """A real, working local sandbox using a temp workspace + subprocess exec.

    Isolation here is *best effort* (no namespaces/cgroups): it runs commands with
    ``cwd`` pinned to the workspace, a wall-clock timeout, captured + truncated
    output, and -- when ``network`` is :attr:`NetworkPolicy.DENY` -- a scrubbed
    environment plus proxy variables pointed at a black hole to discourage egress.
    It is sufficient for the deterministic smoke path and CI; true isolation is the
    job of :class:`ApptainerSandbox` / a future Docker backend on real runs.

    Example:
        >>> with LocalSubprocessSandbox(task_id="t") as sb:
        ...     sb.write("hello.txt", "hi")
        ...     sb.read("hello.txt")
        'hi'
    """

    def __init__(
        self,
        *,
        task_id: str = "task",
        run_id: str | None = None,
        root: str | Path | None = None,
        network: NetworkPolicy = NetworkPolicy.DENY,
        allowlist: list[str] | None = None,
        env_passthrough: list[str] | None = None,
        setup_cmds: list[str] | None = None,
    ) -> None:
        """Initialize a local sandbox.

        Args:
            task_id: Task id (used in the workspace dir name).
            run_id: Optional run id (used in the workspace dir name).
            root: Optional parent dir for the workspace; defaults to a temp dir.
            network: Network policy to enforce best-effort.
            allowlist: Hosts allowed when ``network == allowlist`` (recorded; not
                enforced at the packet level in this backend).
            env_passthrough: Extra environment variable names to pass through.
            setup_cmds: Commands run once during :meth:`setup` (e.g. fixture prep).
        """
        self.task_id = task_id
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.network = network
        self.allowlist = list(allowlist or [])
        self.env_passthrough = list(env_passthrough or [])
        self.setup_cmds = list(setup_cmds or [])
        self._root = Path(root) if root is not None else None
        self._owns_root = root is None
        self.workspace = Path()  # set in setup()
        self._is_setup = False

    def setup(self) -> None:
        """Create the workspace directory and run any setup commands."""
        if self._is_setup:
            return
        if self._root is None:
            base = Path(tempfile.mkdtemp(prefix="usabench-ws-"))
            self._root = base
            self.workspace = base
        else:
            self.workspace = self._root / f"{self.task_id}-{self.run_id}"
            self.workspace.mkdir(parents=True, exist_ok=True)
        self._is_setup = True
        for cmd in self.setup_cmds:
            res = self.exec(cmd, timeout_s=600)
            if res.exit_code != 0:
                _log.warning("sandbox.setup_cmd_failed", cmd=cmd, exit_code=res.exit_code)

    def teardown(self) -> None:
        """Remove the workspace if this sandbox created it."""
        if not self._is_setup:
            return
        if self._owns_root and self.workspace and self.workspace.exists():
            shutil.rmtree(self.workspace, ignore_errors=True)
        self._is_setup = False

    def _build_env(self) -> dict[str, str]:
        """Build the subprocess environment, scrubbing network egress on deny."""
        env: dict[str, str] = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.workspace),
            "TMPDIR": str(self.workspace),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        for name in self.env_passthrough:
            if name in os.environ:
                env[name] = os.environ[name]
        if self.network == NetworkPolicy.DENY:
            # Best-effort: point proxies at a closed port so naive HTTP egress fails
            # fast rather than reaching the network. True deny is the container's job.
            env["http_proxy"] = "http://127.0.0.1:9"
            env["https_proxy"] = "http://127.0.0.1:9"
            env["HTTP_PROXY"] = "http://127.0.0.1:9"
            env["HTTPS_PROXY"] = "http://127.0.0.1:9"
            env["no_proxy"] = ""
        return env

    def exec(self, cmd: str, *, timeout_s: int = 120) -> ExecResult:
        """Run ``cmd`` in the workspace via the shell under a timeout.

        Args:
            cmd: A shell command string.
            timeout_s: Wall-clock timeout in seconds.

        Returns:
            An :class:`ExecResult` with captured, truncated output.

        Raises:
            SandboxError: If the sandbox is not set up.
        """
        if not self._is_setup:
            raise SandboxError("sandbox.exec called before setup()")
        import time as _time

        start = _time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(  # noqa: S602 - intentional shell exec inside sandbox
                cmd,
                shell=True,
                cwd=str(self.workspace),
                env=self._build_env(),
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
            exit_code = proc.returncode
            stdout_b, stderr_b = proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124
            stdout_b = exc.stdout or b""
            stderr_b = (exc.stderr or b"") + b"\n[usabench] command timed out"
        wall_ms = int((_time.monotonic() - start) * 1000)
        stdout, t1 = _truncate(stdout_b)
        stderr, t2 = _truncate(stderr_b)
        return ExecResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            wall_ms=wall_ms,
            timed_out=timed_out,
            truncated=t1 or t2,
        )


def _truncate(data: bytes, limit: int = _MAX_CAPTURE_BYTES) -> tuple[str, bool]:
    """Decode and truncate captured bytes to ``limit``; return (text, truncated)."""
    truncated = len(data) > limit
    if truncated:
        data = data[:limit]
    return data.decode("utf-8", errors="replace"), truncated


class ApptainerSandbox(SandboxBackend):
    """DAIC compute-node backend stub built on Apptainer 1.5.0 (rootless, SLURM).

    File ops use the shared host workspace (so snapshot/diff work identically to
    the local backend). ``exec`` constructs the correct ``apptainer exec`` command
    line -- binding the workspace to ``/work`` with ``--no-home`` and ``--net
    --network none`` for the hermetic default -- but, because this has not yet been
    validated on the cluster, it raises :class:`SandboxError` unless explicitly
    marked ``validated=True``. This avoids silently running unisolated commands on
    a node that the design promises is hermetic.

    TODO(daic): validate the constructed command on a DAIC compute node with a
    pinned SIF image, then flip the default ``validated`` to True.
    """

    def __init__(
        self,
        *,
        sif_image: str,
        task_id: str = "task",
        run_id: str | None = None,
        root: str | Path | None = None,
        network: NetworkPolicy = NetworkPolicy.DENY,
        allowlist: list[str] | None = None,
        validated: bool = False,
        apptainer_bin: str = "apptainer",
    ) -> None:
        """Initialize the Apptainer backend.

        Args:
            sif_image: Path to the pinned ``.sif`` image.
            task_id: Task id (workspace dir name).
            run_id: Optional run id (workspace dir name).
            root: Parent dir for the workspace (project storage on DAIC).
            network: Network policy; ``deny`` maps to ``--network none``.
            allowlist: Allowed hosts when ``network == allowlist`` (recorded).
            validated: Set True only after validating exec on a real DAIC node.
            apptainer_bin: The apptainer executable name.
        """
        self.sif_image = sif_image
        self.task_id = task_id
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.network = network
        self.allowlist = list(allowlist or [])
        self.validated = validated
        self.apptainer_bin = apptainer_bin
        self._root = Path(root) if root is not None else None
        self.workspace = Path()
        self._is_setup = False

    def setup(self) -> None:
        """Create the host workspace bound into the container at ``/work``."""
        if self._is_setup:
            return
        if self._root is None:
            self.workspace = Path(tempfile.mkdtemp(prefix="usabench-apptainer-"))
        else:
            self.workspace = self._root / f"{self.task_id}-{self.run_id}"
            self.workspace.mkdir(parents=True, exist_ok=True)
        self._is_setup = True

    def teardown(self) -> None:
        """No-op cleanup (project storage workspaces are reaped by the batch layer)."""
        self._is_setup = False

    def build_command(self, cmd: str) -> list[str]:
        """Construct the ``apptainer exec`` argv for ``cmd`` (no execution).

        Exposed so the DAIC integration tests can assert the exact isolation flags
        without needing apptainer installed.

        Args:
            cmd: The shell command to run inside the container.

        Returns:
            The argv list for :func:`subprocess.run`.
        """
        net_flags = ["--net", "--network", "none"] if self.network == NetworkPolicy.DENY else []
        return [
            self.apptainer_bin,
            "exec",
            "--containall",
            "--no-home",
            "--writable-tmpfs",
            "--bind",
            f"{self.workspace}:/work",
            "--pwd",
            "/work",
            *net_flags,
            self.sif_image,
            "/bin/sh",
            "-c",
            cmd,
        ]

    def exec(self, cmd: str, *, timeout_s: int = 120) -> ExecResult:
        """Run ``cmd`` via apptainer (guarded until validated on DAIC).

        Raises:
            SandboxError: Always, unless :attr:`validated` is True (the constructed
                command is still available via :meth:`build_command`).
        """
        if not self._is_setup:
            raise SandboxError("sandbox.exec called before setup()")
        if not self.validated:
            raise SandboxError(
                "ApptainerSandbox.exec is not yet validated on DAIC; "
                "use build_command() to inspect the argv, or set validated=True "
                "once verified on a compute node."
            )
        import time as _time

        argv = self.build_command(cmd)
        start = _time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(  # noqa: S603 - argv list, no shell
                argv,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
            exit_code = proc.returncode
            stdout_b, stderr_b = proc.stdout, proc.stderr
        except FileNotFoundError as exc:
            raise SandboxError(f"apptainer not found: {self.apptainer_bin!r}") from exc
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124
            stdout_b = exc.stdout or b""
            stderr_b = exc.stderr or b""
        wall_ms = int((_time.monotonic() - start) * 1000)
        stdout, t1 = _truncate(stdout_b)
        stderr, t2 = _truncate(stderr_b)
        return ExecResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            wall_ms=wall_ms,
            timed_out=timed_out,
            truncated=t1 or t2,
        )
