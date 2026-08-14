"""Execution for the CodeAct student: notebook semantics, persistent namespace.

WHY THIS IS NOT `python file.py`. Measured on qwen3-1.7b over 30 sampled turns,
57% of the programs it writes contain no `print()` at all -- it writes notebook
style and ends on a bare expression. Run as a script, those produce NO output, so
33% of turns handed the teacher "(nothing)" and a program that had correctly
computed the answer scored 0. Real CodeAct runs an IPython kernel; echoing the
last top-level expression is what makes the paradigm work, not a detail. Running
each turn in a fresh interpreter also produced NameError crashes on variables the
student had defined two turns earlier, which a real kernel would never do.

So a cell here behaves like a notebook cell:
  * stdout is captured
  * if the last top-level statement is an expression and its value is not None,
    that value is echoed
  * names defined in earlier accepted cells are still bound

HOW THE NAMESPACE PERSISTS. Each call replays the session's earlier accepted
cells with their output muted, then runs the new cell captured. That is O(T^2)
work across an episode, which is nothing at T=5 turns of arithmetic, and it buys
statelessness: no long-lived interpreter per episode to supervise, kill, or leak
inside an async rollout worker serving many concurrent episodes.

ASYNC. Uses asyncio subprocesses. A blocking `subprocess.run` here would stall
the whole event loop, and the rollout runs dozens of episodes concurrently.

SANDBOX. `-I` (isolated), stdin closed, a temp cwd, and rlimits on CPU, address
space, file size and subprocesses. That bounds accidental damage from generated
math code; it is NOT a security boundary against hostile code. If this ever runs
anything untrusted, put it in a container.
"""

from __future__ import annotations

import ast
import asyncio
import os
import resource
import sys
import tempfile
from dataclasses import dataclass, field

# Wall-clock ceiling per cell. Generated math programs that take longer than this
# are almost always an unbounded loop rather than heavy computation.
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_CPU_S = 10
DEFAULT_MEMORY_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_FSIZE_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_CHARS = 2000

OK = "ok"
CRASH = "crash"
TIMEOUT = "timeout"

# Runs inside the subprocess. Replays history muted, then runs the new cell with
# the notebook last-expression rule.
_DRIVER = r'''
import ast, contextlib, io, sys, traceback

history = open(sys.argv[1], encoding="utf-8").read()
cell = open(sys.argv[2], encoding="utf-8").read()

globals_ns = {"__name__": "__main__"}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try:
        exec(compile(history, "<history>", "exec"), globals_ns)
    except Exception:
        # A cell that crashed earlier was never added to history, so this should
        # not fire; if it does, the new cell still gets whatever was bound.
        pass

buffer = io.StringIO()
status = "ok"
detail = ""
try:
    tree = ast.parse(cell)
    tail = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        tail = ast.Expression(tree.body[-1].value)
        tree.body = tree.body[:-1]
    with contextlib.redirect_stdout(buffer):
        exec(compile(tree, "<cell>", "exec"), globals_ns)
        if tail is not None:
            value = eval(compile(tail, "<cell>", "eval"), globals_ns)
            if value is not None:
                print(value if isinstance(value, str) else repr(value))
except BaseException:
    status = "crash"
    lines = traceback.format_exc().strip().splitlines()
    detail = lines[-1] if lines else "unknown error"

sys.stdout.write(buffer.getvalue())
if status != "ok":
    sys.stderr.write(detail)
    sys.exit(3)
'''


def _limits() -> None:  # pragma: no cover - runs in the child process
    resource.setrlimit(resource.RLIMIT_CPU, (DEFAULT_CPU_S, DEFAULT_CPU_S))
    resource.setrlimit(resource.RLIMIT_AS, (DEFAULT_MEMORY_BYTES, DEFAULT_MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (DEFAULT_FSIZE_BYTES, DEFAULT_FSIZE_BYTES))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except (ValueError, OSError):
        pass


def is_valid_python(src: str) -> bool:
    """Whether the reply parses at all.

    Needed because prefilling the reply with a ```python fence only constrains the
    FIRST token: by turn 2 the student writes prose inside the fence, which then
    executes as a SyntaxError. Callers regenerate on a False here.
    """
    try:
        ast.parse(src)
    except SyntaxError:
        return False
    return True


_COMPUTE_NODES = (
    ast.BinOp, ast.Compare, ast.For, ast.While, ast.comprehension,
    ast.If, ast.IfExp, ast.FunctionDef, ast.AsyncFunctionDef,
    ast.Import, ast.ImportFrom,
)


def is_constant_print(src: str) -> bool:
    """True when the program does no computation -- it prints a literal the model
    already worked out in its head, which makes the code channel cosmetic.

    Ran at 15-20% of turns in probes. Detectable, so it can be rejected rather
    than silently counted as the student having computed something.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, _COMPUTE_NODES):
            return False
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name != "print":
                return False
    return True


@dataclass
class CellResult:
    output: str
    status: str

    @property
    def ok(self) -> bool:
        return self.status == OK

    @property
    def silent(self) -> bool:
        """Ran fine and said nothing. The teacher learns nothing from this turn."""
        return self.status == OK and not self.output.strip()


@dataclass
class CodeSession:
    """One episode's interpreter. Accepted cells accumulate; crashed ones do not,
    so a broken turn cannot poison the names later turns rely on."""

    timeout_s: float = DEFAULT_TIMEOUT_S
    python: str = field(default_factory=lambda: sys.executable)
    _history: list[str] = field(default_factory=list)

    @property
    def turns_kept(self) -> int:
        return len(self._history)

    async def run(self, src: str) -> CellResult:
        if not src.strip():
            return CellResult("", CRASH)
        result = await self._exec(src)
        if result.ok:
            self._history.append(src)
        return result

    async def peek(self, src: str) -> CellResult:
        """Run without keeping it. Used for the solo re-test, which must not
        change the session the conversation built."""
        if not src.strip():
            return CellResult("", CRASH)
        return await self._exec(src)

    async def _exec(self, src: str) -> CellResult:
        with tempfile.TemporaryDirectory() as workdir:
            history_path = os.path.join(workdir, "_history.py")
            cell_path = os.path.join(workdir, "_cell.py")
            driver_path = os.path.join(workdir, "_driver.py")
            with open(history_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(self._history))
            with open(cell_path, "w", encoding="utf-8") as handle:
                handle.write(src)
            with open(driver_path, "w", encoding="utf-8") as handle:
                handle.write(_DRIVER)

            try:
                proc = await asyncio.create_subprocess_exec(
                    self.python, "-I", driver_path, history_path, cell_path,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=workdir,
                    preexec_fn=_limits,
                )
            except Exception:  # noqa: BLE001 - spawn failure is a crash, not a raise
                return CellResult("", CRASH)

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout_s
                )
            except asyncio.TimeoutError:
                for kill in (proc.terminate, proc.kill):
                    try:
                        kill()
                    except ProcessLookupError:
                        break
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                return CellResult("", TIMEOUT)

            out = (stdout or b"").decode("utf-8", "replace").strip()[:MAX_OUTPUT_CHARS]
            if proc.returncode != 0:
                err = (stderr or b"").decode("utf-8", "replace").strip()
                last = err.splitlines()[-1] if err.splitlines() else "unknown error"
                # Whatever the cell printed before dying is kept: a traceback plus
                # the partial output is what a real notebook would show, and it is
                # the most informative thing the teacher can be given.
                return CellResult((out + "\n" + last).strip()[:MAX_OUTPUT_CHARS], CRASH)
            return CellResult(out, OK)
