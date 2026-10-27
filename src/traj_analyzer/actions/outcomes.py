from __future__ import annotations

from traj_analyzer.actions.kinds import ErrorClass, Outcome
from traj_analyzer.harness.base import Harness

OPENING = 400
"""Characters of a result that are read: file contents and search hits quote error text further down."""
TIMEOUT_CODES = (124, 143, 137)
"""Exit codes of a command stopped by `timeout`, SIGTERM or SIGKILL."""
TOOL_MISSING_CODE = 127
PERMISSION_CODE = 126
FLAGGED: tuple[ErrorClass, ...] = ("precondition", "mismatch", "not_found", "permission", "timeout", "tool_missing",
                                   "argument", "exception")
"""The classes read off the opening of a flagged failure, in the order they are tried."""


def classify(
    result: str | None, act: str, is_error: bool | None, harness: Harness
) -> tuple[Outcome, ErrorClass | None]:
    """How the result reads under the harness's profile; only its opening counts, and its closing characters for an
    exit code the profile reads there.

    Under an empty profile a result is unanswered when missing, a failure of unknown class when the adapter
    flagged it, empty when it holds no text, and ok otherwise.
    """
    if result is None:
        return "unanswered", None
    text = result.strip()
    opening = text[:OPENING]
    marks = harness.error_marks
    if _has(opening, harness.rejected):
        return "rejected", None
    if _has(opening, harness.interrupted):
        return "interrupted", None
    if _has(opening, marks.get("denied", ())):
        return "error", "denied"
    if _has(opening, marks.get("sibling", ())):
        return "error", "sibling"
    exit_code = harness.exit_code.search(opening) if harness.exit_code is not None else None
    if exit_code is None and harness.closing_exit_code is not None:
        exit_code = harness.closing_exit_code.search(text[-OPENING:])
    if exit_code is not None and exit_code.group(1) != "0":
        return "error", _exit_class(int(exit_code.group(1)), text, act, harness)
    if _has(opening, harness.error_openings):
        return "error", _error_class(opening, act, harness)
    if act in ("edit", "create") and _has(opening, marks.get("mismatch", ())):
        return "error", "mismatch"
    if act in ("run", "test", "build") and _has(text, marks.get("exception", ())):
        return "error", "exception"
    if is_error:
        return "error", _error_class(opening, act, harness)
    if not text or opening.startswith(harness.empty):
        return "empty", None
    if act in ("read", "inspect") and _has(opening, marks.get("not_found", ())) and len(text) < 300:
        return "error", "not_found"
    return "ok", None


def _has(text: str, marks: tuple[str, ...]) -> bool:
    return any(mark in text for mark in marks)


def _exit_class(code: int, text: str, act: str, harness: Harness) -> ErrorClass:
    opening = text[:OPENING]
    marks = harness.error_marks
    if code in TIMEOUT_CODES or _has(opening, marks.get("timeout", ())):
        return "timeout"
    if code == TOOL_MISSING_CODE or _has(opening, marks.get("tool_missing", ())):
        return "tool_missing"
    if code == PERMISSION_CODE or _has(opening, marks.get("permission", ())):
        return "permission"
    if act == "test":
        return "test_failed"
    if act == "build":
        return "build_failed"
    if _has(text, marks.get("not_found", ())) and len(text) < 600:
        return "not_found"
    if _has(text, marks.get("exception", ())):
        return "exception"
    if _has(opening, marks.get("argument", ())):
        return "argument"
    return "exit"


def _error_class(opening: str, act: str, harness: Harness) -> ErrorClass:
    for label in FLAGGED:
        if _has(opening, harness.error_marks.get(label, ())):
            return label
    if act == "test" and _has(opening, harness.error_marks.get("test_failed", ())):
        return "test_failed"
    if act == "build":
        return "build_failed"
    return "unknown"
