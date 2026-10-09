from __future__ import annotations

from typing import Literal

Act = Literal["read", "search", "edit", "create", "delete", "inspect", "test", "build", "install", "run", "vcs",
              "network", "process", "shell", "delegate", "wait", "plan", "ask", "web", "skill", "notebook", "mcp",
              "other"]
"""What an action does. The first five are file tools of the harness; inspect through shell are the acts of the
shell component; the rest are the harness's other tools."""

Outcome = Literal["ok", "empty", "error", "rejected", "interrupted", "unanswered"]
"""How a call ended: a result; a result that holds nothing; a failure; a refusal by the user or a hook; an
interruption; or no result in the trajectory."""

ErrorClass = Literal["not_found", "mismatch", "precondition", "permission", "timeout", "test_failed", "build_failed",
                     "exit", "exception", "argument", "tool_missing", "denied", "sibling", "unknown"]
"""Why a call failed, as far as the result says: the object does not exist; the text to edit does not match; a
precondition of the tool is unmet, such as a file not read before an edit; a permission or sandbox refusal; a time
limit; failing tests; a failing build or check; a non-zero exit of another kind; an exception; arguments the tool
rejects; a program or tool that does not exist; a hook or policy that denied the call; a sibling call that failed;
or a failure the text does not explain."""
