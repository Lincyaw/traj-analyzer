from __future__ import annotations

import json
import sys
from dataclasses import asdict

from traj_analyzer.shell.parsing import parse_here


def main() -> None:
    """Read one JSON line `{"text": ...}` at a time and answer with the commands of the text, or with an error.

    The process answers from stdin to stdout until stdin closes; `parsing.parse` starts it and restarts it after an
    error, since the dash parser keeps state a rejected text corrupts.
    """
    for line in sys.stdin:
        request = json.loads(line)
        try:
            script = parse_here(request["text"])
        except Exception as error:
            answer = {"error": f"{type(error).__name__}: {error}"}
        else:
            answer = {"commands": [asdict(command) for command in script.commands]}
        sys.stdout.write(json.dumps(answer, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
