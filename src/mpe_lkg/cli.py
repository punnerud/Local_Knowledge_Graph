"""The ``mpe-lkg`` command."""

from __future__ import annotations

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="mpe-lkg",
        description="Local Knowledge Graph — a local LLM reasons step by step, "
                    "and the steps become a graph.",
        epilog="Needs Ollama running locally. Run 'mpe-lkg doctor' to check.",
    )
    parser.add_argument("--version", action="version", version=f"mpe-lkg {__version__}")
    parser.add_argument("--host", default=None, help="default 127.0.0.1")
    parser.add_argument("--port", type=int, default=None, help="default 5100")
    parser.add_argument("--debug", action="store_true", help="Flask debugger; not on a shared network")
    parser.add_argument(
        "command", nargs="?", default="serve", choices=["serve", "doctor", "battery"],
        help="serve (default) starts the web app; doctor reports what is missing; "
             "battery runs the question battery against the installed model",
    )
    return parser


def doctor() -> int:
    """Say exactly what is wrong and what command fixes it."""
    from . import backends

    status = backends.health(backends.DEFAULT_BASE_URL)
    print(json.dumps(status, indent=2))
    if status["ok"]:
        print("\nEverything the app needs is present.")
        return 0
    print(f"\n{status['problem']}\n{status['hint']}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    # The battery has its own flags (--seed, --domains, ...), so it takes the
    # rest of the line rather than sharing the server's parser.
    if argv[:1] == ["battery"]:
        from .battery.bench import main as battery_main

        return battery_main(argv[1:])

    args = build_parser().parse_args(argv)

    if args.command == "doctor":
        return doctor()

    from .app import run

    run(host=args.host, port=args.port, debug=args.debug or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
