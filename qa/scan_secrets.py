# refuses to publish a file that contains a secret from .env
"""python -m qa.scan_secrets FILE [FILE ...]

Exits 1 (naming the KEYS and files, never the values) if any value from .env appears in any of
the given files. publish_pages.sh runs it over everything it is about to push: docs/ is served
publicly, and the logs go to a public repo.

.env is parsed with python-dotenv, so `export KEY=v`, spaces around `=`, single/double quotes,
trailing comments and CRLF line endings all resolve to the real value. Values under 8
characters are ignored (too short to be a secret, and they would match by accident).
"""

import os
import sys

from dotenv import dotenv_values

MIN_SECRET_LENGTH = 8
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def find_leaks(paths: list[str], env_path: str | None = None) -> list[tuple[str, str]]:
    """[(key name, file path), ...] for every secret value found in a file."""
    secrets = {k: v for k, v in dotenv_values(env_path or ENV_PATH).items()
               if v and len(v) >= MIN_SECRET_LENGTH}
    leaks = []
    for path in paths:
        with open(path, "rb") as f:
            data = f.read()
        for key, value in secrets.items():
            if value.encode() in data:
                leaks.append((key, path))
    return leaks


def main(argv: list[str] | None = None) -> int:
    paths = sys.argv[1:] if argv is None else argv
    if not os.path.exists(ENV_PATH):
        print(f"No .env at {ENV_PATH}: nothing to compare against, secret scan skipped.")
        return 0
    leaks = find_leaks(paths)
    if leaks:
        for key, path in leaks:
            print(f"REFUSING TO PUBLISH: {path} contains the value of {key} from .env.", file=sys.stderr)
        return 1
    print(f"Secret scan clean: {len(paths)} file(s) checked against .env.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
