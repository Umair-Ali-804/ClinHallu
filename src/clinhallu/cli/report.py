"""Build publication tables and figures from completed runs."""

from __future__ import annotations

import argparse
import json

from clinhallu.config.paths import find_project_root
from clinhallu.reporting import build_publication_outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="build", choices=("build",))
    args = parser.parse_args(argv)
    del args
    print(json.dumps(build_publication_outputs(find_project_root()), indent=2))


if __name__ == "__main__":
    main()
