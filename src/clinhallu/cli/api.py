"""Inspect current OpenRouter model IDs and supported parameters."""

import argparse
import json

from clinhallu.api.openrouter import OpenRouterClient


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("models",))
    parser.add_argument("--contains", default="")
    args = parser.parse_args(argv)
    entries = OpenRouterClient().models()
    print(
        json.dumps(
            [
                {
                    k: r.get(k)
                    for k in (
                        "id",
                        "name",
                        "supported_parameters",
                        "context_length",
                        "pricing",
                        "reasoning",
                    )
                }
                for r in entries
                if args.contains.lower() in r["id"].lower()
            ],
            indent=2,
        )
    )
