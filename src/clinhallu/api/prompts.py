"""External prompts with literal JSON substitution, never Python format strings."""

import json

from clinhallu.config.paths import absolute_path


def read_prompts(config, root):
    prompts = {
        key: absolute_path(config[key], root).read_text(encoding="utf-8")
        for key in ("system", "user")
    }
    if "{{INPUT_JSON}}" not in prompts["user"]:
        raise ValueError("The user prompt must contain {{INPUT_JSON}}")
    return prompts


def messages_for(prompts, fields):
    return [
        {"role": "system", "content": prompts["system"]},
        {
            "role": "user",
            "content": prompts["user"].replace(
                "{{INPUT_JSON}}", json.dumps(fields, ensure_ascii=False)
            ),
        },
    ]
