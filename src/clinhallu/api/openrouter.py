"""Minimal OpenRouter client; raw completions are cached before validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterError(RuntimeError):
    pass


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_object(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if lines[0].strip().lower() not in ("```", "```json"):
            raise ValueError("Expected a JSON code fence")
        text = "\n".join(lines[1:-1])
    result = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(result, dict):
        raise ValueError("Expected one JSON object")
    return result


def build_request(model, messages, schema, name, settings):
    if not isinstance(model, str) or "/" not in model:
        raise ValueError("Use an explicit OpenRouter model ID such as openai/gpt-4.1-mini")
    raw_temperature = settings.get("temperature", 0)
    temperature = None if raw_temperature is None else float(raw_temperature)
    max_tokens = int(settings.get("max_tokens", 512))
    if (temperature is not None and not 0 <= temperature <= 2) or max_tokens <= 0:
        raise ValueError("temperature must be in [0,2] and max_tokens positive")
    request = dict(model=model, messages=messages, stream=False, max_tokens=max_tokens)
    # GPT-5-family and some reasoning endpoints reject custom temperature.
    # YAML `temperature: null` deliberately omits that request parameter.
    if temperature is not None:
        request["temperature"] = temperature
    mode = settings.get("response_format", "json_schema")
    if mode == "json_schema":
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": schema},
        }
    elif mode == "json_object":
        request["response_format"] = {"type": "json_object"}
    elif mode != "plain":
        raise ValueError("response_format must be json_schema, json_object or plain")
    provider = dict(settings.get("provider", {}))
    if mode != "plain":
        provider.setdefault("require_parameters", True)
    if provider:
        request["provider"] = provider
    for key in ("reasoning", "seed"):
        if key in settings:
            request[key] = settings[key]
    # Intentionally no n=5: all five answers are returned inside ONE completion.
    return request


class OpenRouterClient:
    def __init__(self, api_key=None, timeout=120, rate_limit_retries=0):
        self.api_key = api_key
        self.timeout = float(timeout)
        self.rate_limit_retries = int(rate_limit_retries)
        if self.timeout <= 0 or self.rate_limit_retries < 0:
            raise ValueError("Invalid timeout or retry count")
        self.requests_made = 0

    def complete(self, payload):
        key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise OpenRouterError("Set OPENROUTER_API_KEY in your environment first")
        request = urllib.request.Request(
            BASE_URL + "/chat/completions",
            data=json.dumps(payload, allow_nan=False).encode(),
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
                "X-Title": "ClinHallu",
            },
            method="POST",
        )
        for attempt in range(self.rate_limit_retries + 1):
            try:
                self.requests_made += 1
                with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                    response = json.load(handle)
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < self.rate_limit_retries:
                    try:
                        delay = float(exc.headers.get("Retry-After", 2))
                    except (TypeError, ValueError):
                        delay = 2
                    time.sleep(min(30, max(1, delay)))
                    continue
                message = exc.read().decode("utf-8", errors="replace").replace(key, "[REDACTED]")
                raise OpenRouterError(f"OpenRouter HTTP {exc.code}: {message[:1500]}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise OpenRouterError(
                    "Transport failed; request was not automatically retried because it may "
                    "already have incurred a charge: " + str(exc).replace(key, "[REDACTED]")
                ) from None
        if response.get("error"):
            raise OpenRouterError(str(response["error"]).replace(key, "[REDACTED]"))
        choices = response.get("choices", [])
        if len(choices) != 1:
            raise OpenRouterError("Expected exactly one completion")
        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return {
            "text": content,
            "refusal": message.get("refusal"),
            "finish_reason": choice.get("finish_reason"),
            "completion_id": response.get("id"),
            "model": response.get("model"),
            "provider": response.get("provider"),
            "usage": response.get("usage", {}),
            "created_at": response.get("created"),
        }

    def models(self):
        with urllib.request.urlopen(BASE_URL + "/models", timeout=self.timeout) as handle:
            return json.load(handle)["data"]


class ResponseCache:
    def __init__(self, directory):
        self.directory = Path(directory)

    def path(self, request):
        return self.directory / (fingerprint(request) + ".json")

    def contains(self, request):
        return self.path(request).is_file()

    def obtain(self, request, client):
        path = self.path(request)
        reused = path.is_file()
        if reused:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (
                saved.get("request_sha256") != fingerprint(request)
                or saved.get("request") != request
            ):
                raise ValueError(f"Cache request mismatch: {path}")
            response = saved["response"]
        else:
            response = client.complete(request)
            atomic_json(
                path,
                {"request_sha256": fingerprint(request), "request": request, "response": response},
            )
        try:
            refusal = response.get("refusal")
            finish_reason = response.get("finish_reason")
            if refusal or finish_reason in {
                "length",
                "content_filter",
                "error",
            }:
                completion_tokens = (response.get("usage") or {}).get("completion_tokens")
                details = (
                    f"finish_reason={finish_reason!r}, refusal={bool(refusal)}, "
                    f"completion_tokens={completion_tokens!r}"
                )
                if finish_reason == "length":
                    details += (
                        "; increase request.max_tokens or lower reasoning.effort, then use a "
                        "new output_dir"
                    )
                raise ValueError(f"Completion is unusable ({details})")
            parsed = parse_object(response["text"])
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(
                f"Invalid cached completion at {path}: {exc}. "
                "No automatic regeneration was attempted."
            ) from exc
        return parsed, response, reused


def usage_summary(responses):
    total = {
        "unique_completions": len(responses),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reported_cost_usd": 0.0,
        "missing_cost_count": 0,
    }
    for response in responses.values():
        usage = response.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens"):
            total[key] += int(usage.get(key) or 0)
        cost = usage.get("cost")
        if isinstance(cost, (float, int)) and math.isfinite(cost):
            total["reported_cost_usd"] += cost
        else:
            total["missing_cost_count"] += 1
    return total
