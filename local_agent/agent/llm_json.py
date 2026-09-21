import ast
import json
from typing import Any

from local_agent.agent.validators import ValidationError


def parse_llm_json_object(response: str, label: str) -> dict[str, Any]:
    text = _extract_json_text(response)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as json_exc:
        try:
            data = ast.literal_eval(text)
        except Exception:
            raise ValidationError(f"{label} returned invalid JSON: {json_exc}") from json_exc
    if not isinstance(data, dict):
        raise ValidationError(f"{label} response must be a JSON object")
    return data


def normalize_code_string(value: str) -> str:
    return (
        value.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "    ")
        .strip()
    )


def _extract_json_text(response: str) -> str:
    text = _strip_json_fence(response)
    if text.startswith("{") and text.endswith("}"):
        return text

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return text[index : index + end]
    return text


def _strip_json_fence(response: str) -> str:
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
