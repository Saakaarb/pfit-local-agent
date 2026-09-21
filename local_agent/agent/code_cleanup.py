import re


_FENCED_CODE_RE = re.compile(r"^\s*```(?:python|py)?\s*\n(?P<code>.*)\n```\s*$", re.DOTALL)


def clean_llm_code_response(response: str) -> str:
    match = _FENCED_CODE_RE.match(response)
    if match:
        return match.group("code")
    return response
