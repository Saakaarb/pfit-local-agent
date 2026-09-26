"""Source provenance adapted from pfit-claude deployed_branch.

Hash complete source contents: comments also invalidate translation, and hashes
inside string literals cannot conceal a changed source. Legacy scripts retain
an explicitly qualified timestamp fallback in the readiness checker.
"""

import hashlib
from pathlib import Path

__all__ = ["normalized_hash", "build_stamp", "read_stamp", "verify_stamp",
           "write_stamp", "STAMP_PREFIX"]

STAMP_PREFIX = "# pfit-sources:"

# The two files /pfit-jax reads: the model it translates, and the config it takes
# the parameter order, integrator and step limit from.
SOURCES = ("user_model.py", "user_input.yaml")


def normalized_hash(path) -> str | None:
    """Short content hash of one source, or None when it is absent."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()[:16]


def build_stamp(session_dir) -> str:
    """The stamp line for a session's current sources."""
    session_dir = Path(session_dir)
    parts = []
    for name in SOURCES:
        location = ("inputs" if name.endswith(".yaml") else "generated")
        parts.append(f"{name}={normalized_hash(session_dir / location / name)}")
    return f"{STAMP_PREFIX} " + " ".join(parts)


def read_stamp(script_path) -> dict | None:
    """
    The stamp recorded in a generated script, or None when it carries none.

    Only the first few lines are scanned: the stamp is written at the top, and a
    hash-like string appearing later in the file should not be mistaken for one.
    """
    script_path = Path(script_path)
    if not script_path.is_file():
        return None
    with open(script_path, "r", encoding="utf-8") as handle:
        for _ in range(20):
            line = handle.readline()
            if not line:
                break
            if line.startswith(STAMP_PREFIX):
                fields = line[len(STAMP_PREFIX):].split()
                return dict(
                    field.split("=", 1) for field in fields if "=" in field)
    return None


def verify_stamp(session_dir) -> tuple[bool | None, str]:
    """
    Compare a script's stamp against the current sources.

    Returns `(None, reason)` when there is no stamp to compare -- an older
    script, or one written before stamping existed -- so the caller can fall
    back to modification times and say that it did.
    """
    session_dir = Path(session_dir)
    script = session_dir / "generated" / "generated_script.py"
    if not script.is_file():
        return False, "generated_script.py is missing"

    recorded = read_stamp(script)
    if recorded is None:
        return None, ("generated_script.py carries no source stamp, so its "
                      "agreement with the model cannot be verified by content")

    changed = []
    for name in SOURCES:
        location = "inputs" if name.endswith(".yaml") else "generated"
        current = normalized_hash(session_dir / location / name)
        if current is None or recorded.get(name) != current:
            changed.append(name)
    if changed:
        return False, (f"{sorted(changed)} changed since generated_script.py was "
                       f"written; the fit would use the previous translation")
    return True, "generated_script.py matches the sources it was generated from"


def write_stamp(session_dir) -> str:
    """
    Write or replace the stamp at the top of a session's generated script.

    Returns the stamp written. Idempotent: re-running replaces the existing line
    rather than accumulating them.
    """
    session_dir = Path(session_dir)
    script = session_dir / "generated" / "generated_script.py"
    if not script.is_file():
        raise FileNotFoundError(script)

    stamp = build_stamp(session_dir)
    lines = script.read_text().splitlines()
    kept = [line for line in lines if not line.startswith(STAMP_PREFIX)]

    # Preserve a leading shebang. Comments before a module docstring are valid.
    insert_at = 1 if kept and kept[0].startswith("#!") else 0
    kept.insert(insert_at, stamp)
    script.write_text("\n".join(kept) + "\n")
    return stamp
