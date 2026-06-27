from __future__ import annotations

from typing import Any


def require_session_keys(
    session: dict[str, Any],
    required_keys: list[str],
    module_name: str,
) -> None:
    missing_keys = [key for key in required_keys if key not in session]
    if missing_keys:
        raise ValueError(f"session 缺少{module_name}需要的字段：{missing_keys}")


def compact_round_state(round_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_round": round_state.get("current_round"),
        "max_rounds": round_state.get("max_rounds"),
        "no_progress_count": round_state.get("no_progress_count"),
    }

