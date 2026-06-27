from __future__ import annotations

from typing import Any


MET_SCORE_THRESHOLD = 3


def decide_termination(session: dict[str, Any]) -> dict[str, Any]:
    round_state = session.get("round_state", {})
    current_round = _as_int(round_state.get("current_round"), default=0)
    max_rounds = _as_int(round_state.get("max_rounds"), default=3)
    summary = build_termination_summary(session)

    if current_round <= 0:
        return _continue_result(
            "尚未完成正式客户追问。",
            summary,
        )

    if current_round >= max_rounds:
        return _end_result(
            "max_rounds_reached",
            "本次训练已达到最大轮次，训练结束并进入复盘。",
            summary,
        )

    if summary["all_core_goals_met"]:
        return _decide_post_core_supplemental_phase(
            session,
            current_round,
            summary,
        )

    return _continue_result(
        "核心目标仍未全部达标，继续进入再次追问模块。",
        summary,
    )


def build_termination_summary(session: dict[str, Any]) -> dict[str, Any]:
    round_state = session.get("round_state", {})
    core_goal_ids = _goal_ids(session, "core_goals")
    secondary_goal_ids = _goal_ids(session, "secondary_goals")
    observation_goal_ids = _goal_ids(session, "observation_goals")
    met_core_goal_ids = _met_goal_ids(session, core_goal_ids)
    met_secondary_goal_ids = _met_goal_ids(session, secondary_goal_ids)
    met_observation_goal_ids = _met_goal_ids(session, observation_goal_ids)
    unmet_core_goal_ids = _unmet_goal_ids(session, core_goal_ids)
    unmet_secondary_goal_ids = _unmet_goal_ids(session, secondary_goal_ids)
    unmet_observation_goal_ids = _unmet_goal_ids(session, observation_goal_ids)
    all_core_goals_met = _all_goals_met(session, core_goal_ids)
    all_secondary_goals_met = _all_goals_met(session, secondary_goal_ids)
    available_post_core_goal_ids = _available_post_core_goal_ids(
        session,
        all_secondary_goals_met,
    )

    return {
        "current_round": _as_int(round_state.get("current_round"), default=0),
        "max_rounds": _as_int(round_state.get("max_rounds"), default=3),
        "no_progress_count": _as_int(round_state.get("no_progress_count"), default=0),
        "core_goal_ids": core_goal_ids,
        "secondary_goal_ids": secondary_goal_ids,
        "observation_goal_ids": observation_goal_ids,
        "met_core_goal_ids": met_core_goal_ids,
        "met_secondary_goal_ids": met_secondary_goal_ids,
        "met_observation_goal_ids": met_observation_goal_ids,
        "unmet_core_goal_ids": unmet_core_goal_ids,
        "unmet_secondary_goal_ids": unmet_secondary_goal_ids,
        "unmet_observation_goal_ids": unmet_observation_goal_ids,
        "available_post_core_goal_ids": available_post_core_goal_ids,
        "met_core_goal_count": len(met_core_goal_ids),
        "total_core_goal_count": len(core_goal_ids),
        "met_secondary_goal_count": len(met_secondary_goal_ids),
        "total_secondary_goal_count": len(secondary_goal_ids),
        "met_observation_goal_count": len(met_observation_goal_ids),
        "total_observation_goal_count": len(observation_goal_ids),
        "all_core_goals_met": all_core_goals_met,
        "all_secondary_goals_met": all_secondary_goals_met,
        "all_required_goals_met": _all_goals_met(
            session,
            core_goal_ids + secondary_goal_ids,
        ),
    }


def _decide_post_core_supplemental_phase(
    session: dict[str, Any],
    current_round: int,
    summary: dict[str, Any],
) -> dict[str, Any]:
    termination_state = session.setdefault("termination_state", {})

    if not summary["available_post_core_goal_ids"]:
        return _end_result(
            "supplemental_goals_completed",
            "核心目标已全部达标，且没有可继续补充测试的次级目标或观察目标，训练结束并进入复盘。",
            summary,
        )

    _grant_post_core_probe(termination_state, summary, current_round)
    return _continue_result(
        "核心目标已全部达标，但仍有可补充测试的次级目标或观察目标，继续进入再次追问模块。",
        summary,
    )


def _grant_post_core_probe(
    termination_state: dict[str, Any],
    summary: dict[str, Any],
    current_round: int,
) -> None:
    probe_count = _as_int(
        termination_state.get("post_core_probe_rounds_granted"),
        default=0,
    ) + 1
    next_round = current_round + 1
    probe_type = "observation" if summary["all_secondary_goals_met"] else "secondary"

    termination_state["post_core_supplemental_started"] = True
    termination_state["post_core_probe_rounds_granted"] = probe_count
    termination_state["post_core_last_probe_after_round"] = current_round
    termination_state["post_core_next_probe_round"] = next_round
    termination_state["post_core_next_probe_type"] = probe_type

    summary["post_core_supplemental_started"] = True
    summary["post_core_probe_rounds_granted"] = probe_count
    summary["post_core_next_probe_round"] = next_round
    summary["post_core_next_probe_type"] = probe_type


def _continue_result(reason: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "should_end": False,
        "end_type": "continue",
        "reason": reason,
        "next_stage": "continue_training",
        "next_module": "customer_simulator_next",
        "summary": summary,
    }


def _end_result(
    end_type: str,
    reason: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "should_end": True,
        "end_type": end_type,
        "reason": reason,
        "next_stage": "ended",
        "next_module": None,
        "summary": summary,
    }


def _goal_ids(session: dict[str, Any], bucket: str) -> list[str]:
    return [
        goal["goal_id"]
        for goal in session.get("training_goals", {}).get(bucket, [])
        if goal.get("goal_id")
    ]


def _met_goal_ids(session: dict[str, Any], goal_ids: list[str]) -> list[str]:
    return [
        goal_id
        for goal_id in goal_ids
        if _is_goal_met(session.get("goal_status", {}).get(goal_id, {}))
    ]


def _unmet_goal_ids(session: dict[str, Any], goal_ids: list[str]) -> list[str]:
    return [
        goal_id
        for goal_id in goal_ids
        if not _is_goal_met(session.get("goal_status", {}).get(goal_id, {}))
    ]


def _all_goals_met(session: dict[str, Any], goal_ids: list[str]) -> bool:
    if not goal_ids:
        return True

    return len(_met_goal_ids(session, goal_ids)) == len(goal_ids)


def _available_post_core_goal_ids(
    session: dict[str, Any],
    all_secondary_goals_met: bool,
) -> list[str]:
    if all_secondary_goals_met:
        return _unmet_goal_ids(
            session,
            _goal_ids(session, "observation_goals"),
        )

    return _unmet_goal_ids(
        session,
        _goal_ids(session, "secondary_goals"),
    )


def _is_goal_met(goal_status: dict[str, Any]) -> bool:
    return goal_status.get("status") == "met" or _as_int(
        goal_status.get("best_score"),
        default=0,
    ) >= MET_SCORE_THRESHOLD


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
