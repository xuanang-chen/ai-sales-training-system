from __future__ import annotations

from typing import Any


MET_SCORE_THRESHOLD = 3


def build_training_goal_summary(
    context: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    goal_status = context["goal_status"]
    summary: dict[str, list[dict[str, Any]]] = {
        "core_goals": [],
        "secondary_goals": [],
        "observation_goals": [],
    }

    for bucket, goals in context["training_goals"].items():
        summary.setdefault(bucket, [])
        for goal in goals:
            goal_id = goal["goal_id"]
            status_item = goal_status.get(goal_id, {})
            summary[bucket].append(
                {
                    "goal_id": goal_id,
                    "name": goal["name"],
                    "description": goal.get("description"),
                    "status": status_item.get("status", "untested"),
                    "best_score": status_item.get("best_score"),
                }
            )

    return summary


def build_goal_status_summary(context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary = {}
    for goal_id, status in context["goal_status"].items():
        summary[goal_id] = {
            "status": status.get("status", "untested"),
            "best_score": status.get("best_score"),
            "latest_score": status.get("latest_score"),
            "met_round": status.get("met_round"),
            "has_risk": status.get("has_risk", False),
            "best_reason": status.get("best_reason", ""),
        }

    return summary


def build_unmet_goal_summary(
    context: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    goal_status = context["goal_status"]
    unmet_goals: dict[str, list[dict[str, Any]]] = {
        "core_goals": [],
        "secondary_goals": [],
        "observation_goals": [],
    }

    for bucket, goals in context["training_goals"].items():
        for goal in goals:
            status_item = goal_status.get(goal["goal_id"], {})
            if is_goal_met(status_item):
                continue

            unmet_goals[bucket].append(
                {
                    "goal_id": goal["goal_id"],
                    "name": goal["name"],
                    "description": goal.get("description"),
                    "status": status_item.get("status", "untested"),
                    "best_score": status_item.get("best_score"),
                }
            )

    return unmet_goals


def build_goal_detail_map(context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    goal_map = {}
    for bucket, goals in context["training_goals"].items():
        for goal in goals:
            status_item = context["goal_status"].get(goal["goal_id"], {})
            goal_map[goal["goal_id"]] = {
                "goal_id": goal["goal_id"],
                "name": goal["name"],
                "description": goal.get("description"),
                "bucket": bucket,
                "status": status_item.get("status", "untested"),
                "best_score": status_item.get("best_score"),
            }

    return goal_map


def build_all_goal_items_for_judge(
    session: dict[str, Any],
    question_result: dict[str, Any],
) -> list[dict[str, Any]]:
    primary_goal_ids = set(question_result.get("target_goal_ids", []))
    secondary_goal_ids = set(question_result.get("secondary_probe_goal_ids", []))
    compliance_goal_ids = _compliance_goal_ids(session)
    selected_goal_ids = primary_goal_ids | secondary_goal_ids | compliance_goal_ids
    all_goals = []

    for bucket, goals in session["training_goals"].items():
        for goal in goals:
            goal_id = goal["goal_id"]
            if selected_goal_ids and goal_id not in selected_goal_ids:
                continue

            if goal_id in primary_goal_ids:
                evaluation_scope = "primary"
            elif goal_id in secondary_goal_ids:
                evaluation_scope = "secondary"
            elif goal_id in compliance_goal_ids:
                evaluation_scope = "compliance"
            else:
                evaluation_scope = "natural"

            status_item = session["goal_status"].get(goal_id, {})
            all_goals.append(
                {
                    "goal_id": goal_id,
                    "goal_name": goal["name"],
                    "description": goal.get("description"),
                    "bucket": bucket,
                    "evaluation_scope": evaluation_scope,
                    "current_status": status_item.get("status", "untested"),
                    "best_score": status_item.get("best_score"),
                }
            )

    return all_goals


def _compliance_goal_ids(session: dict[str, Any]) -> set[str]:
    policy = session.get("selected_product", {}).get("compliance_goal_policy", {})
    return {
        goal.get("goal_id")
        for goal in policy.get("goals", [])
        if goal.get("goal_id")
    }


def is_goal_available(context: dict[str, Any], goal_id: str) -> bool:
    return not is_goal_met(context["goal_status"].get(goal_id, {}))


def is_goal_met(goal_status: dict[str, Any]) -> bool:
    return goal_status.get("status") == "met" or score_or_zero(
        goal_status.get("best_score")
    ) >= MET_SCORE_THRESHOLD


def all_goals_met(context: dict[str, Any], bucket: str) -> bool:
    goals = context["training_goals"].get(bucket, [])
    if not goals:
        return True

    for goal in goals:
        goal_id = goal["goal_id"]
        if not is_goal_met(context["goal_status"].get(goal_id, {})):
            return False

    return True


def goal_matches_followup_category(
    goal_status: dict[str, Any],
    category: str,
) -> bool:
    status = goal_status.get("status", "untested")
    best_score = goal_status.get("best_score")

    if category == "partial":
        return status == "partial" or best_score == 2

    return status in {"untested", "not_observed", "not_met", "risk"} or best_score in {
        None,
        0,
        1,
    }


def score_or_zero(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def dedupe_preserve_order(items: list[str]) -> list[str]:
    deduped = []
    for item in items:
        if item not in deduped:
            deduped.append(item)

    return deduped
