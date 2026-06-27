from __future__ import annotations

from typing import Any

from core.prompt_context.common import compact_round_state, require_session_keys
from core.prompt_context.customer_context import build_customer_brief
from core.prompt_context.goal_context import (
    all_goals_met,
    build_goal_detail_map,
    build_goal_status_summary,
    build_training_goal_summary,
    build_unmet_goal_summary,
    dedupe_preserve_order,
    goal_matches_followup_category,
    is_goal_met,
)
from core.prompt_context.product_context import build_product_for_customer_question
from core.prompt_context.turn_context import build_last_turn_summary


PRODUCT_INTRODUCTION_GOAL_ID = "product_basic_introduction"


def build_next_customer_simulator_context(session: dict[str, Any]) -> dict[str, Any]:
    required_keys = [
        "selected_product",
        "selected_customer",
        "customer_preference_profile",
        "training_goals",
        "goal_status",
        "round_state",
        "question_logs",
    ]
    require_session_keys(session, required_keys, "再次追问模块")

    return {
        "session_id": session.get("session_id"),
        "stage": session.get("stage"),
        "selected_product": session["selected_product"],
        "selected_customer": session["selected_customer"],
        "customer_preference_profile": session["customer_preference_profile"],
        "training_goals": session["training_goals"],
        "goal_status": session["goal_status"],
        "round_state": session["round_state"],
        "question_logs": session["question_logs"],
        "risk_logs": session.get("risk_logs", []),
        "exhausted_goal_ids": session.get("exhausted_goal_ids", []),
        "opening": session.get("opening"),
    }


def build_next_followup_prompt_view(
    context: dict[str, Any],
    probe_plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "product": build_product_for_customer_question(context["selected_product"]),
        "customer": build_customer_brief(context["selected_customer"]),
        "customer_preference_profile": context["customer_preference_profile"],
        "training_goals": build_training_goal_summary(context),
        "goal_status": build_goal_status_summary(context),
        "unmet_goals": build_unmet_goal_summary(context),
        "probe_plan": probe_plan,
        "round_state": compact_round_state(context["round_state"]),
        "last_turn": build_last_turn_summary(context),
        "opening": context.get("opening"),
        "is_first_formal_followup": is_first_formal_followup(context),
        "risk_logs": context.get("risk_logs", []),
        "exhausted_goal_ids": context.get("exhausted_goal_ids", []),
    }


def build_next_followup_probe_plan(context: dict[str, Any]) -> dict[str, Any]:
    goal_detail_map = build_goal_detail_map(context)
    exhausted_goal_ids = set(context.get("exhausted_goal_ids", []))
    allow_observation_target = should_allow_observation_targets(context)
    candidate_goal_ids = rank_next_followup_candidate_goal_ids(context)
    target_goal_ids = candidate_goal_ids[:1]
    secondary_probe_goal_ids = [
        goal_id
        for goal_id in candidate_goal_ids[1:2]
        if goal_id not in target_goal_ids
    ]
    round_number = int(context["round_state"].get("current_round", 0)) + 1

    return {
        "round": round_number,
        "strategy": build_followup_strategy_text(allow_observation_target),
        "tone": "realistic_professional_customer",
        "tone_rules": [
            "客户应保持真实业务场景中的专业、直接和审慎。",
            "客户可以提出合理质疑、采购限制或临床顾虑，但不能恶意攻击。",
            "客户不能主动替学员拆解回答方向，也不能像老师一样提示应该怎么答。",
        ],
        "allow_observation_target": allow_observation_target,
        "target_goal_ids": target_goal_ids,
        "target_goals": [
            goal_detail_map[goal_id]
            for goal_id in target_goal_ids
            if goal_id in goal_detail_map
        ],
        "secondary_probe_goal_ids": secondary_probe_goal_ids,
        "secondary_probe_goals": [
            goal_detail_map[goal_id]
            for goal_id in secondary_probe_goal_ids
            if goal_id in goal_detail_map
        ],
        "difficulty": "realistic",
        "last_turn": build_last_turn_summary(context),
        "exhausted_goal_ids": sorted(exhausted_goal_ids),
        "selection_rules": [
            "如果 product_basic_introduction 尚未 met，必须优先追问该目标，先让学员补清产品基础信息。",
            "排除已经 met 的目标。",
            "优先避开已经连续两次主测但仍未 met 的 exhausted 目标。",
            "常规阶段只主动追问 core_goals 和 secondary_goals。",
            "优先追问 partial 核心目标，其次未测试或未达标核心目标，再追问次级目标。",
            "如果只剩 exhausted 目标且尚未达到结束条件，可以把 exhausted 目标作为最后兜底。",
            "只有当核心目标和次级目标都已经 met 时，才允许主动追问 observation_goals。",
        ],
    }


def rank_next_followup_candidate_goal_ids(context: dict[str, Any]) -> list[str]:
    product_intro_status = context["goal_status"].get(PRODUCT_INTRODUCTION_GOAL_ID, {})
    if (
        product_intro_status
        and not is_goal_met(product_intro_status)
        and PRODUCT_INTRODUCTION_GOAL_ID
        not in set(context.get("exhausted_goal_ids", []))
    ):
        return [PRODUCT_INTRODUCTION_GOAL_ID]

    ranked_buckets = [
        ("core_goals", "partial"),
        ("core_goals", "untested_or_not_met"),
        ("secondary_goals", "partial"),
        ("secondary_goals", "untested_or_not_met"),
    ]
    if should_allow_observation_targets(context):
        ranked_buckets.extend(
            [
                ("observation_goals", "partial"),
                ("observation_goals", "untested_or_not_met"),
            ]
        )

    exhausted_goal_ids = set(context.get("exhausted_goal_ids", []))
    ranked_goal_ids = []
    fallback_exhausted_goal_ids = []

    for bucket, category in ranked_buckets:
        for goal in context["training_goals"].get(bucket, []):
            goal_id = goal["goal_id"]
            status = context["goal_status"].get(goal_id, {})
            if is_goal_met(status):
                continue

            if goal_matches_followup_category(status, category):
                if goal_id in exhausted_goal_ids:
                    fallback_exhausted_goal_ids.append(goal_id)
                else:
                    ranked_goal_ids.append(goal_id)

    return dedupe_preserve_order(ranked_goal_ids or fallback_exhausted_goal_ids)


def is_first_formal_followup(context: dict[str, Any]) -> bool:
    return int(context["round_state"].get("current_round", 0)) == 0


def should_allow_observation_targets(context: dict[str, Any]) -> bool:
    return all_goals_met(context, "core_goals") and all_goals_met(
        context,
        "secondary_goals",
    )


def build_followup_strategy_text(allow_observation_target: bool) -> str:
    if allow_observation_target:
        return "核心目标和次级目标均已达标，根据当前 goal_status 选择一个尚未 met 的观察目标，并自然衔接上一轮回答继续追问。"

    return "根据当前 goal_status 选择一个尚未 met 的核心或次级目标，并自然衔接上一轮回答继续追问。"
