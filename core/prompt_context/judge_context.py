from __future__ import annotations

from typing import Any

from core.prompt_context.common import compact_round_state, require_session_keys
from core.prompt_context.customer_context import build_customer_brief
from core.prompt_context.goal_context import build_all_goal_items_for_judge
from core.prompt_context.product_context import build_product_for_judge


def build_judge_context_view(
    session: dict[str, Any],
    question_result: dict[str, Any],
    learner_answer: str,
    *,
    met_score_threshold: int,
) -> dict[str, Any]:
    required_session_keys = [
        "selected_product",
        "selected_customer",
        "customer_preference_profile",
        "training_goals",
        "goal_status",
        "round_state",
    ]
    require_session_keys(session, required_session_keys, "LLM Judge")

    normalized_answer = learner_answer.strip()
    if not normalized_answer:
        raise ValueError("学员本轮回答不能为空。")

    all_goals = build_all_goal_items_for_judge(session, question_result)

    return {
        "session_id": session.get("session_id"),
        "selected_product": session["selected_product"],
        "selected_customer": session["selected_customer"],
        "customer_preference_profile": session["customer_preference_profile"],
        "round_state": session["round_state"],
        "question_result": question_result,
        "learner_answer": normalized_answer,
        "all_goals": all_goals,
        "judge_policy": {
            "score_range": "0-4",
            "met_score_threshold": met_score_threshold,
            "primary_goal_ids": question_result.get("target_goal_ids", []),
            "secondary_probe_goal_ids": question_result.get(
                "secondary_probe_goal_ids", []
            ),
            "evaluate_all_goals": False,
            "evaluate_only_goals_to_evaluate": True,
            "check_compliance_every_round": True,
            "history_policy": "第一版只看当前轮问题和回答，不读取完整历史。",
        },
    }


def build_judge_prompt_view(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "product": build_product_for_judge(context["selected_product"]),
        "customer": build_customer_brief(context["selected_customer"]),
        "customer_preference_profile": context["customer_preference_profile"],
        "round_state": compact_round_state(context["round_state"]),
        "question_result": context["question_result"],
        "learner_answer": context["learner_answer"],
        "goals_to_evaluate": context["all_goals"],
        "judge_policy": context["judge_policy"],
    }
