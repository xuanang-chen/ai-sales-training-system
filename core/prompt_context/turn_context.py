from __future__ import annotations

from typing import Any


def require_opening(context: dict[str, Any]) -> dict[str, Any]:
    opening = context.get("opening")
    if not opening:
        raise ValueError("session 中还没有 opening 信息，请先记录学员开场回答。")

    if not opening.get("learner_answer"):
        raise ValueError("opening 中缺少学员开场回答。")

    return opening


def build_last_turn_summary(context: dict[str, Any]) -> dict[str, Any] | None:
    question_logs = context.get("question_logs", [])
    if not question_logs:
        return None

    last_log = question_logs[-1]
    judge_result = last_log.get("judge_result", {})
    return {
        "round": last_log.get("round"),
        "round_type": last_log.get("round_type"),
        "question_type": last_log.get("question_type"),
        "customer_question": last_log.get("customer_question"),
        "target_goal_ids": last_log.get("target_goal_ids", []),
        "secondary_probe_goal_ids": last_log.get("secondary_probe_goal_ids", []),
        "learner_answer": last_log.get("learner_answer"),
        "judge_overall_comment": judge_result.get("overall_comment", ""),
        "goal_evaluations": [
            {
                "goal_id": item.get("goal_id"),
                "goal_name": item.get("goal_name"),
                "evaluation_scope": item.get("evaluation_scope"),
                "status": item.get("status"),
                "score": item.get("score"),
                "evidence": item.get("evidence"),
                "reason": item.get("reason"),
            }
            for item in judge_result.get("goal_evaluations", [])
            if item.get("evaluation_scope") in {"primary", "secondary"}
            or item.get("score", 0) > 0
        ],
        "risk_flags": judge_result.get("risk_flags", []),
    }
