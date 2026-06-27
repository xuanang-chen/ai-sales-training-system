from __future__ import annotations

from copy import deepcopy
from typing import Any


MET_SCORE_THRESHOLD = 3
REQUIRED_GOAL_BUCKETS = {"core_goals", "secondary_goals"}


def update_training_state(
    session: dict[str, Any],
    question_result: dict[str, Any],
    learner_answer: str,
    judge_result: dict[str, Any],
) -> dict[str, Any]:
    round_number = _resolve_round(session, question_result, judge_result)
    is_opening_round = is_opening_question(question_result)
    _ensure_session_tracking_fields(session)

    newly_met_goal_ids = update_goal_status(session, judge_result, round_number)
    newly_met_required_goal_ids = [
        goal_id
        for goal_id in newly_met_goal_ids
        if session["goal_status"][goal_id]["bucket"] in REQUIRED_GOAL_BUCKETS
    ]
    risk_logs = append_risk_logs(
        session,
        judge_result,
        round_number,
        question_result.get("round_type"),
    )
    append_question_log(
        session,
        question_result,
        learner_answer,
        judge_result,
        round_number,
    )
    if not is_opening_round:
        update_round_state(
            session,
            round_number,
            newly_met_required_goal_ids,
        )

    session["stage"] = "opening_state_updated" if is_opening_round else "state_updated"

    return {
        "stage": session["stage"],
        "round": round_number,
        "round_type": question_result.get("round_type"),
        "newly_met_goal_ids": newly_met_goal_ids,
        "newly_met_required_goal_ids": newly_met_required_goal_ids,
        "risk_count": len(risk_logs),
        "no_progress_count": session["round_state"]["no_progress_count"],
    }


def update_goal_status(
    session: dict[str, Any],
    judge_result: dict[str, Any],
    round_number: int,
) -> list[str]:
    newly_met_goal_ids = []

    for evaluation in judge_result.get("goal_evaluations", []):
        goal_id = evaluation.get("goal_id")
        if goal_id not in session["goal_status"]:
            continue

        status_item = _ensure_goal_status_shape(session["goal_status"][goal_id])
        was_met = status_item["status"] == "met"
        score = _normalize_score(evaluation.get("score"))
        has_evidence = bool(str(evaluation.get("evidence") or "").strip())
        has_goal_risk = evaluation.get("status") == "risk"

        history_item = {
            "round": round_number,
            "status": evaluation.get("status"),
            "score": score,
            "is_goal_met": score >= MET_SCORE_THRESHOLD,
            "evidence": str(evaluation.get("evidence") or "").strip(),
            "reason": str(evaluation.get("reason") or "").strip(),
            "confidence": evaluation.get("confidence"),
            "evaluation_scope": evaluation.get("evaluation_scope"),
        }
        status_item["score_history"].append(history_item)

        status_item["latest_score"] = score
        status_item["latest_evidence"] = history_item["evidence"]
        status_item["latest_reason"] = history_item["reason"]
        status_item["updated_round"] = round_number
        status_item["score"] = score

        if has_evidence and round_number not in status_item["evidence_rounds"]:
            status_item["evidence_rounds"].append(round_number)

        if score > _score_or_negative(status_item["best_score"]):
            status_item["best_score"] = score
            status_item["best_evidence"] = history_item["evidence"]
            status_item["best_reason"] = history_item["reason"]
            status_item["best_round"] = round_number

        if has_goal_risk:
            status_item["has_risk"] = True
            if round_number not in status_item["risk_rounds"]:
                status_item["risk_rounds"].append(round_number)

        if score >= MET_SCORE_THRESHOLD:
            status_item["status"] = "met"
            if status_item["met_round"] is None:
                status_item["met_round"] = round_number
            if not was_met:
                newly_met_goal_ids.append(goal_id)
            continue

        if was_met:
            status_item["status"] = "met"
            continue

        status_item["status"] = _status_from_best_score(
            status_item["best_score"],
            status_item["score_history"],
        )

    return newly_met_goal_ids


def append_question_log(
    session: dict[str, Any],
    question_result: dict[str, Any],
    learner_answer: str,
    judge_result: dict[str, Any],
    round_number: int,
) -> dict[str, Any]:
    log_item = {
        "round": round_number,
        "round_type": question_result.get("round_type"),
        "question_type": question_result.get("question_type"),
        "question_result": deepcopy(question_result),
        "customer_question": question_result.get("customer_question", ""),
        "target_goal_ids": list(question_result.get("target_goal_ids", [])),
        "secondary_probe_goal_ids": list(
            question_result.get("secondary_probe_goal_ids", [])
        ),
        "learner_answer": learner_answer,
        "judge_result": deepcopy(judge_result),
    }
    session["question_logs"].append(log_item)
    return log_item


def append_risk_logs(
    session: dict[str, Any],
    judge_result: dict[str, Any],
    round_number: int,
    round_type: str | None = None,
) -> list[dict[str, Any]]:
    new_risk_logs = []

    for risk_flag in judge_result.get("risk_flags", []):
        risk_log = {
            "round": round_number,
            "round_type": round_type,
            "risk_type": risk_flag.get("risk_type", "未分类风险"),
            "severity": risk_flag.get("severity", "minor"),
            "evidence": risk_flag.get("evidence", ""),
            "reason": risk_flag.get("reason", ""),
        }
        session["risk_logs"].append(risk_log)
        new_risk_logs.append(risk_log)

    return new_risk_logs


def update_round_state(
    session: dict[str, Any],
    round_number: int,
    newly_met_required_goal_ids: list[str],
) -> dict[str, Any]:
    round_state = session["round_state"]
    round_state["current_round"] = max(
        int(round_state.get("current_round", 0)),
        round_number,
    )

    if newly_met_required_goal_ids:
        round_state["no_progress_count"] = 0
    else:
        round_state["no_progress_count"] = int(
            round_state.get("no_progress_count", 0)
        ) + 1

    return round_state


def is_opening_question(question_result: dict[str, Any]) -> bool:
    return question_result.get("round_type") == "opening"


def _ensure_session_tracking_fields(session: dict[str, Any]) -> None:
    session.setdefault("question_logs", [])
    session.setdefault("risk_logs", [])

    for goal_status in session.get("goal_status", {}).values():
        _ensure_goal_status_shape(goal_status)


def _ensure_goal_status_shape(goal_status: dict[str, Any]) -> dict[str, Any]:
    legacy_score = goal_status.get("score")
    goal_status.setdefault("status", "untested")
    goal_status.setdefault("best_score", legacy_score)
    goal_status.setdefault("latest_score", legacy_score)
    goal_status.setdefault("evidence_rounds", [])
    goal_status.setdefault("score_history", [])
    goal_status.setdefault("best_evidence", "")
    goal_status.setdefault("latest_evidence", "")
    goal_status.setdefault("best_reason", "")
    goal_status.setdefault("latest_reason", "")
    goal_status.setdefault("best_round", None)
    goal_status.setdefault("met_round", None)
    goal_status.setdefault("updated_round", None)
    goal_status.setdefault("has_risk", False)
    goal_status.setdefault("risk_rounds", [])
    return goal_status


def _resolve_round(
    session: dict[str, Any],
    question_result: dict[str, Any],
    judge_result: dict[str, Any],
) -> int:
    for source in (question_result, judge_result):
        try:
            return int(source.get("round"))
        except (TypeError, ValueError):
            continue

    return int(session.get("round_state", {}).get("current_round", 0)) + 1


def _normalize_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = 0

    return min(4, max(0, score))


def _score_or_negative(value: Any) -> int:
    if value is None:
        return -1
    return _normalize_score(value)


def _status_from_best_score(
    best_score: int | None,
    score_history: list[dict[str, Any]],
) -> str:
    if best_score is None:
        return "untested"
    if best_score >= MET_SCORE_THRESHOLD:
        return "met"
    if best_score == 2:
        return "partial"
    if _has_observed_attempt(score_history):
        return "not_met"
    return "not_observed"


def _has_observed_attempt(score_history: list[dict[str, Any]]) -> bool:
    for item in score_history:
        if item.get("score", 0) > 0:
            return True
        if item.get("status") in {"not_met", "partial", "met", "risk"}:
            return True
        if item.get("evidence"):
            return True

    return False
