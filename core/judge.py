from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from core.prompt_context.judge_context import (
    build_judge_context_view,
    build_judge_prompt_view,
)
from core.llm_client import build_basic_messages, call_doubao_chat_json


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
JUDGE_ANSWER_PROMPT = "judge_answer_system.txt"
MET_SCORE_THRESHOLD = 3
VALID_STATUSES = {"not_observed", "not_met", "partial", "met", "risk"}
VALID_SCOPES = {"primary", "secondary", "natural", "compliance"}
VALID_CONFIDENCE = {"low", "medium", "high"}
VALID_RISK_SEVERITY = {"minor", "moderate", "severe"}
ProgressCallback = Callable[[dict[str, Any]], None]


def build_judge_context(
    session: dict[str, Any],
    question_result: dict[str, Any],
    learner_answer: str,
) -> dict[str, Any]:
    return build_judge_context_view(
        session,
        question_result,
        learner_answer,
        met_score_threshold=MET_SCORE_THRESHOLD,
    )


def build_judge_prompt(context: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = load_prompt(JUDGE_ANSWER_PROMPT)
    user_prompt = json.dumps(
        build_judge_prompt_view(context),
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return build_basic_messages(system_prompt, user_prompt)


def judge_learner_answer(
    session: dict[str, Any],
    question_result: dict[str, Any],
    learner_answer: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    input_started_at = time.perf_counter()
    _emit_judge_progress(
        progress_callback,
        "llm_judge.build_input",
        "Judge：构造评价输入",
        "running",
        "整理当前问题、学员回答、产品资料和评价目标。",
    )
    try:
        context = build_judge_context(session, question_result, learner_answer)
        messages = build_judge_prompt(context)
    except Exception as exc:
        _emit_judge_progress(
            progress_callback,
            "llm_judge.build_input",
            "Judge：构造评价输入",
            "failed",
            f"构造评价输入失败：{exc}",
            duration_seconds=time.perf_counter() - input_started_at,
        )
        raise

    prompt_chars = sum(len(message.get("content", "")) for message in messages)
    _emit_judge_progress(
        progress_callback,
        "llm_judge.build_input",
        "Judge：构造评价输入",
        "completed",
        f"评价输入已构造：{len(context['all_goals'])} 个目标，约 {prompt_chars} 字符。",
        duration_seconds=time.perf_counter() - input_started_at,
    )

    llm_started_at = time.perf_counter()
    _emit_judge_progress(
        progress_callback,
        "llm_judge.call_llm",
        "Judge：调用 LLM",
        "running",
        "向远程 LLM 发送评价请求，并等待 JSON 返回。",
    )
    try:
        raw_response = call_doubao_chat_json(
            messages,
            model=os.environ.get("ARK_JUDGE_MODEL") or None,
            temperature=0.1,
        )
    except Exception as exc:
        _emit_judge_progress(
            progress_callback,
            "llm_judge.call_llm",
            "Judge：调用 LLM",
            "failed",
            f"LLM 调用失败：{exc}",
            duration_seconds=time.perf_counter() - llm_started_at,
        )
        raise

    _emit_judge_progress(
        progress_callback,
        "llm_judge.call_llm",
        "Judge：调用 LLM",
        "completed",
        f"LLM 已返回 JSON 文本，约 {len(raw_response)} 字符。",
        duration_seconds=time.perf_counter() - llm_started_at,
    )

    parse_started_at = time.perf_counter()
    _emit_judge_progress(
        progress_callback,
        "llm_judge.parse_result",
        "Judge：解析结果",
        "running",
        "解析并规范化 Judge 返回的目标评价和风险信息。",
    )
    try:
        judge_result = parse_judge_response(raw_response, context=context)
    except Exception as exc:
        _emit_judge_progress(
            progress_callback,
            "llm_judge.parse_result",
            "Judge：解析结果",
            "failed",
            f"解析 Judge 结果失败：{exc}",
            duration_seconds=time.perf_counter() - parse_started_at,
        )
        raise

    _emit_judge_progress(
        progress_callback,
        "llm_judge.parse_result",
        "Judge：解析结果",
        "completed",
        f"Judge 结果已解析：{len(judge_result['goal_evaluations'])} 个目标评价。",
        duration_seconds=time.perf_counter() - parse_started_at,
    )
    return judge_result


def parse_judge_response(
    raw_response: str,
    *,
    context: dict[str, Any],
) -> dict[str, Any]:
    parsed = _loads_json_object(raw_response)
    all_goals = context["all_goals"]
    goal_lookup = {goal["goal_id"]: goal for goal in all_goals}
    raw_evaluations = parsed.get("goal_evaluations", [])
    if not isinstance(raw_evaluations, list):
        raw_evaluations = []

    normalized_evaluations = {}
    for raw_evaluation in raw_evaluations:
        if not isinstance(raw_evaluation, dict):
            continue

        goal_id = str(raw_evaluation.get("goal_id", ""))
        if goal_id not in goal_lookup:
            continue

        normalized_evaluations[goal_id] = _normalize_goal_evaluation(
            raw_evaluation,
            goal_lookup[goal_id],
        )

    for goal in all_goals:
        if goal["goal_id"] not in normalized_evaluations:
            normalized_evaluations[goal["goal_id"]] = _empty_goal_evaluation(goal)

    risk_flags = _normalize_risk_flags(parsed.get("risk_flags", []))

    return {
        "round": _normalize_round(parsed.get("round"), context),
        "evaluated_question": context["question_result"].get("customer_question", ""),
        "evaluated_answer": context["learner_answer"],
        "goal_evaluations": [
            normalized_evaluations[goal["goal_id"]]
            for goal in all_goals
        ],
        "risk_flags": risk_flags,
        "has_risk": bool(risk_flags),
        "overall_comment": str(parsed.get("overall_comment") or "").strip(),
        "raw_response": raw_response,
    }


def load_prompt(prompt_filename: str) -> str:
    path = PROMPTS_DIR / prompt_filename
    if not path.exists():
        raise FileNotFoundError(f"找不到 prompt 文件：{path}")

    return path.read_text(encoding="utf-8").strip()


def _emit_judge_progress(
    progress_callback: ProgressCallback | None,
    module_key: str,
    module_label: str,
    status: str,
    message: str,
    *,
    duration_seconds: float | None = None,
) -> None:
    if progress_callback is None:
        return

    event: dict[str, Any] = {
        "scope": "llm_judge_internal",
        "parent_module_key": "llm_judge",
        "module_key": module_key,
        "module_label": module_label,
        "status": status,
        "message": message,
    }
    if duration_seconds is not None:
        event["duration_seconds"] = duration_seconds

    progress_callback(event)


def _build_all_goal_items(
    session: dict[str, Any],
    question_result: dict[str, Any],
) -> list[dict[str, Any]]:
    primary_goal_ids = set(question_result.get("target_goal_ids", []))
    secondary_goal_ids = set(question_result.get("secondary_probe_goal_ids", []))
    all_goals = []

    for bucket, goals in session["training_goals"].items():
        for goal in goals:
            goal_id = goal["goal_id"]
            if goal_id in primary_goal_ids:
                evaluation_scope = "primary"
            elif goal_id in secondary_goal_ids:
                evaluation_scope = "secondary"
            else:
                evaluation_scope = "natural"

            all_goals.append(
                {
                    "goal_id": goal_id,
                    "goal_name": goal["name"],
                    "description": goal.get("description"),
                    "bucket": bucket,
                    "evaluation_scope": evaluation_scope,
                    "current_status": session["goal_status"]
                    .get(goal_id, {})
                    .get("status", "untested"),
                }
            )

    return all_goals


def _product_for_judge(product: dict[str, Any]) -> dict[str, Any]:
    return {
        "product_id": product.get("product_id"),
        "name": product.get("name"),
        "industry": product.get("industry"),
        "category": product.get("category"),
        "risk_level": product.get("risk_level"),
        "description": product.get("description"),
        "selling_product": product.get("selling_product", {}),
        "competitor_product": product.get("competitor_product", {}),
        "comparison_context": product.get("comparison_context", {}),
        "key_values": product.get("key_values", []),
        "risk_constraints": product.get("risk_constraints", []),
        "compliance_goal_policy": product.get("compliance_goal_policy", {}),
    }


def _normalize_goal_evaluation(
    raw_evaluation: dict[str, Any],
    goal: dict[str, Any],
) -> dict[str, Any]:
    score = _normalize_score(raw_evaluation.get("score"))
    raw_status = str(raw_evaluation.get("status") or "").strip()
    status = _status_from_score(score, raw_status)
    confidence = str(raw_evaluation.get("confidence") or "medium").strip()
    if confidence not in VALID_CONFIDENCE:
        confidence = "medium"

    return {
        "goal_id": goal["goal_id"],
        "goal_name": goal["goal_name"],
        "bucket": goal["bucket"],
        "evaluation_scope": _normalize_scope(
            raw_evaluation.get("evaluation_scope"),
            goal["evaluation_scope"],
        ),
        "status": status,
        "score": score,
        "is_goal_met": status == "met" and score >= MET_SCORE_THRESHOLD,
        "evidence": str(raw_evaluation.get("evidence") or "").strip(),
        "reason": str(raw_evaluation.get("reason") or "").strip(),
        "confidence": confidence,
    }


def _empty_goal_evaluation(goal: dict[str, Any]) -> dict[str, Any]:
    return {
        "goal_id": goal["goal_id"],
        "goal_name": goal["goal_name"],
        "bucket": goal["bucket"],
        "evaluation_scope": goal["evaluation_scope"],
        "status": "not_observed",
        "score": 0,
        "is_goal_met": False,
        "evidence": "",
        "reason": "LLM Judge 未返回该目标评价，系统按未观察到处理。",
        "confidence": "low",
    }


def _normalize_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = 0

    return min(4, max(0, score))


def _status_from_score(score: int, raw_status: str) -> str:
    if raw_status == "risk":
        return "risk"
    if score >= MET_SCORE_THRESHOLD:
        return "met"
    if score == 2:
        return "partial"
    if score == 1:
        return "not_met"
    if raw_status in {"not_met", "not_observed"}:
        return raw_status
    return "not_observed"


def _normalize_scope(raw_scope: Any, fallback_scope: str) -> str:
    scope = str(raw_scope or fallback_scope).strip()
    if scope not in VALID_SCOPES:
        return fallback_scope
    return scope


def _normalize_risk_flags(raw_risk_flags: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_risk_flags, list):
        return []

    risk_flags = []
    for raw_flag in raw_risk_flags:
        if not isinstance(raw_flag, dict):
            continue

        severity = str(raw_flag.get("severity") or "minor").strip()
        if severity not in VALID_RISK_SEVERITY:
            severity = "minor"

        evidence = str(raw_flag.get("evidence") or "").strip()
        risk_flags.append(
            {
                "risk_type": str(raw_flag.get("risk_type") or "未分类风险").strip(),
                "severity": severity,
                "evidence": evidence,
                "reason": str(raw_flag.get("reason") or "").strip(),
            }
        )

    return risk_flags


def _normalize_round(raw_round: Any, context: dict[str, Any]) -> int:
    expected_round = _round_from_question_result(context)
    try:
        normalized_round = int(raw_round)
    except (TypeError, ValueError):
        return expected_round

    if normalized_round != expected_round:
        return expected_round

    return normalized_round


def _round_from_question_result(context: dict[str, Any]) -> int:
    try:
        return int(context["question_result"].get("round", 1))
    except (TypeError, ValueError):
        return 1


def _loads_json_object(raw_response: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"无法解析 Judge JSON 输出：{raw_response}") from None
        parsed = json.loads(raw_response[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError(f"Judge 输出不是 JSON 对象：{raw_response}")

    return parsed
