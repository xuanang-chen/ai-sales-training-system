from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.prompt_context.next_followup_context import (
    build_next_customer_simulator_context as _build_next_customer_simulator_context,
    build_next_followup_probe_plan as _build_next_followup_probe_plan,
    build_next_followup_prompt_view,
)
from core.llm_client import build_basic_messages, call_doubao_chat_json


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
NEXT_CUSTOMER_FOLLOWUP_PROMPT = "next_customer_followup_system.txt"


def generate_next_customer_followup(session: dict[str, Any]) -> dict[str, Any]:
    refresh_exhausted_goal_ids(session)
    context = build_next_customer_simulator_context(session)
    probe_plan = build_next_followup_probe_plan(context)

    if not probe_plan["target_goal_ids"]:
        return {
            "should_generate": False,
            "next_action": "no_available_followup_target",
            "reason": "没有可用于再次追问的未达标核心目标或次级目标。",
            "probe_plan": probe_plan,
        }

    raw_response = call_doubao_chat_json(
        build_next_followup_question_prompt(context, probe_plan),
        temperature=0.4,
    )
    question_result = parse_next_followup_response(
        raw_response,
        context=context,
        probe_plan=probe_plan,
    )

    return {
        "should_generate": True,
        "next_action": "ask_customer_question",
        "probe_plan": probe_plan,
        "question_result": question_result,
    }


def build_next_followup_question_prompt(
    context: dict[str, Any],
    probe_plan: dict[str, Any],
) -> list[dict[str, str]]:
    system_prompt = load_prompt(NEXT_CUSTOMER_FOLLOWUP_PROMPT)
    user_prompt = json.dumps(
        build_next_followup_prompt_view(context, probe_plan),
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return build_basic_messages(system_prompt, user_prompt)


def build_next_followup_probe_plan(context: dict[str, Any]) -> dict[str, Any]:
    return _build_next_followup_probe_plan(context)


def refresh_exhausted_goal_ids(session: dict[str, Any]) -> list[str]:
    exhausted_goal_ids = set(session.get("exhausted_goal_ids", []))
    goal_status = session.get("goal_status", {})
    logs = session.get("question_logs", [])

    for goal_id, status in goal_status.items():
        if is_goal_met(status):
            continue
        if has_two_consecutive_target_attempts(logs, goal_id):
            exhausted_goal_ids.add(goal_id)

    session["exhausted_goal_ids"] = sorted(exhausted_goal_ids)
    return session["exhausted_goal_ids"]


def build_next_customer_simulator_context(session: dict[str, Any]) -> dict[str, Any]:
    return _build_next_customer_simulator_context(session)


def parse_next_followup_response(
    raw_response: str,
    *,
    context: dict[str, Any],
    probe_plan: dict[str, Any],
) -> dict[str, Any]:
    parsed = loads_json_object(raw_response)
    known_goal_ids = set(context["goal_status"])
    allowed_target_goal_ids = set(probe_plan.get("target_goal_ids", []))
    allowed_secondary_goal_ids = set(probe_plan.get("secondary_probe_goal_ids", []))

    target_goal_ids = filter_known_goal_ids(
        ensure_list(parsed.get("target_goal_ids")),
        known_goal_ids,
    )
    if allowed_target_goal_ids:
        target_goal_ids = [
            goal_id for goal_id in target_goal_ids if goal_id in allowed_target_goal_ids
        ]
    if not target_goal_ids:
        target_goal_ids = list(probe_plan.get("target_goal_ids", []))

    secondary_probe_goal_ids = filter_known_goal_ids(
        ensure_list(parsed.get("secondary_probe_goal_ids")),
        known_goal_ids,
    )
    if allowed_secondary_goal_ids:
        secondary_probe_goal_ids = [
            goal_id
            for goal_id in secondary_probe_goal_ids
            if goal_id in allowed_secondary_goal_ids
        ]
    else:
        secondary_probe_goal_ids = []

    question = str(parsed.get("customer_question") or "").strip()
    if not question:
        raise ValueError(f"客户追问缺少 customer_question：{raw_response}")
    question = normalize_friendly_customer_question(question)

    difficulty = probe_plan.get("difficulty") or "soft"

    return {
        "round": int(probe_plan.get("round", parsed.get("round", 1))),
        "round_type": parsed.get("round_type") or "formal_question",
        "question_type": parsed.get("question_type") or "customer_followup",
        "target_goal_ids": target_goal_ids,
        "secondary_probe_goal_ids": secondary_probe_goal_ids,
        "customer_question": question,
        "question_intent": str(parsed.get("question_intent") or "").strip(),
        "difficulty": difficulty,
        "raw_response": raw_response,
    }


def rank_next_followup_candidate_goal_ids(context: dict[str, Any]) -> list[str]:
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


def build_followup_strategy_text(allow_observation_target: bool) -> str:
    if allow_observation_target:
        return "核心目标和次级目标均已达标，根据当前 goal_status 选择一个尚未 met 的观察目标，并自然衔接上一轮回答继续追问。"

    return "根据当前 goal_status 选择一个尚未 met 的核心或次级目标，并自然衔接上一轮回答继续追问。"


def should_allow_observation_targets(context: dict[str, Any]) -> bool:
    return all_goals_met(context, "core_goals") and all_goals_met(
        context,
        "secondary_goals",
    )


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


def choose_next_followup_difficulty(
    context: dict[str, Any],
    target_goal_ids: list[str],
) -> str:
    return "soft"


def build_last_turn_summary(context: dict[str, Any]) -> dict[str, Any] | None:
    question_logs = context.get("question_logs", [])
    if not question_logs:
        return None

    last_log = question_logs[-1]
    judge_result = last_log.get("judge_result", {})
    return {
        "round": last_log.get("round"),
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


def has_two_consecutive_target_attempts(
    question_logs: list[dict[str, Any]],
    goal_id: str,
) -> bool:
    consecutive_attempts = 0
    for log in question_logs:
        if log.get("round_type") == "opening":
            continue

        if goal_id in log.get("target_goal_ids", []):
            consecutive_attempts += 1
            if consecutive_attempts >= 2:
                return True
        else:
            consecutive_attempts = 0

    return False


def build_goal_detail_map(context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    goal_map = {}
    for bucket, goals in context["training_goals"].items():
        for goal in goals:
            goal_map[goal["goal_id"]] = {
                "goal_id": goal["goal_id"],
                "name": goal["name"],
                "description": goal.get("description"),
                "bucket": bucket,
                "status": context["goal_status"]
                .get(goal["goal_id"], {})
                .get("status", "untested"),
            }

    return goal_map


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
            status = goal_status.get(goal["goal_id"], {}).get("status", "untested")
            if status == "met":
                continue

            unmet_goals[bucket].append(
                {
                    "goal_id": goal["goal_id"],
                    "name": goal["name"],
                    "description": goal.get("description"),
                    "status": status,
                }
            )

    return unmet_goals


def is_goal_met(goal_status: dict[str, Any]) -> bool:
    return goal_status.get("status") == "met" or score_or_zero(
        goal_status.get("best_score")
    ) >= 3


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


def load_prompt(prompt_filename: str) -> str:
    path = PROMPTS_DIR / prompt_filename
    if not path.exists():
        raise FileNotFoundError(f"找不到 prompt 文件：{path}")

    return path.read_text(encoding="utf-8").strip()


def loads_json_object(raw_response: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"无法解析 JSON 输出：{raw_response}") from None
        parsed = json.loads(raw_response[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError(f"输出不是 JSON 对象：{raw_response}")

    return parsed


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def filter_known_goal_ids(goal_ids: list[Any], known_goal_ids: set[str]) -> list[str]:
    filtered = []
    for goal_id in goal_ids:
        normalized = str(goal_id)
        if normalized in known_goal_ids and normalized not in filtered:
            filtered.append(normalized)

    return filtered


def normalize_friendly_customer_question(question: str) -> str:
    normalized = question.strip()
    replacements = {
        "凭什么让我相信": "这个判断依据是什么",
        "你这说法不对吧": "这个说法我还不能直接接受",
        "你们这说法不对吧": "这个说法我还不能直接接受",
        "凭什么": "依据是什么",
        "你怎么证明": "你们怎么说明",
        "你们怎么证明": "你们怎么说明",
        "这不就是": "这是否就是",
    }

    for harsh_text, realistic_text in replacements.items():
        normalized = normalized.replace(harsh_text, realistic_text)

    return normalized.replace("！", "。").replace("!", ".")
