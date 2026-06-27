from __future__ import annotations

import random
import time
from typing import Any, Callable

from core.customer_simulator import (
    build_fixed_opening_question,
    record_opening_answer,
)
from core.customer_simulator_next import generate_next_customer_followup
from core.initialization import initialize_training_session
from core.judge import judge_learner_answer
from core.state_tracker import update_training_state
from core.termination import decide_termination

ProgressCallback = Callable[[dict[str, Any]], None]


def start_training_session(
    rng: random.Random | None = None,
) -> dict[str, Any]:
    session = initialize_training_session(rng=rng)
    opening_question = build_fixed_opening_question(session)

    session["opening"] = {
        "question": opening_question,
        "learner_answer": None,
    }
    session["stage"] = "opening_questioned"

    return {
        "session": session,
        "opening_question": opening_question,
    }


def submit_opening_answer(
    session: dict[str, Any],
    learner_opening_answer: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    opening_question = (session.get("opening") or {}).get("question")
    opening = record_opening_answer(
        session,
        learner_opening_answer,
        opening_question=opening_question,
    )
    question_result = opening["question"]

    processing_events: list[dict[str, Any]] = []

    def record_progress(event: dict[str, Any]) -> None:
        processing_events.append(event.copy())
        if progress_callback is not None:
            progress_callback(event)

    judge_result = _run_progress_step(
        record_progress,
        "llm_judge",
        "LLM Judge",
        "评价开场介绍中的产品基础信息、证据和风险。",
        lambda: judge_learner_answer(
            session,
            question_result,
            learner_opening_answer,
            progress_callback=record_progress,
        ),
    )
    state_update = _run_progress_step(
        record_progress,
        "state_tracker",
        "状态追踪器",
        "写入开场介绍评价结果并更新目标达成状态。",
        lambda: update_training_state(
            session,
            question_result,
            learner_opening_answer,
            judge_result,
        ),
    )
    termination_result = {
        "should_end": False,
        "end_type": "opening_default_continue",
        "reason": "开场阶段不做结束判断，默认进入首次客户追问。",
        "next_stage": "continue_training",
        "next_module": "customer_simulator_next",
    }
    session["termination_result"] = termination_result
    _emit_progress(
        record_progress,
        "termination_judge",
        "结束判断器",
        "skipped",
        termination_result["reason"],
        duration_seconds=0.0,
    )

    next_followup_result = _run_progress_step(
        record_progress,
        "next_followup",
        "再次追问器",
        "基于开场介绍评价生成首次客户追问。",
        lambda: generate_next_customer_followup(session),
    )
    if not next_followup_result["should_generate"]:
        raise RuntimeError(
            next_followup_result.get(
                "reason",
                "再次追问模块没有生成可用的首次客户追问。",
            )
        )

    next_question = next_followup_result["question_result"]
    session["pending_question"] = next_question
    session["stage"] = "customer_questioned"

    return {
        "session": session,
        "should_stop": False,
        "judge_result": judge_result,
        "state_update": state_update,
        "termination_result": termination_result,
        "next_followup_result": next_followup_result,
        "question_result": next_question,
        "processing_events": processing_events,
    }


def submit_customer_answer(
    session: dict[str, Any],
    learner_answer: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    question_result = session.get("pending_question")
    if not question_result:
        raise ValueError("session 中没有待回答的客户追问。")

    processing_events: list[dict[str, Any]] = []

    def record_progress(event: dict[str, Any]) -> None:
        processing_events.append(event.copy())
        if progress_callback is not None:
            progress_callback(event)

    judge_result = _run_progress_step(
        record_progress,
        "llm_judge",
        "LLM Judge",
        "评价回答中的目标证据、分数和风险。",
        lambda: judge_learner_answer(
            session,
            question_result,
            learner_answer,
            progress_callback=record_progress,
        ),
    )
    state_update = _run_progress_step(
        record_progress,
        "state_tracker",
        "状态追踪器",
        "写入本轮评价结果并更新目标达成状态。",
        lambda: update_training_state(
            session,
            question_result,
            learner_answer,
            judge_result,
        ),
    )
    termination_result = _run_progress_step(
        record_progress,
        "termination_judge",
        "结束判断器",
        "判断训练是否继续、结束或进入补充测试。",
        lambda: decide_termination(session),
    )
    session["termination_result"] = termination_result

    if termination_result["should_end"]:
        session["stage"] = "ended"
        session.pop("pending_question", None)
        next_followup_result = None
        _emit_progress(
            record_progress,
            "next_followup",
            "再次追问器",
            "skipped",
            "结束判断器决定训练结束，本轮不再生成新问题。",
        )
    else:
        next_followup_result = _run_progress_step(
            record_progress,
            "next_followup",
            "再次追问器",
            "选择下一轮追问目标，并生成新的客户问题。",
            lambda: generate_next_customer_followup(session),
        )
        if next_followup_result["should_generate"]:
            session["pending_question"] = next_followup_result["question_result"]
            session["stage"] = "customer_questioned"
        else:
            raise RuntimeError(
                next_followup_result.get(
                    "reason",
                    "再次追问模块没有生成可用问题，但结束判断器未要求终止。",
                )
            )

    return {
        "session": session,
        "judge_result": judge_result,
        "state_update": state_update,
        "termination_result": termination_result,
        "next_followup_result": next_followup_result,
        "processing_events": processing_events,
    }


def _run_progress_step(
    progress_callback: ProgressCallback | None,
    module_key: str,
    module_label: str,
    message: str,
    action: Callable[[], Any],
) -> Any:
    started_at = time.perf_counter()
    _emit_progress(
        progress_callback,
        module_key,
        module_label,
        "running",
        message,
    )
    try:
        result = action()
    except Exception as exc:
        _emit_progress(
            progress_callback,
            module_key,
            module_label,
            "failed",
            f"{module_label} 执行失败：{exc}",
            duration_seconds=time.perf_counter() - started_at,
        )
        raise

    _emit_progress(
        progress_callback,
        module_key,
        module_label,
        "completed",
        f"{module_label} 已完成。",
        duration_seconds=time.perf_counter() - started_at,
    )
    return result


def _emit_progress(
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
        "module_key": module_key,
        "module_label": module_label,
        "status": status,
        "message": message,
    }
    if duration_seconds is not None:
        event["duration_seconds"] = duration_seconds

    progress_callback(event)
