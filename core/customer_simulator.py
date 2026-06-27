from __future__ import annotations

from typing import Any


PRODUCT_INTRODUCTION_GOAL_ID = "product_basic_introduction"
OPENING_QUESTION_TEXT = "请你先用简短的话，面向这位客户介绍一下本次产品。"


def build_fixed_opening_question(session: dict[str, Any]) -> dict[str, Any]:
    selected_customer = session["selected_customer"]
    selected_product = session["selected_product"]

    return {
        "round": 0,
        "round_type": "opening",
        "question_type": "fixed_opening",
        "customer_role": selected_customer["role"],
        "product_name": selected_product["name"],
        "customer_question": OPENING_QUESTION_TEXT,
        "target_goal_ids": [PRODUCT_INTRODUCTION_GOAL_ID],
        "secondary_probe_goal_ids": [],
        "question_intent": "让学员先完成基础产品介绍，评价其是否能讲清产品定位、适用场景、核心价值和关键边界。",
        "difficulty": "opening",
    }


def record_opening_answer(
    session: dict[str, Any],
    learner_answer: str,
    *,
    opening_question: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_answer = learner_answer.strip()
    if not normalized_answer:
        raise ValueError("学员开场回答不能为空。")

    question = opening_question or build_fixed_opening_question(session)
    opening = {
        "question": question,
        "learner_answer": normalized_answer,
    }
    session["opening"] = opening
    session["stage"] = "opening_answered"
    return opening
