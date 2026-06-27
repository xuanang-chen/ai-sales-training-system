from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class InitializationConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    product_file: str = "products.json"
    customer_file: str = "customers.json"
    customer_preference_file: str = "customer_preferences.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_initialization_sources(
    config: InitializationConfig | None = None,
) -> dict[str, Any]:
    config = config or InitializationConfig()
    data_dir = config.data_dir

    return {
        "products": load_json(data_dir / config.product_file),
        "customers": load_json(data_dir / config.customer_file),
        "customer_preferences": load_json(data_dir / config.customer_preference_file),
    }


def choose_random_context(
    sources: dict[str, Any],
    rng: random.Random | None = None,
) -> dict[str, Any]:
    rng = rng or random

    return {
        "product": rng.choice(sources["products"]),
        "customer": rng.choice(sources["customers"]),
        "preference_profile": generate_preference_profile(
            sources["customer_preferences"],
            rng=rng,
        ),
    }


def generate_preference_profile(
    customer_preferences: dict[str, Any],
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    rng = rng or random
    preference_types = list(customer_preferences["preference_types"])
    strength_levels = [item["level"] for item in customer_preferences["strength_levels"]]

    if len(preference_types) != len(strength_levels):
        raise ValueError("偏好类型数量必须和强度等级数量一致，才能一一随机分配。")

    shuffled_strengths = list(strength_levels)
    rng.shuffle(shuffled_strengths)

    profile = []
    for preference, strength in zip(preference_types, shuffled_strengths):
        profile.append(
            {
                "preference_id": preference["preference_id"],
                "name": preference["name"],
                "strength": strength,
                "description": preference["description"],
                "sample_customer_signals": preference.get("sample_customer_signals", []),
            }
        )

    return sorted(profile, key=lambda item: _strength_order(item["strength"]))


def generate_training_goals(
    product: dict[str, Any],
    preference_profile: list[dict[str, Any]],
    customer_preferences: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    preference_lookup = {
        item["preference_id"]: item
        for item in customer_preferences["preference_types"]
    }
    strength_to_bucket = {
        item["level"]: item["target_bucket"]
        for item in customer_preferences["strength_levels"]
    }

    goals: dict[str, list[dict[str, Any]]] = {
        "core_goals": [],
        "secondary_goals": [],
        "observation_goals": [],
    }
    goals["core_goals"].extend(customer_preferences.get("mandatory_core_goals", []))

    for profile_item in preference_profile:
        preference = preference_lookup[profile_item["preference_id"]]
        strength = profile_item["strength"]
        bucket = strength_to_bucket[strength]
        goals[bucket].extend(preference["goals_by_strength"].get(strength, []))

    _add_product_compliance_goals(product, goals)

    return {
        bucket: _dedupe_goals(bucket_goals)
        for bucket, bucket_goals in goals.items()
    }


def initialize_training_session(
    config: InitializationConfig | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    sources = load_initialization_sources(config)
    context = choose_random_context(sources, rng=rng)
    goals = generate_training_goals(
        product=context["product"],
        preference_profile=context["preference_profile"],
        customer_preferences=sources["customer_preferences"],
    )

    return {
        "session_id": str(uuid.uuid4()),
        "stage": "initialized",
        "selected_product": context["product"],
        "selected_customer": context["customer"],
        "customer_preference_profile": context["preference_profile"],
        "training_goals": goals,
        "goal_status": build_initial_goal_status(goals),
        "round_state": {
            "current_round": 0,
            "no_progress_count": 0,
            "max_rounds": calculate_max_rounds(goals),
        },
        "question_logs": [],
    }


def build_initial_goal_status(
    goals: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    status = {}

    for bucket, bucket_goals in goals.items():
        for goal in bucket_goals:
            status[goal["goal_id"]] = {
                "name": goal["name"],
                "bucket": bucket,
                "status": "untested",
                "score": None,
                "best_score": None,
                "latest_score": None,
                "evidence_rounds": [],
                "score_history": [],
                "best_evidence": "",
                "latest_evidence": "",
                "best_reason": "",
                "latest_reason": "",
                "best_round": None,
                "met_round": None,
                "updated_round": None,
                "has_risk": False,
                "risk_rounds": [],
            }

    return status


def calculate_max_rounds(goals: dict[str, list[dict[str, Any]]]) -> int:
    required_goal_count = len(goals["core_goals"]) + len(goals["secondary_goals"])
    return max(3, required_goal_count + 1)


def _add_product_compliance_goals(
    product: dict[str, Any],
    goals: dict[str, list[dict[str, Any]]],
) -> None:
    policy = product.get("compliance_goal_policy") or {}
    target_level = policy.get("target_level")
    compliance_goals = policy.get("goals", [])

    if not target_level or not compliance_goals:
        return

    bucket_by_level = {
        "core": "core_goals",
        "secondary": "secondary_goals",
        "observation": "observation_goals",
    }
    bucket = bucket_by_level.get(target_level)

    if bucket is None:
        raise ValueError(f"未知的产品合规目标层级：{target_level}")

    goals[bucket].extend(compliance_goals)


def _dedupe_goals(goals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []

    for goal in goals:
        goal_id = goal["goal_id"]
        if goal_id in seen:
            continue

        seen.add(goal_id)
        deduped.append(goal)

    return deduped


def _strength_order(strength: str) -> int:
    order = {
        "strong": 0,
        "medium": 1,
        "weak": 2,
    }
    return order.get(strength, 99)


if __name__ == "__main__":
    session = initialize_training_session()
    print(json.dumps(session, ensure_ascii=False, indent=2))
