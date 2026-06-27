from __future__ import annotations

from typing import Any


def build_customer_brief(customer: dict[str, Any]) -> dict[str, Any]:
    return {
        "customer_id": customer.get("customer_id"),
        "role": customer.get("role"),
        "industry": customer.get("industry"),
        "work_context": customer.get("work_context"),
        "decision_factors": customer.get("decision_factors", []),
        "communication_style": customer.get("communication_style"),
    }

