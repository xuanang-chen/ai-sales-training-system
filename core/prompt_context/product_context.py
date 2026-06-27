from __future__ import annotations

from typing import Any


def build_product_for_customer_question(product: dict[str, Any]) -> dict[str, Any]:
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
    }


def build_product_for_judge(product: dict[str, Any]) -> dict[str, Any]:
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
