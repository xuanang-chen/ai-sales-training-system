from __future__ import annotations

import importlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def reload_project_modules() -> None:
    module_names = [
        "core.llm_client",
        "core.initialization",
        "core.customer_simulator",
        "core.customer_simulator_next",
        "core.judge",
        "core.state_tracker",
        "core.termination",
        "core.workflow",
    ]

    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is not None:
            importlib.reload(module)


reload_project_modules()

from core.workflow import (  # noqa: E402
    start_training_session,
    submit_customer_answer,
    submit_opening_answer,
)

CUSTOMER_ANSWER_PROCESSING_STEPS = [
    {
        "module_key": "llm_judge",
        "module_label": "LLM Judge",
        "message": "评价回答中的目标证据、分数和风险。",
    },
    {
        "module_key": "state_tracker",
        "module_label": "状态追踪器",
        "message": "写入本轮评价结果并更新目标达成状态。",
    },
    {
        "module_key": "termination_judge",
        "module_label": "结束判断器",
        "message": "判断训练是否继续、结束或进入补充测试。",
    },
    {
        "module_key": "next_followup",
        "module_label": "再次追问器",
        "message": "选择下一轮追问目标，并生成新的客户问题。",
    },
]


st.set_page_config(
    page_title="AI 销售训练系统",
    layout="wide",
)


def reset_training() -> None:
    for key in [
        "session",
        "opening_question",
        "opening_result",
        "customer_answer_result",
        "opening_answer_text",
        "customer_answer_text",
        "processing_time_logs",
        "judge_time_logs",
    ]:
        st.session_state.pop(key, None)

    for key in list(st.session_state.keys()):
        if key.startswith("customer_answer_text_round_"):
            st.session_state.pop(key, None)


def is_deprecated_mvp_session(session: dict[str, Any] | None) -> bool:
    if not session:
        return False

    termination = session.get("termination_result", {})
    reason = str(termination.get("reason", ""))
    return (
        "product_basic_introduction" not in session.get("goal_status", {})
        or
        termination.get("end_type") == "mvp_single_round_complete"
        or "MVP 阶段规则" in reason
    )


def start_training() -> None:
    result = start_training_session()
    st.session_state.session = result["session"]
    st.session_state.session["processing_time_logs"] = []
    st.session_state.session["judge_time_logs"] = []
    st.session_state.opening_question = result["opening_question"]
    st.session_state.opening_result = None
    st.session_state.customer_answer_result = None
    st.session_state.opening_answer_text = ""
    st.session_state.customer_answer_text = ""
    st.session_state.processing_time_logs = []
    st.session_state.judge_time_logs = []


def render_training_context(session: dict[str, Any]) -> None:
    product = session["selected_product"]
    customer = session["selected_customer"]
    preferences = session["customer_preference_profile"]

    left, middle, right = st.columns([1.1, 1.1, 1])

    with left:
        st.markdown("**本次产品**")
        st.write(product["name"])
        st.caption(product.get("description", ""))
        render_product_comparison(product)

    with middle:
        st.markdown("**本次客户**")
        st.write(customer["role"])
        st.caption(customer.get("communication_style", ""))

    with right:
        st.markdown("**客户偏好**")
        for item in preferences:
            st.write(f"{item['name']}：{_strength_name(item['strength'])}")


def render_product_comparison(product: dict[str, Any]) -> None:
    selling_product = product.get("selling_product") or {}
    competitor_product = product.get("competitor_product") or {}
    comparison_context = product.get("comparison_context") or {}

    if not selling_product and not competitor_product:
        return

    with st.expander("查看主推产品与竞品资料", expanded=False):
        columns = st.columns(2)
        with columns[0]:
            st.markdown(f"**{selling_product.get('name', '主推产品 A')}**")
            st.caption(selling_product.get("role", "本次公司主推产品"))
            st.markdown("优势")
            for item in selling_product.get("advantages", []):
                st.write(f"- {item}")
            st.markdown("不足")
            for item in selling_product.get("disadvantages", []):
                st.write(f"- {item}")

        with columns[1]:
            st.markdown(f"**{competitor_product.get('name', '竞品 B')}**")
            st.caption(competitor_product.get("role", "客户当前常用或正在比较的竞品"))
            st.markdown("优势")
            for item in competitor_product.get("advantages", []):
                st.write(f"- {item}")
            st.markdown("不足")
            for item in competitor_product.get("disadvantages", []):
                st.write(f"- {item}")

        standards = comparison_context.get("comparison_standard", [])
        if standards:
            st.markdown("**客观比较标准**")
            for item in standards:
                st.write(f"- {item}")


def render_training_progress(session: dict[str, Any]) -> None:
    round_state = session.get("round_state", {})
    termination_state = session.get("termination_state", {})
    current_round = int(round_state.get("current_round", 0))
    max_rounds = int(round_state.get("max_rounds", 0))

    core_met, core_total = _goal_bucket_progress(session, "core_goals")
    secondary_met, secondary_total = _goal_bucket_progress(session, "secondary_goals")
    observation_met, observation_total = _goal_bucket_progress(
        session,
        "observation_goals",
    )

    left, middle, right, far_right = st.columns(4)
    left.metric("正式轮次", f"{current_round}/{max_rounds}")
    middle.metric("核心目标", f"{core_met}/{core_total}")
    right.metric("次级目标", f"{secondary_met}/{secondary_total}")
    far_right.metric("观察目标", f"{observation_met}/{observation_total}")

    if termination_state.get("post_core_supplemental_started"):
        probe_type = termination_state.get("post_core_next_probe_type")
        st.caption(
            f"当前处于补充测试阶段；下一轮补充方向：{_probe_type_name(probe_type)}。"
        )


def render_goal_table(session: dict[str, Any]) -> None:
    rows = []
    for goal_id, status in session["goal_status"].items():
        rows.append(
            {
                "目标": status["name"],
                "类型": _bucket_name(status["bucket"]),
                "状态": _status_name(status["status"]),
                "最高分": _empty_dash(status.get("best_score")),
                "最近分": _empty_dash(status.get("latest_score")),
                "达标轮次": _round_label(status.get("met_round")),
                "风险": "有" if status.get("has_risk") else "无",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_latest_result(result: dict[str, Any] | None) -> None:
    if not result:
        return

    with st.expander("查看上一轮评价与流程判断", expanded=False):
        render_judge_result(result)
        termination = result.get("termination_result", {})
        st.markdown("**流程判断**")
        st.info(termination.get("reason", "暂无流程判断。"))

        next_followup = result.get("next_followup_result")
        if next_followup and next_followup.get("probe_plan"):
            st.markdown("**下一轮追问计划**")
            st.json(
                {
                    "target_goal_ids": next_followup["probe_plan"].get(
                        "target_goal_ids",
                        [],
                    ),
                    "secondary_probe_goal_ids": next_followup["probe_plan"].get(
                        "secondary_probe_goal_ids",
                        [],
                    ),
                    "allow_observation_target": next_followup["probe_plan"].get(
                        "allow_observation_target",
                        False,
                    ),
                    "strategy": next_followup["probe_plan"].get("strategy"),
                }
            )


def render_opening_result(result: dict[str, Any] | None) -> None:
    if not result:
        return

    with st.expander("查看开场介绍评价", expanded=False):
        render_judge_result(result)
        termination = result.get("termination_result", {})
        st.markdown("**流程判断**")
        st.info(termination.get("reason", "开场阶段默认进入首次客户追问。"))

        next_followup = result.get("next_followup_result")
        if next_followup and next_followup.get("probe_plan"):
            st.markdown("**首次追问计划**")
            st.json(
                {
                    "target_goal_ids": next_followup["probe_plan"].get(
                        "target_goal_ids",
                        [],
                    ),
                    "secondary_probe_goal_ids": next_followup["probe_plan"].get(
                        "secondary_probe_goal_ids",
                        [],
                    ),
                    "strategy": next_followup["probe_plan"].get("strategy"),
                }
            )


def render_judge_result(result: dict[str, Any]) -> None:
    judge_result = result["judge_result"]
    st.markdown("**本轮评价总结**")
    st.info(judge_result.get("overall_comment") or "暂无总结")

    rows = []
    for item in judge_result["goal_evaluations"]:
        rows.append(
            {
                "目标": item["goal_name"],
                "范围": _scope_name(item["evaluation_scope"]),
                "状态": _status_name(item["status"]),
                "分数": item["score"],
                "是否达标": "是" if item["is_goal_met"] else "否",
                "证据": item["evidence"],
                "原因": item["reason"],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_risk_logs(session: dict[str, Any]) -> None:
    risk_logs = session.get("risk_logs", [])
    if not risk_logs:
        st.success("本轮未记录合规风险。")
        return

    st.warning("系统记录到合规风险，后续复盘报告会重点审查。")
    st.dataframe(pd.DataFrame(risk_logs), use_container_width=True, hide_index=True)


def render_question_logs(session: dict[str, Any]) -> None:
    question_logs = session.get("question_logs", [])
    if not question_logs:
        st.caption("暂无已完成问答。")
        return

    rows = []
    for item in question_logs:
        rows.append(
            {
                "轮次": _round_label(item.get("round"), item.get("round_type")),
                "客户问题": item.get("customer_question"),
                "主测目标": ", ".join(item.get("target_goal_ids", [])),
                "学员回答": item.get("learner_answer"),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def build_processing_steps() -> dict[str, dict[str, Any]]:
    return {
        step["module_key"]: {
            **step,
            "status": "pending",
            "duration_seconds": None,
        }
        for step in CUSTOMER_ANSWER_PROCESSING_STEPS
    }


def update_processing_steps(
    steps: dict[str, dict[str, Any]],
    event: dict[str, Any],
) -> None:
    module_key = str(event.get("module_key", ""))
    if not module_key:
        return

    step = steps.setdefault(
        module_key,
        {
            "module_key": module_key,
            "module_label": event.get("module_label", module_key),
            "message": "",
            "status": "pending",
            "duration_seconds": None,
        },
    )
    step["module_label"] = event.get("module_label", step["module_label"])
    step["status"] = event.get("status", step["status"])
    step["message"] = event.get("message", step["message"])

    if "duration_seconds" in event:
        step["duration_seconds"] = event["duration_seconds"]


def render_processing_steps(
    placeholder: Any,
    steps: dict[str, dict[str, Any]],
) -> None:
    ordered_steps = list(steps.values())
    active_step = next(
        (step for step in ordered_steps if step.get("status") == "running"),
        None,
    )
    failed_step = next(
        (step for step in ordered_steps if step.get("status") == "failed"),
        None,
    )
    completed_or_skipped = all(
        step.get("status") in {"completed", "skipped"}
        for step in ordered_steps
    )

    rows = [
        {
            "模块": step["module_label"],
            "状态": _processing_status_name(step["status"]),
            "耗时": _format_duration(step.get("duration_seconds")),
            "当前动作": step.get("message", ""),
        }
        for step in ordered_steps
    ]

    with placeholder.container():
        st.markdown("**回答提交后处理进度**")
        if failed_step:
            st.error(f"{failed_step['module_label']} 执行失败。")
        elif active_step:
            st.info(f"正在执行：{active_step['module_label']}")
        elif completed_or_skipped:
            st.success("本轮处理完成。")
        else:
            st.info("已收到回答，等待开始处理。")

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def append_processing_time_log(round_number: int, event: dict[str, Any]) -> None:
    if event.get("scope") == "llm_judge_internal":
        append_judge_time_log(round_number, event)
        return

    status = str(event.get("status", ""))
    if status not in {"completed", "skipped", "failed"}:
        return

    record = {
        "记录时间": datetime.now().strftime("%H:%M:%S"),
        "轮次": _round_label(round_number),
        "模块": event.get("module_label", event.get("module_key", "")),
        "状态": _processing_status_name(status),
        "耗时": _format_duration(event.get("duration_seconds")),
        "说明": event.get("message", ""),
    }

    logs = st.session_state.setdefault("processing_time_logs", [])
    logs.append(record)

    current_session = st.session_state.get("session")
    if isinstance(current_session, dict):
        current_session.setdefault("processing_time_logs", []).append(record.copy())


def append_judge_time_log(round_number: int, event: dict[str, Any]) -> None:
    status = str(event.get("status", ""))
    if status not in {"completed", "skipped", "failed"}:
        return

    record = {
        "记录时间": datetime.now().strftime("%H:%M:%S"),
        "轮次": _round_label(round_number),
        "内部步骤": event.get("module_label", event.get("module_key", "")),
        "状态": _processing_status_name(status),
        "耗时": _format_duration(event.get("duration_seconds")),
        "说明": event.get("message", ""),
    }

    logs = st.session_state.setdefault("judge_time_logs", [])
    logs.append(record)

    current_session = st.session_state.get("session")
    if isinstance(current_session, dict):
        current_session.setdefault("judge_time_logs", []).append(record.copy())


def render_processing_time_logs() -> None:
    logs = st.session_state.get("processing_time_logs", [])
    if not logs:
        current_session = st.session_state.get("session")
        if isinstance(current_session, dict):
            logs = current_session.get("processing_time_logs", [])

    if not logs:
        st.caption("暂无模块耗时记录。提交一轮客户追问回答后，这里会保留每一步的耗时。")
        return

    st.markdown("**模块耗时记录**")
    st.dataframe(
        pd.DataFrame(logs),
        use_container_width=True,
        hide_index=True,
    )


def render_judge_time_logs() -> None:
    logs = st.session_state.get("judge_time_logs", [])
    if not logs:
        current_session = st.session_state.get("session")
        if isinstance(current_session, dict):
            logs = current_session.get("judge_time_logs", [])

    if not logs:
        st.caption("暂无 LLM Judge 内部耗时记录。提交一轮客户追问回答后，这里会保留 Judge 内部三步耗时。")
        return

    st.markdown("**LLM Judge 内部耗时记录**")
    st.dataframe(
        pd.DataFrame(logs),
        use_container_width=True,
        hide_index=True,
    )


def _processing_status_name(status: str) -> str:
    return {
        "pending": "待开始",
        "running": "进行中",
        "completed": "已完成",
        "skipped": "已跳过",
        "failed": "失败",
    }.get(status, status)


def _format_duration(duration_seconds: Any) -> str:
    if duration_seconds is None:
        return "-"

    try:
        return f"{float(duration_seconds):.2f}s"
    except (TypeError, ValueError):
        return "-"


def _bucket_name(bucket: str) -> str:
    return {
        "core_goals": "核心目标",
        "secondary_goals": "次级目标",
        "observation_goals": "观察目标",
    }.get(bucket, bucket)


def _status_name(status: str) -> str:
    return {
        "untested": "未测试",
        "not_observed": "未观察到",
        "not_met": "未达标",
        "partial": "部分满足",
        "met": "已达标",
        "risk": "有风险",
    }.get(status, status)


def _scope_name(scope: str) -> str:
    return {
        "primary": "主评",
        "secondary": "兼评",
        "natural": "自然体现",
    }.get(scope, scope)


def _strength_name(strength: str) -> str:
    return {
        "strong": "强",
        "medium": "中",
        "weak": "弱",
    }.get(strength, strength)


def _probe_type_name(probe_type: str | None) -> str:
    return {
        "secondary": "次级目标",
        "observation": "观察目标",
    }.get(probe_type or "", "未进入补充测试")


def _goal_bucket_progress(session: dict[str, Any], bucket: str) -> tuple[int, int]:
    statuses = [
        status
        for status in session.get("goal_status", {}).values()
        if status.get("bucket") == bucket
    ]
    met_count = sum(1 for status in statuses if status.get("status") == "met")
    return met_count, len(statuses)


def _empty_dash(value: Any) -> Any:
    return "-" if value is None else value


def _round_label(round_number: Any, round_type: str | None = None) -> Any:
    if round_number is None:
        return "-"

    if round_type == "opening":
        return "开场"

    try:
        normalized_round = int(round_number)
    except (TypeError, ValueError):
        return round_number

    if normalized_round == 0:
        return "开场"

    return normalized_round


st.title("AI 销售训练系统")

top_left, top_right = st.columns([1, 1])
with top_left:
    if st.button("开始新的训练", type="primary", use_container_width=True):
        start_training()
with top_right:
    if st.button("重置当前训练", use_container_width=True):
        reset_training()
        st.rerun()

session = st.session_state.get("session")

if is_deprecated_mvp_session(session):
    reset_training()
    st.warning("检测到旧版本训练状态，已自动清空。请点击“开始新的训练”重新测试。")
    st.stop()

if not session:
    st.info("点击“开始新的训练”后，系统会随机抽取产品、客户和训练目标。")
    st.stop()

render_training_context(session)
render_training_progress(session)

with st.expander("查看本次训练目标", expanded=False):
    render_goal_table(session)

st.divider()

if session["stage"] == "opening_questioned":
    st.subheader("第一步：产品开场介绍")
    st.chat_message("assistant").write(
        st.session_state.opening_question["customer_question"]
    )

    with st.form("opening_form"):
        opening_answer = st.text_area(
            "你的开场介绍",
            key="opening_answer_text",
            height=160,
            placeholder="请面向本次客户，简要介绍本次产品。",
        )
        submitted = st.form_submit_button("提交开场介绍", type="primary")

    if submitted:
        if not opening_answer.strip():
            st.warning("请先输入开场介绍。")
        else:
            try:
                progress_placeholder = st.empty()
                processing_steps = build_processing_steps()
                render_processing_steps(progress_placeholder, processing_steps)

                def handle_progress(event: dict[str, Any]) -> None:
                    if event.get("scope") != "llm_judge_internal":
                        update_processing_steps(processing_steps, event)
                    append_processing_time_log(0, event)
                    render_processing_steps(progress_placeholder, processing_steps)

                st.session_state.opening_result = submit_opening_answer(
                    session,
                    opening_answer,
                    progress_callback=handle_progress,
                )
                render_processing_steps(progress_placeholder, processing_steps)
                st.rerun()
            except Exception as exc:  # pragma: no cover - Streamlit display path
                st.error(f"生成客户追问时出错：{exc}")

elif session["stage"] == "customer_questioned":
    question = session["pending_question"]
    round_number = question["round"]

    if st.session_state.get("customer_answer_result"):
        render_latest_result(st.session_state.get("customer_answer_result"))
    else:
        render_opening_result(st.session_state.get("opening_result"))

    st.subheader(f"客户第 {round_number} 轮追问")
    st.chat_message("assistant").write(question["customer_question"])

    with st.expander("查看本轮追问后台信息", expanded=False):
        st.json(
            {
                "target_goal_ids": question["target_goal_ids"],
                "secondary_probe_goal_ids": question["secondary_probe_goal_ids"],
                "question_intent": question["question_intent"],
                "difficulty": question["difficulty"],
            }
        )

    answer_key = f"customer_answer_text_round_{round_number}"
    with st.form(f"customer_answer_form_round_{round_number}"):
        customer_answer = st.text_area(
            "你的回答",
            key=answer_key,
            height=180,
            placeholder="请回答客户的追问。",
        )
        submitted = st.form_submit_button("提交回答", type="primary")

    if submitted:
        if not customer_answer.strip():
            st.warning("请先输入你的回答。")
        else:
            try:
                progress_placeholder = st.empty()
                processing_steps = build_processing_steps()
                render_processing_steps(progress_placeholder, processing_steps)

                def handle_progress(event: dict[str, Any]) -> None:
                    if event.get("scope") != "llm_judge_internal":
                        update_processing_steps(processing_steps, event)
                    append_processing_time_log(round_number, event)
                    render_processing_steps(progress_placeholder, processing_steps)

                st.session_state.customer_answer_result = submit_customer_answer(
                    session,
                    customer_answer,
                    progress_callback=handle_progress,
                )
                render_processing_steps(progress_placeholder, processing_steps)
                st.rerun()
            except Exception as exc:  # pragma: no cover - Streamlit display path
                st.error(f"评价回答时出错：{exc}")

    render_processing_time_logs()
    render_judge_time_logs()

    with st.expander("查看已完成问答记录", expanded=False):
        render_question_logs(session)

elif session["stage"] == "ended":
    st.subheader("训练已结束")
    termination = session.get("termination_result", {})
    st.success(termination.get("reason", "本次训练已结束。"))

    customer_answer_result = st.session_state.get("customer_answer_result")
    if customer_answer_result:
        render_judge_result(customer_answer_result)

    st.markdown("**能力状态表**")
    render_goal_table(session)

    render_processing_time_logs()
    render_judge_time_logs()

    st.markdown("**风险记录**")
    render_risk_logs(session)

    with st.expander("查看已完成问答记录", expanded=False):
        render_question_logs(session)

    with st.expander("查看完整 session", expanded=False):
        st.json(session)

else:
    st.warning(f"当前状态暂未接入前端：{session['stage']}")
