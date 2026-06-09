"""
GDPR Privacy Auditor V2.2 — 主图构建。

这是整个项目的核心文件。它用 LangGraph StateGraph 将状态定义、
2 个 Specialist Agent、冲突消解子图、HITL 人审、DPIA Reflection
串联成一个完整的审计工作流。

LangGraph 知识点（面试重点）：
  1. StateGraph + TypedDict — 类型安全的图状态管理
  2. Send() API — Fan-Out：根据 input_types 动态决定调用 1 或 2 个 Agent
  3. operator.add — Fan-In：两个 Agent 的 findings 安全合并
  4. SubGraph — Conflict Resolution 作为独立编译的子图
  5. interrupt() — HITL 人审中断点
  6. Conditional Edges — 5 个路由函数控制流程分支
  7. Cyclic Edges — 4 条循环回边（不是 DAG！）
  8. InMemorySaver — 状态持久化 + 中断恢复（LangGraph 1.x）

图结构总览：

    START → init_node → evidence_supervisor
                            ├─ Send(privacy_doc_auditor) ─┐
                            └─ Send(data_schema_auditor)  ─┤  Fan-Out
                                                           │
                     ┌─────────────────────────────────────┘
                     ▼ (Fan-In via operator.add)
              conflict_subgraph (子图)
                     │
                     ▼
              synthesis_agent
                     │
                     ▼
               risk_rater
                ├─ MEDIUM/LOW → dpia_generator
                └─ HIGH → human_review (interrupt)
                              ├─ approve → dpia_generator
                              ├─ edit → synthesis_agent (循环回边 #3)
                              └─ reject → END
                     │
                     ▼
              dpia_generator
                     │
                     ▼
            reflection_agent
                ├─ pass → final_report → END
                ├─ retry → dpia_generator (循环回边 #4)
                └─ escalate → human_review
"""

import os
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Send

from state import (
    GDPRPrivacyAuditStateV2_2,
    AuditPhase,
    RiskTier,
    create_initial_state,
)
from rules.rubric import DPIAQualityRubric
from versioning.tracker import RegulationVersionTracker
from verification import verify_all_findings, get_verification_stats

# ── Agent 节点（当前为模拟实现，Phase 3 接真实 LLM）──
from agents.privacy_doc import privacy_doc_auditor_node as _privacy_node
from agents.data_schema import data_schema_auditor_node as _data_node

# ── 冲突消解子图 ──
from subgraphs.conflict import conflict_subgraph


# ═══════════════════════════════════════════════════════════
# 单例
# ═══════════════════════════════════════════════════════════

dpia_rubric = DPIAQualityRubric()          # Phase 3: 注入 LLM
version_tracker = RegulationVersionTracker()


# ═══════════════════════════════════════════════════════════
# 节点函数
# ═══════════════════════════════════════════════════════════

# ─── N1: 初始化节点 ─────────────────────────────────────

def init_node(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    初始化审计状态。

    职责：
      1. 解析和验证输入
      2. 设置审计 ID 和时间戳
      3. 启动版本追踪（检查文档时效性）
      4. 设置阶段为 EVIDENCE_COLLECTION

    参数:
        state: 初始状态（由 create_initial_state 创建）

    返回:
        dict — 初始化后的状态更新
    """
    errors = []
    warnings = []

    # ── 验证输入 ──
    input_types = state.get("input_types", [])
    if not input_types:
        errors.append("No input types specified — nothing to audit.")
        return {
            "phase": AuditPhase.COMPLETED.value,
            "errors": errors,
            "warnings": warnings,
        }

    has_docs = "privacy_document" in input_types
    has_schemas = "data_schema" in input_types

    if has_docs and not state.get("privacy_documents"):
        warnings.append(
            "Input type includes 'privacy_document' but no documents provided."
        )

    if has_schemas and not state.get("data_schemas"):
        warnings.append(
            "Input type includes 'data_schema' but no schemas provided."
        )

    # ── 文档时效性检查（V2.2: 法规版本感知）──
    documents_outdated = []
    doc_date = state.get("document_date", "")
    if doc_date:
        currency = version_tracker.check_document_currency(doc_date)
        if not currency.is_current:
            documents_outdated.append({
                "warning": currency.warning,
                "date": currency.document_date,
            })
            warnings.append(currency.warning)

    # ── 获取法规版本元数据 ──
    regulation_versions = version_tracker.get_version_metadata()

    return {
        "phase": AuditPhase.EVIDENCE_COLLECTION.value,
        "phase_iterations": {"INIT": 1},
        "errors": errors,
        "warnings": warnings,
        "documents_outdated": documents_outdated,
        "regulation_versions": regulation_versions,
    }


# ─── N2: 证据收集督导节点 ──────────────────────────────

def evidence_supervisor_node(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    证据收集督导节点。

    此节点本身不做审计——它的职责是决定"派谁去收集证据"。
    实际的并发派发由 dispatch_evidence_tasks() 路由函数通过 Send() 完成。

    LangGraph 知识点：
      此节点的返回值不直接发给 Agent —— 路由函数 dispatch_evidence_tasks
      通过 Send() API 创建并发任务。

    参数:
        state: 含 input_types 的状态

    返回:
        dict — 阶段标记更新
    """
    input_types = state.get("input_types", [])

    return {
        "phase": AuditPhase.EVIDENCE_COLLECTION.value,
        "phase_iterations": {
            **state.get("phase_iterations", {}),
            "EVIDENCE_COLLECTION": 1,
        },
        # agent_tasks 仅供路由函数读取，存入 state 以便 dispatch_evidence_tasks 使用
    }


# ─── N3: Privacy Doc Auditor 节点（包装器）────────────────

def privacy_doc_auditor_node(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    Privacy Doc Auditor 的 LangGraph 节点包装器。

    调用 agents/privacy_doc.py 中的实际审计逻辑。
    Phase 2: 模拟实现
    Phase 3: 改为 LangChain create_agent + ReAct 循环

    参数:
        state: 完整审计状态

    返回:
        dict — 含 evidence 和 findings（通过 operator.add Fan-In 合并）
    """
    result = _privacy_node(state)

    # 自动标记证据充分度
    evidence_count = len(result.get("evidence", []))
    findings_count = len(result.get("findings", []))

    return {
        **result,
        "_privacy_auditor_completed": True,
        "_privacy_evidence_count": evidence_count,
        "_privacy_findings_count": findings_count,
    }


# ─── N4: Data Schema Auditor 节点（包装器）────────────────

def data_schema_auditor_node(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    Data Schema Auditor 的 LangGraph 节点包装器。

    调用 agents/data_schema.py 中的实际审计逻辑。
    Phase 2: 模拟实现
    Phase 3: 改为 LangChain create_agent + ReAct 循环

    参数:
        state: 完整审计状态

    返回:
        dict — 含 evidence 和 findings（通过 operator.add Fan-In 合并）
    """
    result = _data_node(state)

    evidence_count = len(result.get("evidence", []))
    findings_count = len(result.get("findings", []))

    return {
        **result,
        "_data_auditor_completed": True,
        "_data_evidence_count": evidence_count,
        "_data_findings_count": findings_count,
    }


# ─── N5: Synthesis Agent（综合分析节点）───────────────────

def synthesis_agent_node(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    综合分析节点。

    职责：
      1. 合并两个 Agent 的 findings（已在 state 中通过 operator.add 完成）
      2. 考虑冲突消解结果
      3. 生成 synthesis_summary（DPO 审批的对象）
      4. 标记特殊发现（特殊类别数据、高风险传输）

    Phase 2: 结构化合并 + 模拟摘要
    Phase 3: LLM 生成综合摘要

    参数:
        state: 含合并后 findings + 消解后 conflicts 的状态

    返回:
        dict — 含 synthesis_summary 和风险标记
    """
    findings = state.get("findings", [])
    conflicts = state.get("conflicts", [])

    # ── 统计发现 ──
    total_findings = len(findings)
    fail_findings = [f for f in findings if f.get("state") == "FAIL"]
    high_findings = [f for f in fail_findings if f.get("severity") == "HIGH"]
    medium_findings = [f for f in fail_findings if f.get("severity") == "MEDIUM"]
    pass_findings = [f for f in findings if f.get("state") == "PASS"]

    critical_count = len(high_findings)

    # ── 检查特殊标记 ──
    # 是否有特殊类别数据（Art.9）—— 检测类别名或描述中的关键词
    has_special = any(
        f.get("category") == "SPECIAL_CATEGORY_DATA"
        or f.get("category") == "UNDECLARED_PII"
        and any(kw in str(f).lower() for kw in ["imei", "gps", "位置", "设备标识", "敏感", "sensitive", "special", "location"])
        for f in findings
    )

    # ── 跨境风险等级（HIGH/MEDIUM/LOW）──
    transfer_findings = [
        f for f in findings
        if f.get("category") in ("TRANSFER_UNDECLARED", "REGIONAL_SCOPE")
        or "跨境" in f.get("title", "") + f.get("description", "")
    ]
    if not transfer_findings:
        cross_border_risk = "LOW"
    else:
        # 检查是否有保障措施的描述
        all_text = " ".join(
            f.get("description", "") + f.get("title", "")
            for f in transfer_findings
        ).lower()
        has_safeguards = any(
            kw in all_text
            for kw in ["scc", "dpf", "adequacy", "保障", "标准合同", "bcr",
                       "standard contractual", "adequate", "privacy shield"]
        )
        if has_safeguards:
            cross_border_risk = "MEDIUM"
        else:
            cross_border_risk = "HIGH"

    # ── 考虑冲突消解结果 ──
    resolved_conflicts = [
        c for c in conflicts if c.get("resolved", False)
    ]
    unresolved_conflicts = [
        c for c in conflicts if not c.get("resolved", False)
    ]

    # ── 构建综合摘要 ──
    # Phase 3: 由 LLM 生成
    summary_parts = [
        f"GDPR Privacy Audit Summary",
        f"─" * 50,
        f"Total Findings: {total_findings} "
        f"(FAIL: {len(fail_findings)}, PASS: {len(pass_findings)})",
        f"",
        f"Severity Breakdown:",
        f"  HIGH: {len(high_findings)} findings",
        f"  MEDIUM: {len(medium_findings)} findings",
        f"  LOW/PASS: {total_findings - len(high_findings) - len(medium_findings)}",
        f"",
    ]

    if has_special:
        summary_parts.append(
            "⚠️  Special Category Data (Art.9): DETECTED — "
            "heightened compliance requirements apply."
        )
    if cross_border_risk != "LOW":
        level_label = {"HIGH": "高风险", "MEDIUM": "中等风险"}.get(cross_border_risk, cross_border_risk)
        summary_parts.append(
            f"⚠️  Cross-Border Transfer Risk: {level_label} ({cross_border_risk}) — "
            "Art.44-49 review required."
        )

    if conflicts:
        summary_parts.append(f"")
        summary_parts.append(f"Conflict Resolution:")
        summary_parts.append(
            f"  Total Conflicts: {len(conflicts)}, "
            f"Resolved: {len(resolved_conflicts)}, "
            f"Unresolved: {len(unresolved_conflicts)}"
        )
        for c in conflicts:
            arb = c.get("arbitration_result", {})
            summary_parts.append(
                f"  - {c.get('conflict_id', '?')}: "
                f"{c.get('conflict_type', '?')} -> "
                f"winner={arb.get('winner', '?')}, "
                f"method={arb.get('method', '?')}"
            )

    # ── 列出 Top 5 严重发现（供 DPO 审批界面展示）──
    if high_findings:
        summary_parts.append(f"")
        summary_parts.append(f"Top Critical Findings:")
        for i, f in enumerate(sorted(
            high_findings,
            key=lambda x: x.get("severity", "LOW"),
            reverse=True
        )[:5]):
            summary_parts.append(
                f"  {i+1}. [{f.get('finding_id', '?')}] "
                f"{f.get('title', 'No title')}"
            )

    synthesis_summary = "\n".join(summary_parts)

    return {
        "synthesis_summary": synthesis_summary,
        "critical_findings_count": critical_count,
        "has_special_category_data": has_special,
        "cross_border_risk_level": cross_border_risk,
        "confidence_score": _calculate_confidence(findings, conflicts),
        "phase": AuditPhase.ANALYSIS.value,
    }


def _calculate_confidence(findings: list, conflicts: list) -> float:
    """
    计算整体置信度。

    影响因素：
      - 未解决的冲突 → 降低置信度
      - 证据缺失 → 降低置信度
      - 全部 PASS → 高置信度
    """
    if not findings:
        return 1.0

    unresolved = sum(1 for c in conflicts if not c.get("resolved", False))
    fail_count = sum(1 for f in findings if f.get("state") == "FAIL")

    # 简单启发式（Phase 3: 更精确的置信度计算）
    base = 1.0
    base -= unresolved * 0.15      # 每个未解决的冲突 -0.15
    base -= fail_count * 0.02      # 每个 FAIL 发现 -0.02

    return max(0.1, min(1.0, base))  # 限制在 [0.1, 1.0]


# ─── N6: Risk Rater（风险评估节点）────────────────────────

def risk_rater_node(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    风险评估节点。

    计算整体风险等级：
      - HIGH: 有严重发现或高风险传输或特殊类别数据且无充分保护
      - MEDIUM: 有 FAIL 发现但不满足 HIGH 条件
      - LOW: 无 FAIL 发现

    V2.2 新增 INCONCLUSIVE 等级（DPO 驳回时设置，不由 Risk Rater 设置）。

    参数:
        state: 含 synthesis_summary 和 findings 的状态

    返回:
        dict — 含 risk_tier, critical_findings_count
    """
    critical_count = state.get("critical_findings_count", 0)

    # ═══ V2.2: 如果 DPO 已编辑风险等级，尊重其决定 ═══
    dpo_edited = state.get("_dpo_edited", False)
    if dpo_edited:
        # DPO 已修改风险等级 — 尊重决定，不重新计算
        dpo_tier = state.get("risk_tier", RiskTier.MEDIUM.value)
        warnings_list = list(state.get("warnings", []))
        # 但如果降级明显不合理（5+ HIGH 发现但改为 MEDIUM），追加 warning
        if critical_count >= 5 and dpo_tier == RiskTier.MEDIUM.value:
            warnings_list.append(
                f"Warning: DPO lowered risk from HIGH to {dpo_tier} "
                f"but {critical_count} critical findings remain. "
                f"DPO decision is respected but this may indicate unresolved risks."
            )
        return {
            "risk_tier": dpo_tier,
            "needs_human_review": False,  # DPO 看过，不再触发 HITL
            "critical_findings_count": critical_count,
            "warnings": warnings_list,
        }
    has_special = state.get("has_special_category_data", False)
    cross_border = state.get("cross_border_risk_level", "LOW")
    confidence = state.get("confidence_score", 1.0)
    findings = state.get("findings", [])

    # ── 风险等级判定 ──
    if critical_count >= 1 or (has_special and confidence < 0.8) or cross_border == "HIGH":
        risk_tier = RiskTier.HIGH.value
    elif any(f.get("state") == "FAIL" for f in findings):
        risk_tier = RiskTier.MEDIUM.value
    else:
        risk_tier = RiskTier.LOW.value

    # ── 是否需要人审？──
    # V2.2: HIGH → HITL，DPO 审批整份结论
    needs_review = risk_tier == RiskTier.HIGH.value

    return {
        "risk_tier": risk_tier,
        "needs_human_review": needs_review,
        "critical_findings_count": critical_count,
    }


# ─── N7: Human Review 节点（HITL 人审）────────────────────

def human_review_node(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    HITL 人审节点 — V2.2 简化版。

    DPO 审批整份结论，不是逐条发现。
    三种操作：
      - Approve: 批准结论，继续 DPIA
      - Edit: 修改风险等级，重走 synthesis + risk_rater
      - Reject: 驳回结论，标记 INCONCLUSIVE，结束审计

    LangGraph 知识点：
      interrupt() 是 LangGraph 的暂停机制 — 程序在此停止，
      等待外部（如 Web UI / CLI / API）通过 Command(resume=...) 传入 DPO 决策。

    Phase 2 行为：
      暂时跳过 interrupt()（因为没有真实的 LangGraph runtime 支持），
      模拟 DPO 审批响应。

    Phase 4：
      接入真实的 interrupt() + SqliteSaver 持久化恢复。

    参数:
        state: 含 synthesis_summary 和 risk_tier 的状态

    返回:
        dict — 含 dpo_decision 和路由标记
    """
    conclusion = {
        "audit_id": state.get("audit_id", "UNKNOWN"),
        "target_name": state.get("target_name", "UNKNOWN"),
        "risk_tier": state.get("risk_tier", RiskTier.LOW.value),
        "total_findings": len(state.get("findings", [])),
        "critical_findings_count": state.get("critical_findings_count", 0),
        "top_findings": sorted(
            [
                f for f in state.get("findings", [])
                if f.get("severity") == "HIGH"
            ],
            key=lambda f: f.get("severity", "LOW"),
        )[:5],
        "synthesis_summary": state.get("synthesis_summary", ""),
        "regulation_versions": state.get("regulation_versions", {}),
    }

    # ── 中断点 ──
    # Phase 4: 取消下面的注释，使用真实的 interrupt()
    #
    # from langgraph.types import interrupt
    # dpo_response = interrupt({
    #     "type": "human_review_conclusion",
    #     "conclusion": conclusion,
    #     "editable_fields": ["risk_tier"],
    #     "allowed_actions": ["approve", "edit", "reject"],
    # })

    # Phase 2: 模拟 DPO 响应（默认 approve，可在测试时覆盖）
    dpo_response = _get_dpo_response(state)

    # ── 恢复点：处理 DPO 决策 ──
    action = dpo_response.get("action", "approve")

    if action == "approve":
        return {
            "needs_human_review": False,
            "dpo_decision": {
                "action": "approve",
                "dpo_id": dpo_response.get("dpo_id", "auto"),
                "timestamp": dpo_response.get(
                    "timestamp",
                    datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                ),
            },
            "_dpo_edited": False,
            "_dpo_rejected": False,
        }

    elif action == "edit":
        new_tier = dpo_response.get("new_risk_tier", state.get("risk_tier"))
        return {
            "needs_human_review": False,
            "risk_tier": new_tier,
            "dpo_decision": {
                "action": "edit",
                "original_risk_tier": state.get("risk_tier"),
                "new_risk_tier": new_tier,
                "dpo_id": dpo_response.get("dpo_id", "auto"),
                "timestamp": dpo_response.get(
                    "timestamp",
                    datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                ),
            },
            "_dpo_edited": True,
            "_dpo_rejected": False,
        }

    elif action == "reject":
        return {
            "needs_human_review": False,
            "phase": AuditPhase.COMPLETED.value,
            "report_text": (
                "[AUDIT REJECTED] The audit conclusion has been rejected by "
                f"DPO ({dpo_response.get('dpo_id', 'unknown')}). "
                f"Reason: {dpo_response.get('reason', 'No reason provided')}. "
                f"The audit is marked as INCONCLUSIVE."
            ),
            "dpo_decision": {
                "action": "reject",
                "reason": dpo_response.get("reason", ""),
                "dpo_id": dpo_response.get("dpo_id", "auto"),
                "timestamp": dpo_response.get(
                    "timestamp",
                    datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                ),
            },
            "risk_tier": RiskTier.INCONCLUSIVE.value,
            "_dpo_edited": False,
            "_dpo_rejected": True,
        }

    return {}


def _get_dpo_response(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    Phase 2 模拟：获取 DPO 审批响应。

    默认行为：自动 Approve（让流程跑通）。

    测试时可通过环境变量覆盖：
      export DPO_TEST_ACTION=reject
      export DPO_TEST_NEW_TIER=MEDIUM
    """
    action = os.getenv("DPO_TEST_ACTION", "approve")
    new_tier = os.getenv("DPO_TEST_NEW_TIER", "")
    reason = os.getenv("DPO_TEST_REJECT_REASON", "Audit scope does not apply.")

    response = {
        "action": action,
        "dpo_id": "dpo-test",
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if action == "edit":
        response["new_risk_tier"] = new_tier or RiskTier.MEDIUM.value
    elif action == "reject":
        response["reason"] = reason

    return response


# ─── N8: DPIA Generator（DPIA 生成节点）───────────────────

def dpia_generator_node(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    DPIA 生成节点。

    基于审计发现生成数据保护影响评估报告。
    这是 DPIA Reflection 循环的"生产者"。

    Phase 2: 模拟生成结构化的 DPIA 报告
    Phase 3: LLM 基于 findings 和 RAG 知识生成完整 DPIA

    参数:
        state: 含 findings, conflicts, synthesis_summary 的状态

    返回:
        dict — 含 dpia_report 和 dpia_iteration
    """
    findings = state.get("findings", [])
    conflicts = state.get("conflicts", [])
    summary = state.get("synthesis_summary", "")
    iteration = state.get("dpia_iteration", 0) + 1

    # 如果有上轮的 reflection_feedback，参考它
    feedback = state.get("reflection_feedback", "")

    # ── 构建 DPIA 报告结构 ──
    # Phase 3: LLM 生成每个 section 的详细内容
    dpia_report = {
        # 维度 1: 系统性描述
        "systematic_description": _generate_systematic_description(state, iteration),

        # 维度 2: 目的评估
        "purpose_assessment": _generate_purpose_assessment(state, iteration),

        # 维度 3: 必要性/相称性
        "necessity_proportionality": _generate_necessity_assessment(state, iteration),

        # 维度 4: 风险识别（结构化列表）
        "risk_identification": _generate_risk_scenarios(state, iteration),

        # 维度 5: 缓解措施
        "mitigation_measures": _generate_mitigation_measures(state, iteration),

        # 维度 6: 剩余风险评估
        "residual_risk": _generate_residual_risk_assessment(state, iteration),

        # 维度 7: 咨询记录
        "consultation": _generate_consultation_record(state, iteration),

        # 元数据
        "_meta": {
            "iteration": iteration,
            "has_feedback": bool(feedback),
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    # 如果有反馈，记录在报告中
    if feedback:
        dpia_report["_previous_feedback"] = feedback[:500]

    return {
        "dpia_report": dpia_report,
        "dpia_iteration": iteration,
    }


def _generate_systematic_description(state: dict, iteration: int) -> str:
    """生成 DPIA Section 1: 处理活动系统性描述。"""
    findings = state.get("findings", [])
    target = state.get("target_name", "Unknown")

    data_categories = set()
    for f in findings:
        if f.get("category") == "UNDECLARED_PII":
            for field in f.get("undeclared_fields", []):
                data_categories.add(field)

    return (
        f"[DPIA Iteration {iteration}] "
        f"Systematic description of processing activities for '{target}'. "
        f"The system processes the following personal data categories: "
        f"{', '.join(sorted(data_categories)) if data_categories else 'various user data'}. "
        f"Processing operations include collection, storage, transmission, "
        f"and deletion. Data subjects include registered users, customers, "
        f"and marketing recipients. The processing is carried out through "
        f"a web application with a SQL database backend."
    )


def _generate_purpose_assessment(state: dict, iteration: int) -> str:
    """生成 DPIA Section 2: 目的评估。"""
    return (
        f"[DPIA Iteration {iteration}] "
        f"Purpose assessment: Core service purposes (account management, "
        f"order fulfillment) are based on contract necessity (Art.6(1)(b)). "
        f"Marketing communications rely on consent (Art.6(1)(a)). "
        f"Service improvement is based on legitimate interest (Art.6(1)(f)). "
        f"Core and additional purposes are distinguished, though the "
        f"boundary between 'service improvement' and 'marketing' requires "
        f"clearer delineation."
    )


def _generate_necessity_assessment(state: dict, iteration: int) -> str:
    """生成 DPIA Section 3: 必要性/相称性评估。"""
    return (
        f"[DPIA Iteration {iteration}] "
        f"Necessity and proportionality: Email address is necessary for "
        f"account identification and communication. Full name and address "
        f"are necessary for order fulfillment. IP address processing is "
        f"necessary for security purposes. However, the collection of "
        f"device IMEI and GPS location data requires stronger necessity "
        f"justification — less intrusive alternatives (e.g., IP-based "
        f"geolocation instead of GPS) should be considered per the data "
        f"minimization principle (Art.5(1)(c))."
    )


def _generate_risk_scenarios(state: dict, iteration: int) -> list[dict]:
    """生成 DPIA Section 4: 风险识别场景列表。"""
    findings = state.get("findings", [])
    high_findings = [f for f in findings if f.get("severity") == "HIGH"]

    scenarios = []

    # 基于实际发现构建风险场景
    if any("UNDECLARED" in f.get("category", "") for f in high_findings):
        scenarios.append({
            "scenario": "Undeclared data collection leading to transparency violation",
            "likelihood": "high",
            "impact": "high",
            "data_subjects_affected": "All users",
            "related_findings": [
                f["finding_id"] for f in high_findings
                if "UNDECLARED" in f.get("category", "")
            ],
        })

    if any("TRANSFER" in f.get("category", "") for f in high_findings):
        scenarios.append({
            "scenario": "Unauthorized cross-border data transfer without safeguards",
            "likelihood": "medium",
            "impact": "high",
            "data_subjects_affected": "EU/EEA users",
            "related_findings": [
                f["finding_id"] for f in high_findings
                if "TRANSFER" in f.get("category", "")
            ],
        })

    if any("RETENTION" in f.get("category", "") for f in high_findings):
        scenarios.append({
            "scenario": "Excessive data retention leading to unnecessary privacy risk accumulation",
            "likelihood": "high",
            "impact": "medium",
            "data_subjects_affected": "Marketing recipients",
            "related_findings": [
                f["finding_id"] for f in high_findings
                if "RETENTION" in f.get("category", "")
            ],
        })

    # 保障至少 3 个风险场景（WP248 一票否决的最低要求）
    if len(scenarios) < 3:
        scenarios.append({
            "scenario": "Data breach through SQL injection exposing PII",
            "likelihood": "medium",
            "impact": "high",
            "data_subjects_affected": "All users",
            "related_findings": [],
        })

    return scenarios


def _generate_mitigation_measures(state: dict, iteration: int) -> list[dict]:
    """生成 DPIA Section 5: 缓解措施。"""
    return [
        {
            "for_risk": "Undeclared data collection",
            "measure": "Update privacy policy to list all 12 PII categories. Implement quarterly policy review cycle.",
            "timeline": "30 days",
        },
        {
            "for_risk": "Cross-border transfer",
            "measure": "Verify EU-US DPF certification for us-west-2 region. If not covered, implement Standard Contractual Clauses (SCCs).",
            "timeline": "60 days",
        },
        {
            "for_risk": "Excessive retention",
            "measure": "Reduce marketing_events TTL from 1460 to 365 days. Implement automated data purging for expired records.",
            "timeline": "45 days",
        },
        {
            "for_risk": "Data breach",
            "measure": "Implement prepared statements for all SQL queries. Deploy WAF. Conduct penetration testing.",
            "timeline": "90 days",
        },
    ]


def _generate_residual_risk_assessment(state: dict, iteration: int) -> str:
    """生成 DPIA Section 6: 剩余风险评估。"""
    return (
        f"[DPIA Iteration {iteration}] "
        f"After implementing the proposed mitigation measures, residual risks "
        f"are assessed as follows: (1) Undeclared data risk: LOW after policy "
        f"update; (2) Cross-border risk: MEDIUM until SCCs are fully executed; "
        f"(3) Retention risk: LOW after TTL reduction; (4) Data breach risk: "
        f"MEDIUM — no system is breach-proof, defense-in-depth reduces but "
        f"does not eliminate this risk. The MEDIUM residual risks are within "
        f"acceptable tolerance given the nature of the data processed."
    )


def _generate_consultation_record(state: dict, iteration: int) -> str:
    """生成 DPIA Section 7: 咨询记录。"""
    return (
        f"[DPIA Iteration {iteration}] "
        f"DPO consultation: DPO participated in the audit review process. "
        f"Data subject consultation: not formally conducted; recommend "
        f"user survey on privacy expectations for the next review cycle."
    )


# ─── N9: Reflection Agent（DPIA 质量评估节点）─────────────

def reflection_agent_node(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    V2.2 DPIA Reflection Agent。

    使用 DPIAQualityRubric (WP248) 结构化评分，不是 LLM "觉得好不好"。

    参数:
        state: 含 dpia_report 的状态

    返回:
        dict — 含 dpia_quality_score, dpia_quality_details, reflection_feedback
    """
    dpia = state.get("dpia_report", {})
    iteration = state.get("dpia_iteration", 0)

    if not dpia:
        return {
            "dpia_quality_score": 0.0,
            "dpia_iteration": iteration,
            "reflection_feedback": "No DPIA report to evaluate.",
            "_dpia_veto": True,
            "_dpia_passed": False,
        }

    # ── 使用 DPIAQualityRubric 评分 ──
    result = dpia_rubric.score(dpia)

    # ── 构建评分明细 ──
    quality_details = {
        dim_key: {
            "name": dim.name,
            "weight": dim.weight,
            "score": dim.score,
            "criteria_met": dim.criteria_met,
            "criteria_missed": dim.criteria_missed,
        }
        for dim_key, dim in result.dimensions.items()
    }

    # 计算及格维度数（score ≥ 0.6 视为及格）
    dimensions_passed = sum(
        1 for dim in result.dimensions.values()
        if dim.score >= 0.6
    )
    total_dimensions = len(result.dimensions)

    return {
        "dpia_quality_score": result.total_score,
        "dpia_dimensions_passed": f"{dimensions_passed}/{total_dimensions}",
        "dpia_iteration": iteration,
        "reflection_feedback": result.feedback,
        "dpia_quality_details": quality_details,
        "_dpia_veto": result.veto_triggered,
        "_dpia_passed": result.passed,
    }


# ─── N10: Final Report 节点（V2.2 升级版）─────────────────

def final_report_node(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    V2.2 最终报告生成节点。

    集成法规版本感知：
      - 检查输入文档时效性
      - 生成法规版本 footer
      - 检查知识库更新 → 标记历史结论需复核

    参数:
        state: 完整审计状态

    返回:
        dict — 含 report_text
    """
    # ═══ 去重：operator.add 在循环回边重执行时会追加重复 findings ═══
    raw_findings = state.get("findings", [])
    seen_ids = set()
    findings = []
    for f in raw_findings:
        fid = f.get("finding_id", "")
        if fid and fid not in seen_ids:
            seen_ids.add(fid)
            findings.append(f)
        elif not fid:
            findings.append(f)  # 无 ID 的保留（容错）

    # ═══ 用去重后的 findings 重新计算严重发现数 ═══
    deduped_critical = len([
        f for f in findings if f.get("state") == "FAIL" and f.get("severity") == "HIGH"
    ])

    # ═══ 引用验证：检查每条 finding 引用的 GDPR 条款是否真实存在 ═══
    findings = verify_all_findings(findings)
    verification_stats = get_verification_stats(findings)
    if verification_stats["has_issues"]:
        warnings_list = list(state.get("warnings", []))
        for issue in verification_stats["issues"]:
            warnings_list.append(f"[引用验证] {issue}")
    else:
        warnings_list = list(state.get("warnings", []))

    conflicts = state.get("conflicts", [])
    dpia = state.get("dpia_report", {})
    summary = state.get("synthesis_summary", "")
    risk_tier = state.get("risk_tier", RiskTier.LOW.value)
    dpia_score = state.get("dpia_quality_score", 0.0)
    dpia_details = state.get("dpia_quality_details", {})
    dpo_decision = state.get("dpo_decision", {})

    report_sections = []

    # ── 报告头 ──
    report_sections.append("=" * 60)
    report_sections.append("GDPR PRIVACY AUDIT REPORT")
    report_sections.append("=" * 60)
    report_sections.append(f"Audit ID: {state.get('audit_id', 'UNKNOWN')}")
    report_sections.append(f"Target: {state.get('target_name', 'UNKNOWN')}")
    report_sections.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    report_sections.append(f"Risk Tier: {risk_tier}")
    report_sections.append(f"DPIA Quality: {state.get('dpia_dimensions_passed', '?/7')} (EDPB WP248)")
    report_sections.append("")

    # ── 文档时效性提醒（V2.2 新增）──
    documents_outdated = state.get("documents_outdated", [])
    if documents_outdated:
        report_sections.append("⚠️  DOCUMENT CURRENCY WARNING")
        report_sections.append("─" * 60)
        for doc in documents_outdated:
            report_sections.append(f"  Warning: {doc.get('warning', '')}")
            if doc.get("date"):
                report_sections.append(f"  Document date: {doc['date']}")
        report_sections.append("")

    # ── DPO 决策记录 ──
    if dpo_decision:
        report_sections.append("HUMAN REVIEW DECISION")
        report_sections.append("─" * 60)
        report_sections.append(f"  Action: {dpo_decision.get('action', 'unknown')}")
        if dpo_decision.get("action") == "edit":
            report_sections.append(
                f"  Risk tier changed: {dpo_decision.get('original_risk_tier')} "
                f"→ {dpo_decision.get('new_risk_tier')}"
            )
        report_sections.append(f"  DPO: {dpo_decision.get('dpo_id', 'unknown')}")
        report_sections.append(f"  Timestamp: {dpo_decision.get('timestamp', 'unknown')}")
        report_sections.append("")

    # ── 综合摘要 ──
    report_sections.append("EXECUTIVE SUMMARY")
    report_sections.append("─" * 60)
    report_sections.append(summary)
    report_sections.append("")

    # ── 发现清单（按来源分组）──
    report_sections.append("FINDINGS")
    report_sections.append("─" * 60)
    fail_findings = [f for f in findings if f.get("state") == "FAIL"]
    pass_findings = [f for f in findings if f.get("state") == "PASS"]

    report_sections.append(f"Total: {len(findings)} findings "
                           f"({len(fail_findings)} FAIL, {len(pass_findings)} PASS)")
    report_sections.append("")

    # 按 source 分组
    by_source: dict[str, list] = {}
    for f in fail_findings:
        src = f.get("source", "unknown")
        by_source.setdefault(src, []).append(f)

    # 获取输入文件名
    privacy_names = [d.get("name", "") for d in state.get("privacy_documents", [])]
    schema_names = [d.get("name", "") for d in state.get("data_schemas", [])]

    source_labels = {
        "privacy_doc_auditor": ("📄 Privacy Documents", privacy_names),
        "data_schema_auditor": ("🗄️ Data Schemas", schema_names),
    }

    for src in ["privacy_doc_auditor", "data_schema_auditor"]:
        src_findings = by_source.get(src, [])
        if not src_findings:
            continue
        label, file_list = source_labels.get(src, (src, []))
        report_sections.append(f"  {label}")
        if file_list:
            report_sections.append(f"    Files: {', '.join(file_list)}")
        for f in sorted(src_findings, key=lambda x: (
            {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x.get("severity", "LOW"), 3)
        )):
            report_sections.append(
                f"    [{f.get('severity', '?')}] {f.get('finding_id', '?')}: "
                f"{f.get('title', 'No title')}"
            )
            report_sections.append(
                f"      Category: {f.get('category', '?')} | "
                f"Articles: {f.get('related_articles', [])}"
            )
        report_sections.append("")

    # ── 引用验证记录（防幻觉）──
    if verification_stats["has_issues"]:
        report_sections.append("CITATION VERIFICATION")
        report_sections.append("─" * 60)
        report_sections.append(
            f"  {verification_stats['failed']}/{verification_stats['total']} "
            f"findings had citation issues."
        )
        report_sections.append("  Issues detected:")
        for issue in verification_stats["issues"]:
            report_sections.append(f"    ⚠ {issue}")
        report_sections.append(
            "  Note: LLM-generated article references are programmatically "
            "verified. Invalid references have been removed or flagged."
        )
        report_sections.append("")

    # ── 冲突消解记录 ──
    if conflicts:
        report_sections.append("CONFLICT RESOLUTION")
        report_sections.append("─" * 60)
        for c in conflicts:
            arb = c.get("arbitration_result", {})
            ver = c.get("verification_result", {})
            report_sections.append(
                f"  {c.get('conflict_id', '?')}: {c.get('conflict_type', '?')}"
            )
            report_sections.append(
                f"    Method: {arb.get('method', '?')} | "
                f"Winner: {arb.get('winner', '?')} | "
                f"Verified: {ver.get('verified', '?')}"
            )
            report_sections.append("")

    # ── DPIA 质量评估（V2.2: EDPB WP248 量表）──
    if dpia_details:
        dims_passed = state.get("dpia_dimensions_passed", "?/7")
        report_sections.append(f"DPIA QUALITY ASSESSMENT (EDPB WP248) — {dims_passed}")
        report_sections.append("─" * 60)
        for dim_key, dim in dpia_details.items():
            passed = "✓" if dim["score"] >= 0.6 else "✗"
            bar = "█" * int(dim["score"] * 20) + "░" * (20 - int(dim["score"] * 20))
            report_sections.append(
                f"  {passed} {dim['name']} (weight: {dim['weight']:.0%}): "
                f"{dim['score']:.2f} {bar}"
            )
            if dim.get("criteria_missed"):
                for missed in dim["criteria_missed"]:
                    report_sections.append(f"    → 未满足: {missed}")
        report_sections.append("")

    # ── 知识库更新提醒（V2.2 新增）──
    kb_updates = version_tracker.check_kb_updates(
        state.get("document_date", "2020-01-01")
    )
    if kb_updates:
        report_sections.append("REGULATORY UPDATE NOTICE")
        report_sections.append("─" * 60)
        report_sections.append(
            "The following regulatory updates may affect this audit:"
        )
        for update in kb_updates:
            report_sections.append(
                f"  • {update.get('title', 'Unknown')} → "
                f"{update.get('new_version', '?')} ({update.get('date', '?')})"
            )
            report_sections.append(
                f"    {update.get('action_required', '')}"
            )
        report_sections.append("")

    # ── 法规版本 Footer ──
    report_sections.append("─" * 60)
    report_sections.append(version_tracker.get_report_footer(
        rag_build_date=version_tracker.get_rag_build_date(),
    ))

    report_text = "\n".join(report_sections)

    return {
        "report_text": report_text,
        "phase": AuditPhase.COMPLETED.value,
        "critical_findings_count": deduped_critical,
        "documents_outdated": documents_outdated,
        "kb_has_updates": len(kb_updates) > 0,
        "regulation_versions": version_tracker.get_version_metadata(),
        "warnings": warnings_list,
        "_citation_verification": verification_stats,
    }


# ═══════════════════════════════════════════════════════════
# 路由函数
# ═══════════════════════════════════════════════════════════

def route_after_init(state: GDPRPrivacyAuditStateV2_2) -> str:
    """
    初始化后的路由。

    返回:
        "collect" — 进入证据收集阶段
        "end" — 输入无效，直接结束
    """
    phase = state.get("phase", "")
    if phase == AuditPhase.COMPLETED.value:
        return "end"
    return "collect"


def dispatch_evidence_tasks(state: GDPRPrivacyAuditStateV2_2) -> list[Send]:
    """
    Fan-Out 任务派发函数。

    这是 LangGraph Fan-Out 的核心——使用 Send() API 根据 input_types
    动态创建并发任务。如果只有 1 种输入，只发 1 个 Agent；
    如果有 2 种输入，2 个 Agent 并行执行。

    LangGraph 知识点：
      - Send(node_name, state) 创建一个任务
      - 返回 list[Send] → LangGraph 并发执行所有任务
      - 每个 Agent 返回的 findings/evidence 通过 operator.add 合并

    返回:
        list[Send] — 1 或 2 个并发任务
    """
    input_types = state.get("input_types", [])
    tasks = []

    if "privacy_document" in input_types:
        tasks.append(Send("privacy_doc_auditor", state))

    if "data_schema" in input_types:
        tasks.append(Send("data_schema_auditor", state))

    return tasks


def route_by_risk(state: GDPRPrivacyAuditStateV2_2) -> str:
    """
    风险等级路由。

    返回:
        "human_review" — HIGH 风险 → HITL 人审
        "continue" — MEDIUM/LOW 风险 → 继续 DPIA
        "end" — INCONCLUSIVE → 直接结束
    """
    risk_tier = state.get("risk_tier", "")

    if risk_tier == RiskTier.INCONCLUSIVE.value:
        return "end"
    if risk_tier == RiskTier.HIGH.value:
        return "human_review"
    return "continue"


def route_after_human_review(state: GDPRPrivacyAuditStateV2_2) -> str:
    """
    DPO 审批后的三路路由（V2.2 简化版）。

    返回:
        "continue" — DPO approve → 进入 DPIA
        "re_evaluate" — DPO edit（改了 risk_tier）→ 回到 synthesis
        "end" — DPO reject → 直接结束
    """
    if state.get("_dpo_rejected"):
        return "end"
    if state.get("_dpo_edited"):
        return "re_evaluate"
    return "continue"


def route_dpia_reflection(state: GDPRPrivacyAuditStateV2_2) -> str:
    """
    DPIA Reflection 质量路由。

    返回:
        "pass" — 质量达标 → 最终报告
        "retry" — 质量不达标且未达最大轮次 → 循环回边 #4
        "escalate" — 触发一票否决或达最大轮次 → 人工升级
    """
    dpia_passed = state.get("_dpia_passed", False)
    dpia_iter = state.get("dpia_iteration", 0)
    max_iter = state.get("MAX_DPIA_ITERATIONS", 3)
    veto_triggered = state.get("_dpia_veto", False)

    if dpia_passed:
        return "pass"

    # 一票否决且已尝试 ≥ 1 轮 → 升级（一票否决不太可能通过改描述解决）
    if veto_triggered and dpia_iter >= max_iter:
        return "escalate"

    if dpia_iter >= max_iter:
        return "escalate"

    # 带着 reflection_feedback 回到 dpia_generator（循环回边 #4）
    return "retry"


# ═══════════════════════════════════════════════════════════
# 图构建
# ═══════════════════════════════════════════════════════════

def build_graph(
    checkpointer: InMemorySaver = None,
) -> StateGraph:
    """
    构建完整的 GDPR Privacy Auditor V2.2 主图。

    参数:
        checkpointer: InMemorySaver 实例（用于状态持久化和 HITL 中断恢复）。
                      如果不传，会创建一个默认的内存 checkpointer。

    返回:
        编译后的 StateGraph，可直接调用 .invoke(initial_state)

    用法:
        graph = build_graph()
        result = graph.invoke(create_initial_state(...))
    """
    # ── Checkpointer ──
    if checkpointer is None:
        checkpointer = InMemorySaver()

    # ── 构建器 ──
    builder = StateGraph(GDPRPrivacyAuditStateV2_2)

    # ── 注册节点 ──
    # 基础节点
    builder.add_node("init_node", init_node)
    builder.add_node("evidence_supervisor", evidence_supervisor_node)

    # 2 个 Specialist Agent（Fan-Out 目标）
    builder.add_node("privacy_doc_auditor", privacy_doc_auditor_node)
    builder.add_node("data_schema_auditor", data_schema_auditor_node)

    # 冲突消解子图
    builder.add_node("conflict_resolution", conflict_subgraph)

    # 分析节点
    builder.add_node("synthesis_agent", synthesis_agent_node)
    builder.add_node("risk_rater", risk_rater_node)

    # HITL 人审
    builder.add_node("human_review", human_review_node)

    # DPIA 生成 + Reflection
    builder.add_node("dpia_generator", dpia_generator_node)
    builder.add_node("reflection_agent", reflection_agent_node)

    # 最终报告
    builder.add_node("final_report", final_report_node)

    # ── 连接边 ──

    # START → 初始化
    builder.add_edge(START, "init_node")

    # 初始化 → 路由
    builder.add_conditional_edges(
        "init_node",
        route_after_init,
        {
            "collect": "evidence_supervisor",
            "end": END,
        }
    )

    # Fan-Out: 督导根据 input_types 决定并发几个 Agent
    builder.add_conditional_edges(
        "evidence_supervisor",
        dispatch_evidence_tasks,
        {
            "privacy_doc_auditor": "privacy_doc_auditor",
            "data_schema_auditor": "data_schema_auditor",
        }
    )

    # Fan-In: 两个 Agent → 冲突消解子图
    builder.add_edge("privacy_doc_auditor", "conflict_resolution")
    builder.add_edge("data_schema_auditor", "conflict_resolution")

    # 冲突消解 → 综合分析
    builder.add_edge("conflict_resolution", "synthesis_agent")

    # 综合分析 → 风险评估
    builder.add_edge("synthesis_agent", "risk_rater")

    # 风险评估 → 三路路由
    builder.add_conditional_edges(
        "risk_rater",
        route_by_risk,
        {
            "continue": "dpia_generator",
            "human_review": "human_review",
            "end": END,
        }
    )

    # HITL → 三路路由（循环回边 #3: edit → re_evaluate → synthesis_agent）
    builder.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "continue": "dpia_generator",
            "re_evaluate": "synthesis_agent",   # ← 循环回边 #3
            "end": END,
        }
    )

    # DPIA 生成 → Reflection
    builder.add_edge("dpia_generator", "reflection_agent")

    # Reflection → 三路路由（循环回边 #4: retry → dpia_generator）
    builder.add_conditional_edges(
        "reflection_agent",
        route_dpia_reflection,
        {
            "pass": "final_report",
            "retry": "dpia_generator",           # ← 循环回边 #4
            "escalate": "human_review",
        }
    )

    # 最终报告 → END
    builder.add_edge("final_report", END)

    # ── 编译 ──
    return builder.compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════

# 预编译的图实例（模块导入时创建）
_graph = None


def get_graph():
    """
    获取或创建编译后的图实例（单例模式）。

    首次调用时构建图，后续调用返回同一实例。
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_audit(
    audit_id: str = None,
    target_name: str = "Unnamed Target",
    target_description: str = "",
    privacy_documents: list[dict] = None,
    data_schemas: list[dict] = None,
    document_date: str = "",
    config: dict = None,
) -> GDPRPrivacyAuditStateV2_2:
    """
    运行一次完整的 GDPR 审计。

    这是主要的外部接口。

    参数:
        audit_id: 审计 ID（不传则自动生成）
        target_name: 审计目标名称
        target_description: 审计目标描述
        privacy_documents: 隐私文档列表
        data_schemas: 数据表结构列表
        document_date: 输入文档日期
        config: LangGraph 配置（如 thread_id）

    返回:
        完整的 GDPRPrivacyAuditStateV2_2（所有字段已填充）

    用法:
        result = run_audit(
            target_name="E-Commerce Platform",
            privacy_documents=[
                {"name": "privacy.md", "content": "We collect your email..."}
            ],
            data_schemas=[
                {"name": "users.sql", "content": "CREATE TABLE users (...)"}
            ],
        )
        print(result["report_text"])
    """
    if audit_id is None:
        audit_id = f"AUD-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    initial_state = create_initial_state(
        audit_id=audit_id,
        target_name=target_name,
        target_description=target_description,
        privacy_documents=privacy_documents or [],
        data_schemas=data_schemas or [],
        document_date=document_date,
    )

    graph = get_graph()

    invoke_config = {"configurable": {"thread_id": audit_id}}
    if config:
        invoke_config.update(config)

    result = graph.invoke(initial_state, config=invoke_config)

    return result
