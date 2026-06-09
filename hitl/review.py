"""
HITL（Human-in-the-Loop）人审模块 — V2.2 整份结论级审批。

DPO 审批的是整份审计结论，不是逐条发现。
三种操作：Approve / Edit / Reject。

LangGraph 知识点（面试重点）：
  1. interrupt() — LangGraph 的暂停机制，程序在此等待外部输入
  2. Command(resume=...) — 外部恢复机制，传入 DPO 决策后继续执行
  3. SqliteSaver/InMemorySaver — 状态持久化，保证中断后的状态不丢失
  4. 循环回边 #3 — DPO edit → 重走 synthesis_agent + risk_rater

中断流程：
  graph.invoke(state) → 执行到 human_review_node
                      → interrupt(conclusion)  ← 程序暂停
                      → 等待外部输入...
                      → Command(resume={"action": "approve"})
                      → 继续执行后续节点

使用方式（CLI 模式）：
  from graph import get_graph
  from state import create_initial_state

  graph = get_graph()
  config = {"configurable": {"thread_id": "audit-001"}}

  # 第一次调用 — 会停在 human_review 节点
  for event in graph.stream(initial_state, config):
      print(event)

  # 检查是否需要人审
  state = graph.get_state(config)
  if state.next == ('human_review',):
      # DPO 做出决策
      graph.invoke(Command(resume={"action": "approve"}), config)
"""

from datetime import datetime
from typing import Optional
from langgraph.types import interrupt

from state import GDPRPrivacyAuditStateV2_2, RiskTier


# ═══════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════

class DPOAction:
    """DPO 可执行的三种操作常量。"""
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


# ═══════════════════════════════════════════════════════════
# HITL 人审节点（生产版 — 使用真实 interrupt）
# ═══════════════════════════════════════════════════════════

def human_review_node(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    HITL 人审节点 — V2.2 整份结论级审批。

    DPO 看到的是完整的审计结论摘要 + Top 5 严重发现 + 法规版本信息，
    然后做出 Approve / Edit / Reject 决策。

    此节点使用真实的 LangGraph interrupt() 机制：
      - 执行到此 → 暂停，等待外部 Command(resume=...)
      - DPO 可通过 CLI / Web UI / API 三种途径传入决策

    参数:
        state: 含 synthesis_summary, risk_tier, findings 的完整状态

    返回:
        dict — 含 dpo_decision 和路由标记（resume 后继续执行）

    LangGraph interrupt 工作原理：
      1. interrupt(value) 将 value 发送给外部调用者
      2. 程序在此暂停，状态被 checkpointer 持久化
      3. 外部通过 graph.invoke(Command(resume=response), config) 恢复
      4. interrupt() 返回 response（即 DPO 的决策）
    """
    # ═══ 构建 DPO 审批界面数据 ═══
    conclusion = _build_conclusion(state)

    # ═══ 中断点 ═══
    # 程序在此暂停。外部调用者收到 conclusion 数据，
    # 展示给 DPO，等待 DPO 决策。
    dpo_response = interrupt({
        "type": "human_review_conclusion",
        "conclusion": conclusion,
        "editable_fields": ["risk_tier"],
        "allowed_actions": [DPOAction.APPROVE, DPOAction.EDIT, DPOAction.REJECT],
        "instructions": (
            "Please review the audit conclusion above.\n"
            "- approve: Accept the conclusion and continue to DPIA generation.\n"
            "- edit: Override the risk tier (e.g., downgrade from HIGH to MEDIUM).\n"
            "- reject: Dismiss the audit as inapplicable (marks INCONCLUSIVE)."
        ),
    })
    # ═══ 恢复点 ═══
    # DPO 决策通过 Command(resume=dpo_response) 传入后，代码从此继续

    return _process_dpo_response(state, dpo_response)


def human_review_node_simulated(
    state: GDPRPrivacyAuditStateV2_2,
    dpo_action: str = DPOAction.APPROVE,
    dpo_new_tier: str = "",
    dpo_reason: str = "",
) -> dict:
    """
    HITL 人审节点 — 模拟版（测试用，不走真实 interrupt）。

    与 human_review_node 的区别：
      - 直接接受 DPO 决策参数，不调用 interrupt()
      - 用于自动化测试场景

    参数:
        state: 完整审计状态
        dpo_action: "approve" | "edit" | "reject"
        dpo_new_tier: 如果 action=edit，新的风险等级
        dpo_reason: 如果 action=reject，驳回原因

    返回:
        dict — 状态更新
    """
    dpo_response = {
        "action": dpo_action,
        "dpo_id": "dpo-simulated",
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if dpo_action == DPOAction.EDIT:
        dpo_response["new_risk_tier"] = dpo_new_tier or RiskTier.MEDIUM.value
    elif dpo_action == DPOAction.REJECT:
        dpo_response["reason"] = dpo_reason or "DPO determined audit scope does not apply."

    return _process_dpo_response(state, dpo_response)


# ═══════════════════════════════════════════════════════════
# 内部函数
# ═══════════════════════════════════════════════════════════

def _build_conclusion(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    构建 DPO 审批界面所需的数据结构。

    这是 DPO 看到的"审批单"内容。
    """
    findings = state.get("findings", [])
    risk_tier = state.get("risk_tier", RiskTier.LOW.value)

    # Top 5 严重发现
    high_findings = sorted(
        [f for f in findings if f.get("severity") == "HIGH"],
        key=lambda f: f.get("severity", "LOW"),
        reverse=True,
    )[:5]

    # 按类别汇总
    categories = {}
    for f in findings:
        cat = f.get("category", "UNKNOWN")
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1

    return {
        "audit_id": state.get("audit_id", "UNKNOWN"),
        "target_name": state.get("target_name", "UNKNOWN"),
        "target_description": state.get("target_description", ""),
        "risk_tier": risk_tier,
        "total_findings": len(findings),
        "critical_findings_count": state.get("critical_findings_count", 0),
        "finding_categories": categories,
        "top_critical_findings": [
            {
                "id": f.get("finding_id", "?"),
                "title": f.get("title", "No title"),
                "category": f.get("category", "?"),
                "severity": f.get("severity", "?"),
                "description": f.get("description", "")[:200],
            }
            for f in high_findings
        ],
        "synthesis_summary": state.get("synthesis_summary", ""),
        "conflict_resolution_summary": {
            "total_conflicts": len(state.get("conflicts", [])),
            "resolved": sum(
                1 for c in state.get("conflicts", [])
                if c.get("resolved", False)
            ),
        },
        "regulation_versions": state.get("regulation_versions", {}),
        "confidence_score": state.get("confidence_score", 0.0),
    }


def _process_dpo_response(
    state: GDPRPrivacyAuditStateV2_2,
    dpo_response: dict,
) -> dict:
    """
    处理 DPO 决策并返回状态更新。

    三种路径:
      approve → 继续到 DPIA
      edit    → 修改 risk_tier，触发重评估（循环回边 #3）
      reject  → 标记 INCONCLUSIVE，结束审计
    """
    action = dpo_response.get("action", DPOAction.APPROVE)

    if action == DPOAction.APPROVE:
        return {
            "needs_human_review": False,
            "dpo_decision": {
                "action": DPOAction.APPROVE,
                "dpo_id": dpo_response.get("dpo_id", "unknown"),
                "timestamp": dpo_response.get(
                    "timestamp",
                    datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                ),
            },
            "_dpo_edited": False,
            "_dpo_rejected": False,
        }

    elif action == DPOAction.EDIT:
        new_tier = dpo_response.get(
            "new_risk_tier",
            state.get("risk_tier", RiskTier.MEDIUM.value),
        )
        return {
            "needs_human_review": False,
            "risk_tier": new_tier,
            "dpo_decision": {
                "action": DPOAction.EDIT,
                "original_risk_tier": state.get("risk_tier"),
                "new_risk_tier": new_tier,
                "dpo_id": dpo_response.get("dpo_id", "unknown"),
                "timestamp": dpo_response.get(
                    "timestamp",
                    datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                ),
            },
            "_dpo_edited": True,
            "_dpo_rejected": False,
        }

    elif action == DPOAction.REJECT:
        return {
            "needs_human_review": False,
            "phase": "COMPLETED",
            "report_text": (
                "[AUDIT REJECTED] "
                f"The audit conclusion has been rejected by DPO "
                f"({dpo_response.get('dpo_id', 'unknown')}). "
                f"Reason: {dpo_response.get('reason', 'No reason provided')}. "
                f"The audit is marked as INCONCLUSIVE."
            ),
            "dpo_decision": {
                "action": DPOAction.REJECT,
                "reason": dpo_response.get("reason", ""),
                "dpo_id": dpo_response.get("dpo_id", "unknown"),
                "timestamp": dpo_response.get(
                    "timestamp",
                    datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                ),
            },
            "risk_tier": RiskTier.INCONCLUSIVE.value,
            "_dpo_edited": False,
            "_dpo_rejected": True,
        }

    return {}


# ═══════════════════════════════════════════════════════════
# 路由函数
# ═══════════════════════════════════════════════════════════

def route_after_human_review(state: GDPRPrivacyAuditStateV2_2) -> str:
    """
    DPO 审批后的三路路由（V2.2 简化版）。

    返回:
        "continue"    — DPO approve → 进入 DPIA 生成
        "re_evaluate" — DPO edit（改了 risk_tier）→ 回到 synthesis（循环回边 #3）
        "end"         — DPO reject → 直接结束审计

    面试要点：
      这是 LangGraph 循环回边 #3 的路由函数。
      DPO edit 不是简单的状态修改——而是触发重新评估。
      synthesis_agent 会用新的 risk_tier 重新生成摘要，
      risk_rater 会尊重 DPO 决定但追加 warning 如果不合理。
    """
    if state.get("_dpo_rejected"):
        return "end"
    if state.get("_dpo_edited"):
        return "re_evaluate"
    return "continue"


# ═══════════════════════════════════════════════════════════
# CLI 审批界面（开发用）
# ═══════════════════════════════════════════════════════════

def display_conclusion(conclusion: dict) -> str:
    """
    将审计结论格式化为可读的终端文本。

    用于开发调试和 CLI 模式下的 DPO 审批。

    参数:
        conclusion: _build_conclusion() 返回的字典

    返回:
        格式化的多行字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"  GDPR AUDIT — Human Review Required")
    lines.append("=" * 60)
    lines.append(f"  Audit ID:     {conclusion.get('audit_id', '?')}")
    lines.append(f"  Target:       {conclusion.get('target_name', '?')}")
    lines.append(f"  Risk Tier:    {conclusion.get('risk_tier', '?')}")
    lines.append(f"  Confidence:   {conclusion.get('confidence_score', 0):.0%}")
    lines.append(f"  Total Findings: {conclusion.get('total_findings', 0)}")
    lines.append(f"  Critical:     {conclusion.get('critical_findings_count', 0)}")
    lines.append("")

    # 发现按类别分布
    categories = conclusion.get("finding_categories", {})
    if categories:
        lines.append("  Finding Categories:")
        for cat, count in sorted(categories.items()):
            lines.append(f"    • {cat}: {count}")
    lines.append("")

    # Top 5 严重发现
    top = conclusion.get("top_critical_findings", [])
    if top:
        lines.append("  Top Critical Findings:")
        for i, f in enumerate(top):
            lines.append(f"    {i+1}. [{f['severity']}] {f['title']}")
            lines.append(f"       Category: {f['category']}")
            lines.append(f"       {f['description'][:150]}")
    lines.append("")

    # 冲突消解摘要
    cr = conclusion.get("conflict_resolution_summary", {})
    if cr.get("total_conflicts", 0) > 0:
        lines.append(f"  Conflicts: {cr['total_conflicts']} detected, "
                      f"{cr['resolved']} resolved")
    lines.append("")

    lines.append("  Regulation Versions:")
    for reg_id, info in conclusion.get("regulation_versions", {}).items():
        lines.append(f"    • {info.get('name', reg_id)} {info.get('version', '?')}")
    lines.append("")

    lines.append("-" * 60)
    lines.append("  DPO Actions:")
    lines.append("    [A]pprove  — Accept conclusion, continue to DPIA")
    lines.append("    [E]dit     — Override risk tier (HIGH→MEDIUM)")
    lines.append("    [R]eject   — Dismiss as INCONCLUSIVE, end audit")
    lines.append("=" * 60)

    return "\n".join(lines)
