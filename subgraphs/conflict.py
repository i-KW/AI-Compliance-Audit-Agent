"""
Conflict Resolution 子图 — V2.2 冲突消解双层架构。

这是整个项目中展示 LangGraph 核心机制（子图嵌套、循环回边、条件路由）
最密集的地方。

子图内部流程：
  START → conflict_detector
            ├─ no_conflict → conflict_resolved → END
            ├─ has_conflict → arbitration_agent → verification_agent
            │                    ↑                      │
            │                    └── retry ─────────────┘  (循环回边 #1)
            │                    escalation → END
            └─ evidence_gap → evidence_refinement
                                └── retry ──┘             (循环回边 #2)

集成：
  - GDPRPriorityEngine: Layer 1 规则引擎（80% 常规冲突）
  - LLM: Layer 2 情境推断（20% 同级冲突）+ 解释文本生成
  - 每条裁决记录 ResolutionMethod → 可审计

LangGraph 知识点（面试时展示此文件）：
  - StateGraph 子图独立编译，作为父图节点使用
  - 循环回边的计数和终止条件
  - 条件边根据冲突类型和轮次路由
"""

from langgraph.graph import StateGraph, START, END
from typing import Literal

from state import (
    GDPRPrivacyAuditStateV2_2,
    ConflictType,
    ResolutionMethod,
)
from rules.priority import GDPRPriorityEngine, parse_llm_winner


# ═══════════════════════════════════════════════════════════
# 单例
# ═══════════════════════════════════════════════════════════

priority_engine = GDPRPriorityEngine()


# ═══════════════════════════════════════════════════════════
# 节点：冲突检测器
# ═══════════════════════════════════════════════════════════

def conflict_detector_node(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    冲突检测节点。

    在两个 Agent 各自产出 findings 后，此节点对比双方发现：
      - 声明 vs 实际数据范围不一致 → DATA_SCOPE_DISCREPANCY
      - 声明保留期 vs 实际 TTL 不匹配 → RETENTION_MISMATCH
      - 实际跨境传输但政策未声明 → TRANSFER_UNDECLARED
      - 同意范围与实际使用不匹配 → CONSENT_SCOPE_GAP

    还会检测证据是否充足（为 evidence_refinement 路由）。

    参数:
        state: 包含两个 Agent 的 findings 的状态

    返回:
        dict — 含 conflicts 列表和 conflict_detected 标志
    """
    findings = state.get("findings", [])
    evidence = state.get("evidence", [])

    # 分离两个 Agent 的发现
    privacy_findings = [
        f for f in findings if f.get("source") == "privacy_doc_auditor"
    ]
    data_findings = [
        f for f in findings if f.get("source") == "data_schema_auditor"
    ]

    # 如果没有两个 Agent 的输出，无法对比 → 无冲突
    if not privacy_findings or not data_findings:
        return {
            "conflicts": [],
            "conflict_detected": False,
        }

    conflicts = []
    evidence_gaps = []

    # ── 检测模式 1: 数据范围不一致 ──
    # 对比隐私政策声明的类别 vs 实际 PII 扫描结果
    undeclared_pii = [
        f for f in data_findings
        if f.get("category") == "UNDECLARED_PII"
    ]
    if undeclared_pii:
        declared_count = undeclared_pii[0].get("declared_pii_count", 0)
        actual_count = undeclared_pii[0].get("actual_pii_count", 0)
        if actual_count > declared_count:
            conflicts.append({
                "conflict_id": f"C-{state.get('audit_id', 'UNKNOWN')}-001",
                "conflict_type": ConflictType.DATA_SCOPE_DISCREPANCY.value,
                "description": (
                    f"Privacy policy declares {declared_count} data categories, "
                    f"but database schema contains {actual_count} PII fields. "
                    f"Undeclared fields: {undeclared_pii[0].get('undeclared_fields', [])}"
                ),
                "description_privacy": (
                    f"Privacy Doc Auditor found: policy declares {declared_count} "
                    f"categories with valid legal bases."
                ),
                "description_data": (
                    f"Data Schema Auditor found: database has {actual_count} PII "
                    f"columns including sensitive fields."
                ),
                "resolved": False,
            })

    # ── 检测模式 2: 跨境传输未声明 ──
    transfer_findings = [
        f for f in data_findings
        if f.get("category") == "TRANSFER_UNDECLARED"
    ]
    privacy_has_transfer_declaration = any(
        "cross-border" in f.get("description", "").lower() or
        "transfer" in f.get("description", "").lower() or
        "international" in f.get("description", "").lower()
        for f in privacy_findings
    )
    if transfer_findings and not privacy_has_transfer_declaration:
        conflicts.append({
            "conflict_id": f"C-{state.get('audit_id', 'UNKNOWN')}-002",
            "conflict_type": ConflictType.TRANSFER_UNDECLARED.value,
            "description": (
                f"Data Schema Auditor detected cross-border transfer to "
                f"{transfer_findings[0].get('destination', 'unknown')}, "
                f"but Privacy Doc Auditor did not find a corresponding "
                f"declaration in the privacy policy."
            ),
            "description_privacy": (
                "Privacy Doc Auditor: no cross-border data transfer "
                "declaration found in policy text."
            ),
            "description_data": (
                f"Data Schema Auditor: data stored in "
                f"{transfer_findings[0].get('destination', 'unknown')} "
                f"without documented safeguards."
            ),
            "resolved": False,
        })

    # ── 检测模式 3: 保留期不匹配 ──
    retention_findings = [
        f for f in data_findings
        if f.get("category") == "RETENTION_EXCESSIVE"
    ]
    if retention_findings:
        conflicts.append({
            "conflict_id": f"C-{state.get('audit_id', 'UNKNOWN')}-003",
            "conflict_type": ConflictType.RETENTION_MISMATCH.value,
            "description": (
                f"Marketing data retention period of "
                f"{retention_findings[0].get('actual_ttl_days', 'unknown')} days "
                f"exceeds guideline of {retention_findings[0].get('guideline_max_days', 'unknown')} days. "
                f"Privacy policy may not adequately address this."
            ),
            "description_privacy": (
                "Privacy Doc Auditor: policy mentions data retention "
                "but may not specify per-category limits."
            ),
            "description_data": (
                f"Data Schema Auditor: marketing_events TTL = "
                f"{retention_findings[0].get('actual_ttl_days', 'unknown')} days, "
                f"{retention_findings[0].get('excess_factor', 'unknown')}x guideline."
            ),
            "resolved": False,
        })

    # ── 检测模式 4: 同意范围差距 ──
    consent_findings = [
        f for f in privacy_findings
        if f.get("category") in ("CONSENT_LANGUAGE_VAGUE", "VAGUE_PURPOSE")
    ]
    special_category_data = [
        f for f in data_findings
        if f.get("category") == "SPECIAL_CATEGORY_DATA"
    ]
    if consent_findings and special_category_data:
        conflicts.append({
            "conflict_id": f"C-{state.get('audit_id', 'UNKNOWN')}-004",
            "conflict_type": ConflictType.CONSENT_SCOPE_GAP.value,
            "description": (
                f"Privacy policy has vague consent language, while database "
                f"contains sensitive data types ({special_category_data[0].get('sensitive_fields', [])}) "
                f"that require explicit consent under Art.9."
            ),
            "description_privacy": (
                f"Privacy Doc Auditor: consent language is ambiguous - "
                f"'{consent_findings[0].get('title', 'unknown')}'"
            ),
            "description_data": (
                f"Data Schema Auditor: detected sensitive fields "
                f"{special_category_data[0].get('sensitive_fields', [])} "
                f"without explicit consent mechanism."
            ),
            "resolved": False,
        })

    # ── 检查证据缺口 ──
    # 如果任何一个 Agent 产生的证据少于预期，标记为证据缺口
    privacy_evidence_count = len([
        e for e in evidence
        if e.get("source") == "privacy_doc_auditor"
    ])
    data_evidence_count = len([
        e for e in evidence
        if e.get("source") == "data_schema_auditor"
    ])

    if privacy_evidence_count == 0 and state.get("privacy_documents"):
        evidence_gaps.append({
            "gap_id": f"GAP-PRIV-{state.get('audit_id', 'UNKNOWN')}",
            "agent": "privacy_doc_auditor",
            "reason": "Privacy Doc Auditor produced zero evidence despite input documents.",
            "recommended_action": "re_run_privacy_auditor",
        })

    if data_evidence_count == 0 and state.get("data_schemas"):
        evidence_gaps.append({
            "gap_id": f"GAP-DATA-{state.get('audit_id', 'UNKNOWN')}",
            "agent": "data_schema_auditor",
            "reason": "Data Schema Auditor produced zero evidence despite input schemas.",
            "recommended_action": "re_run_data_auditor",
        })

    return {
        "conflicts": conflicts,
        "conflict_detected": len(conflicts) > 0,
        "evidence_gaps": evidence_gaps,
    }


# ═══════════════════════════════════════════════════════════
# 节点：仲裁 Agent（V2.2 升级版——集成 PriorityEngine）
# ═══════════════════════════════════════════════════════════

def arbitration_agent_node(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    V2.2 仲裁节点：规则引擎预处理 + LLM 解释/推断。

    双层架构：
      Layer 1 (GDPRPriorityEngine): 硬编码条款权重比较 → 直接裁决
      Layer 2 (LLM): 同权重条款 → 情境推断 / 生成解释文本

    Phase 2 行为：
      - 规则引擎完整运行（真实逻辑）
      - LLM 部分使用模拟响应（Phase 3 接入真实 LLM）

    参数:
        state: 含 conflicts 列表的状态

    返回:
        dict — 含更新后 conflicts + conflict_resolution_methods
    """
    conflicts = state.get("conflicts", [])
    resolution_round = state.get("conflict_resolution_round", 0)
    current_methods = list(state.get("conflict_resolution_methods", []))

    results = []

    for conflict in conflicts:
        # 跳过已解决的冲突
        if conflict.get("resolved"):
            results.append(conflict)
            continue

        conflict_id = conflict.get("conflict_id", "UNKNOWN")

        # ── Step 1: 规则引擎预处理 ──
        engine_result = priority_engine.resolve(conflict)

        # ── Step 2: LLM 处理 ──
        # Phase 2: 模拟 LLM 响应。Phase 3: llm.invoke(engine_result["llm_prompt"])
        if engine_result["needs_llm"]:
            if engine_result["method"] == ResolutionMethod.LLM_CONTEXTUAL.value:
                # LLM 情境推断 — 模拟一个合理的推断结果
                # Phase 3: 真实 llm.invoke() + parse_llm_winner()
                winner = _simulate_contextual_decision(conflict, engine_result)
                explanation = (
                    f"[SIMULATED - Phase 3 will use real LLM] "
                    f"After contextual analysis of conflict {conflict_id}: "
                    f"both articles carry equal weight ({engine_result['weight_privacy']}), "
                    f"but the context favors {winner}'s interpretation because "
                    f"'{conflict.get('description', '')}' primarily concerns "
                    f"{'transparency of disclosure' if winner == 'privacy_doc_auditor' else 'actual data handling practices'}."
                )
            else:
                # RULE_ENGINE — LLM 只生成解释
                # Phase 3: 真实 llm.invoke(engine_result["llm_prompt"])
                winner = engine_result["winner"]
                explanation = (
                    f"[SIMULATED - Phase 3 will use real LLM] "
                    f"Per GDPRPriorityEngine: {winner}'s conclusion takes precedence. "
                    f"Weight comparison: {engine_result.get('weight_privacy', '?')} vs "
                    f"{engine_result.get('weight_data', '?')}. "
                    f"Higher-weight article reflects stricter GDPR penalty tier "
                    f"(Art.83(4) vs Art.83(5))."
                )
        else:
            winner = engine_result["winner"]
            explanation = "Resolved without LLM intervention."

        # ── Step 3: 记录仲裁结果 ──
        method_record = {
            "conflict_id": conflict_id,
            "method": engine_result["method"],
            "rule_applied": engine_result["rule_applied"],
            "winner": winner,
            "weight_privacy": engine_result["weight_privacy"],
            "weight_data": engine_result["weight_data"],
            "round": resolution_round + 1,
        }
        current_methods.append(method_record)

        resolved = {
            **conflict,
            "arbitration_result": {
                "method": engine_result["method"],
                "rule_applied": engine_result["rule_applied"],
                "winner": winner,
                "weight_privacy": engine_result["weight_privacy"],
                "weight_data": engine_result["weight_data"],
                "explanation": explanation,
                "round": resolution_round + 1,
            },
        }
        results.append(resolved)

    return {
        "conflicts": results,
        "conflict_resolution_round": resolution_round + 1,
        "conflict_resolution_methods": current_methods,
    }


def _simulate_contextual_decision(conflict: dict, engine_result: dict) -> str:
    """
    Phase 2 模拟函数：在 LLM_CONTEXTUAL 模式下做合理推断。

    Phase 3 会用真实 LLM 替代此函数。

    当前逻辑：根据冲突类型选择通常占优势的 Agent。
    这是简化的启发式——真实实现中由 LLM 做情境推理。
    """
    conflict_type = conflict.get("conflict_type", "")
    # 按冲突类型的合理默认
    contextual_defaults = {
        "DATA_SCOPE_DISCREPANCY": "data_schema_auditor",
        "RETENTION_MISMATCH": "data_schema_auditor",
        "TRANSFER_UNDECLARED": "data_schema_auditor",
        "CONSENT_SCOPE_GAP": "privacy_doc_auditor",
    }
    return contextual_defaults.get(
        conflict_type,
        "data_schema_auditor"
    )


# ═══════════════════════════════════════════════════════════
# 节点：验证 Agent（V2.2 升级版）
# ═══════════════════════════════════════════════════════════

def verification_agent_node(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    V2.2 验证节点：验证"冲突是否实际消除"，不是"仲裁逻辑是否自洽"。

    改进（vs V2.1 MVP）：
      旧：验证仲裁逻辑自洽 → 两条 LLM 可以互相确认幻觉
      新：验证冲突是否消除 → 回到原始证据，用 loser 的证据挑战 winner 结论

    Phase 2 行为：模拟验证逻辑。
    Phase 3 升级：真实 LLM 用 loser 证据挑战 winner 结论。

    参数:
        state: 含已仲裁的 conflicts

    返回:
        dict — 含验证后 conflicts（带 verification_result 和 resolved 标记）
    """
    all_conflicts = state.get("conflicts", [])

    # 找出已仲裁但未验证的冲突
    unverified = [
        c for c in all_conflicts
        if c.get("arbitration_result") and not c.get("verification_result")
    ]

    if not unverified:
        return {"conflicts": all_conflicts}

    verified_results = []

    for conflict in unverified:
        arb = conflict["arbitration_result"]
        winner = arb["winner"]
        rule_applied = arb["rule_applied"]

        # ── 验证逻辑 ──
        # Phase 2 模拟: 规则引擎裁决 → 直接通过（因为是确定性规则）
        # RULE_ENGINE 方法的裁决不需要再验证——规则是正确的
        if arb["method"] == ResolutionMethod.RULE_ENGINE.value:
            verified = True
            detail = (
                f"VERIFIED: {rule_applied} is a deterministic rule based on "
                f"GDPR Article 83 penalty tiers. Winner ({winner}) carries "
                f"higher regulatory weight."
            )
        else:
            # LLM_CONTEXTUAL: Phase 2 模拟通过
            # Phase 3: 真实 LLM 用 loser 证据挑战 winner
            verified = True
            detail = (
                f"[SIMULATED VERIFICATION] Contextual decision favoring {winner} "
                f"appears reasonable. Phase 3 will perform adversarial verification."
            )

        verified_results.append({
            **conflict,
            "verification_result": {
                "verified": verified,
                "verification_detail": detail,
                "round": arb["round"],
            },
            "resolved": verified,  # 验证通过 → 标记已解决
        })

    # 合并回原列表（保持未参与验证的冲突不变）
    verified_ids = {c["conflict_id"] for c in verified_results}
    all_updated = []
    for c in all_conflicts:
        if c["conflict_id"] in verified_ids:
            matched = next(
                v for v in verified_results
                if v["conflict_id"] == c["conflict_id"]
            )
            all_updated.append(matched)
        else:
            all_updated.append(c)

    return {"conflicts": all_updated}


# ═══════════════════════════════════════════════════════════
# 节点：证据精炼
# ═══════════════════════════════════════════════════════════

def evidence_refinement_node(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    证据精炼节点 — 当证据不足时，做定向 RAG 补充。

    循环回边 #2: 证据不足 → 定向搜索 → 补充证据 → 重新审计
    最大轮次由 MAX_RETRIEVAL_ROUNDS 控制。

    Phase 2 行为：标记证据缺口，生成补充建议。
    Phase 3 升级：调用 RAG 搜索引擎做实际补充。

    参数:
        state: 含 evidence_gaps 的状态

    返回:
        dict — 补充的 evidence_supplements
    """
    gaps = state.get("evidence_gaps", [])
    retrieval_round = state.get("evidence_retrieval_round", 0) + 1
    max_rounds = state.get("MAX_RETRIEVAL_ROUNDS", 2)

    if not gaps:
        return {"evidence_retrieval_round": retrieval_round}

    supplements = []

    for gap in gaps:
        # Phase 2 模拟补充证据
        # Phase 3: 调用 RAG search_gdpr_knowledge(gap_keywords)
        supplement = {
            "source": "evidence_refinement",
            "supplement_id": f"SUPP-{gap.get('gap_id', 'UNKNOWN')}",
            "gap_reference": gap.get("gap_id", ""),
            "round": retrieval_round,
            "content": (
                f"[SIMULATED - Phase 3 RAG] Targeted search for: "
                f"{gap.get('reason', 'unknown gap')}. "
                f"Recommended action: {gap.get('recommended_action', 're_run')}"
            ),
            "sufficient": retrieval_round >= max_rounds,
        }
        supplements.append(supplement)

    return {
        "evidence_supplements": supplements,
        "evidence_retrieval_round": retrieval_round,
    }


# ═══════════════════════════════════════════════════════════
# 节点：冲突已解决（壳节点）
# ═══════════════════════════════════════════════════════════

def conflict_resolved_node(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    冲突消解完成节点。

    确认所有冲突已处理，汇总消解结果。
    无状态变更——此节点是流程标记。
    """
    resolved_count = sum(
        1 for c in state.get("conflicts", [])
        if c.get("resolved", False)
    )
    total_count = len(state.get("conflicts", []))

    return {
        "conflict_detected": total_count > 0,
        # 如果有未解决的冲突，保持标记
        "_all_conflicts_resolved": resolved_count == total_count,
    }


# ═══════════════════════════════════════════════════════════
# 路由函数
# ═══════════════════════════════════════════════════════════

def route_conflict(state: GDPRPrivacyAuditStateV2_2) -> str:
    """
    冲突检测后的三路路由。

    返回:
        "no_conflict" — 无需消解，直接到 resolved
        "has_conflict" — 检测到冲突，进入仲裁
        "evidence_gap" — 证据不足，先补充证据
    """
    has_conflict = state.get("conflict_detected", False)
    has_gaps = len(state.get("evidence_gaps", [])) > 0

    if has_gaps and state.get("evidence_retrieval_round", 0) < state.get("MAX_RETRIEVAL_ROUNDS", 2):
        return "evidence_gap"
    if has_conflict:
        return "has_conflict"
    return "no_conflict"


def route_verification(state: GDPRPrivacyAuditStateV2_2) -> str:
    """
    验证后的三路路由。

    返回:
        "verified" — 所有冲突已解决
        "retry" — 有未解决的冲突且未达最大轮次 → 循环回边 #1
        "escalate" — 达到最大轮次仍有未解决冲突 → 升级
    """
    all_conflicts = state.get("conflicts", [])
    if not all_conflicts:
        return "verified"

    round_num = state.get("conflict_resolution_round", 0)
    max_rounds = state.get("MAX_CONFLICT_ROUNDS", 2)

    # 全部已解决 → 通过
    if all(c.get("resolved", False) for c in all_conflicts):
        return "verified"

    # 达到最大轮次 → 升级到人工处理
    if round_num >= max_rounds:
        return "escalate"

    # 还有未解决的 → 重试（换推断策略）
    return "retry"


def route_refinement(state: GDPRPrivacyAuditStateV2_2) -> str:
    """
    证据精炼后的两路路由。

    返回:
        "done" — 精炼完成，回到冲突检测
        "retry" — 需要继续补充 → 循环回边 #2
    """
    round_num = state.get("evidence_retrieval_round", 0)
    max_rounds = state.get("MAX_RETRIEVAL_ROUNDS", 2)

    if round_num >= max_rounds:
        return "done"
    return "retry"


# ═══════════════════════════════════════════════════════════
# 子图构建
# ═══════════════════════════════════════════════════════════

def build_conflict_subgraph() -> StateGraph:
    """
    构建并返回 Conflict Resolution 子图。

    这个子图被编译后作为主图的一个节点使用。
    它是 LangGraph "子图嵌套"概念的演示。

    返回:
        编译后的 StateGraph（可作为父图节点）
    """
    builder = StateGraph(GDPRPrivacyAuditStateV2_2)

    # ── 注册节点 ──
    builder.add_node("conflict_detector", conflict_detector_node)
    builder.add_node("arbitration_agent", arbitration_agent_node)
    builder.add_node("verification_agent", verification_agent_node)
    builder.add_node("evidence_refinement", evidence_refinement_node)
    builder.add_node("conflict_resolved", conflict_resolved_node)

    # ── 连接边 ──
    builder.add_edge(START, "conflict_detector")

    # 冲突检测 → 三路路由
    builder.add_conditional_edges(
        "conflict_detector",
        route_conflict,
        {
            "no_conflict": "conflict_resolved",
            "has_conflict": "arbitration_agent",
            "evidence_gap": "evidence_refinement",
        }
    )

    # 仲裁 → 验证
    builder.add_edge("arbitration_agent", "verification_agent")

    # 验证 → 三路路由（循环回边 #1: retry → arbitration_agent）
    builder.add_conditional_edges(
        "verification_agent",
        route_verification,
        {
            "verified": "conflict_resolved",
            "retry": "arbitration_agent",     # ← 循环回边 #1
            "escalate": "conflict_resolved",  # Phase 4: 改为 escalation → HITL
        }
    )

    # 证据精炼 → 两路路由（循环回边 #2: retry → evidence_refinement）
    builder.add_conditional_edges(
        "evidence_refinement",
        route_refinement,
        {
            "done": "conflict_resolved",       # 回到冲突检测
            "retry": "evidence_refinement",    # ← 循环回边 #2
        }
    )

    # 完成 → END
    builder.add_edge("conflict_resolved", END)

    return builder.compile()


# ═══════════════════════════════════════════════════════════
# 对外接口
# ═══════════════════════════════════════════════════════════

# 子图实例 — 由 graph.py 作为节点使用
conflict_subgraph = build_conflict_subgraph()
