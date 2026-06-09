"""
GDPR Privacy Auditor V2.2 — 完整审计状态定义。

包含所有 TypedDict、dataclass 和 Enum 类型。
这是整个项目的类型基础，所有模块都依赖此文件。

V2.2 核心设计：
  - 2 个 Specialist Agent (Privacy Doc / Data Schema) Fan-Out 并发审计
  - 4 条循环回边（冲突消解重试、证据补充、DPO edit 重评估、DPIA Reflection）
  - 冲突消解 = GDPRPriorityEngine 规则引擎 + LLM 双层架构
  - DPIA 质量 = EDPB WP248 7 维度结构化量表 + 风险识别一票否决
  - 法规版本感知全链路追踪
  - HITL = 整份结论级审批（非逐条发现审批）
"""

from typing import TypedDict, Annotated, Optional
from datetime import datetime
from enum import Enum
import operator


# ═══════════════════════════════════════════════════════════
# 阶段控制
# ═══════════════════════════════════════════════════════════

class AuditPhase(str, Enum):
    """
    审计阶段状态机。

    流程: INIT → EVIDENCE_COLLECTION → ANALYSIS → REPORT → COMPLETED

    LangGraph 知识点：
      - 阶段状态机是 StateGraph 的核心概念
      - 每个阶段内部可以有独立的子图（如 Conflict Resolution）
      - 条件边 (conditional_edges) 根据阶段状态路由
    """
    INIT = "INIT"                          # 初始：解析输入、初始化状态
    EVIDENCE_COLLECTION = "EVIDENCE_COLLECTION"  # 证据收集：Fan-Out 两个 Agent
    ANALYSIS = "ANALYSIS"                  # 分析：冲突消解 + 综合评估
    REPORT = "REPORT"                      # 报告：DPIA 生成 + 质量评估
    COMPLETED = "COMPLETED"                # 完成：最终报告已生成


# ═══════════════════════════════════════════════════════════
# 风险等级
# ═══════════════════════════════════════════════════════════

class RiskTier(str, Enum):
    """
    审计风险等级。

    V2.2 新增 INCONCLUSIVE — DPO 驳回审计结论时使用。
    """
    LOW = "LOW"                # 低风险：可接受
    MEDIUM = "MEDIUM"          # 中等风险：建议改进
    HIGH = "HIGH"              # 高风险：需要 HITL 人审
    INCONCLUSIVE = "INCONCLUSIVE"  # 无法判定：DPO 驳回


# ═══════════════════════════════════════════════════════════
# 冲突类型
# ═══════════════════════════════════════════════════════════

class ConflictType(str, Enum):
    """
    两个 Agent 之间可能检测到的冲突类型。

    每种冲突类型对应不同的 GDPR 条款权重组合。
    """
    DATA_SCOPE_DISCREPANCY = "DATA_SCOPE_DISCREPANCY"    # 声明 vs 实际数据范围不一致
    RETENTION_MISMATCH = "RETENTION_MISMATCH"            # 声明保留期 vs 实际 TTL 不匹配
    TRANSFER_UNDECLARED = "TRANSFER_UNDECLARED"          # 实际跨境传输但政策未声明
    CONSENT_SCOPE_GAP = "CONSENT_SCOPE_GAP"              # 同意范围与实际使用不匹配


# ═══════════════════════════════════════════════════════════
# 发现状态
# ═══════════════════════════════════════════════════════════

class FindingState(str, Enum):
    """
    每一条审计发现的状态标记。

    V2.2 新增：
      - DPO_APPROVED: DPO 批准整份结论（报告级）
      - DPO_REJECTED: DPO 驳回整份结论
      - NEEDS_RECHECK: 法规更新后旧发现需复核
    """
    PASS = "PASS"                          # 通过：未发现问题
    FAIL = "FAIL"                          # 不合规：发现问题
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # 证据不足：无法判定
    EVIDENCE_GAP = "EVIDENCE_GAP"          # 证据缺失：需要补充 RAG
    NEEDS_MANUAL_REVIEW = "NEEDS_MANUAL_REVIEW"      # 需要人工审核
    DPO_APPROVED = "DPO_APPROVED"          # DPO 已批准
    DPO_REJECTED = "DPO_REJECTED"          # DPO 已驳回
    NEEDS_RECHECK = "NEEDS_RECHECK"        # 法规更新，需复核


# ═══════════════════════════════════════════════════════════
# 冲突消解方式
# ═══════════════════════════════════════════════════════════

class ResolutionMethod(str, Enum):
    """
    冲突消解的两种方法。

    RULE_ENGINE: 规则引擎根据 GDPR 罚款梯度权重直接裁决（80% 常规冲突）
    LLM_CONTEXTUAL: LLM 在同权重条款间做情境推断（20% 复杂冲突）

    面试重点：每次裁决都记录此方法，保证可审计、可复现。
    """
    RULE_ENGINE = "RULE_ENGINE"         # 规则引擎裁决
    LLM_CONTEXTUAL = "LLM_CONTEXTUAL"   # LLM 情境推断


# ═══════════════════════════════════════════════════════════
# V2.2 完整审计状态 (TypedDict)
# ═══════════════════════════════════════════════════════════

class GDPRPrivacyAuditStateV2_2(TypedDict):
    """
    GDPR Privacy Auditor V2.2 MVP 完整审计状态。

    LangGraph 知识点：
      - TypedDict 定义 State Schema，每个字段有明确类型
      - Annotated[list, operator.add] 实现 Fan-In 的安全合并
      - Send() API 在 dispatch_evidence_tasks 中并发派发

    相比 V2.1 MVP 的变化：
      + dpo_decision: 整份结论级审批（替代逐条审批队列）
      + regulation_versions: 法规版本追踪
      + documents_outdated: 输入文档时效性标记
      + kb_has_updates: 知识库是否有新版本
      + dpia_quality_details: WP248 各维度评分明细
      + conflict_resolution_methods: 每次仲裁的方法记录
      - human_review_queue: 不再逐条审批
    """

    # ═══ 阶段控制 ═══
    phase: str                                          # 当前审计阶段 (AuditPhase)
    phase_iterations: dict[str, int]                    # 每个阶段的迭代计数 {"ANALYSIS": 2}
    MAX_ITERATIONS: int                                 # 全局最大迭代次数，默认 3

    # ═══ 输入 ═══
    audit_id: str                                       # 审计唯一 ID，如 "AUD-20260605-001"
    target_name: str                                    # 审计目标名称，如 "E-Commerce Platform"
    target_description: str                             # 审计目标描述
    input_types: list[str]                              # 输入类型列表：["privacy_document"] | ["data_schema"] | 两者
    privacy_documents: list[dict]                       # 隐私文档列表 [{"name": "privacy.md", "content": "..."}]
    data_schemas: list[dict]                            # 数据表结构列表 [{"name": "users.sql", "content": "CREATE TABLE..."}]
    document_date: str                                  # V2.2 新增：输入文档的日期（从内容中提取，用于版本感知）

    # ═══ 证据与发现 (Fan-In 用 operator.add 安全合并) ═══
    evidence: Annotated[list, operator.add]             # 证据列表 [{"source": "privacy_doc_auditor", ...}]
    findings: Annotated[list, operator.add]             # 发现列表 [{"finding_id": "F001", "state": "FAIL", ...}]
    synthesis_summary: str                              # V2.2: 综合结论摘要（DPO 审批的对象）

    # ═══ 冲突管理 (V2.2 优化①: 双层架构) ═══
    conflicts: list[dict]                               # 冲突列表 [{"conflict_id": "C001", "conflict_type": "..."}]
    conflict_detected: bool                             # 是否检测到冲突
    conflict_resolution_round: int                      # 当前冲突消解轮次
    MAX_CONFLICT_ROUNDS: int                            # 最大冲突消解轮次，默认 2
    conflict_resolution_methods: list[dict]             # V2.2 新增：每次仲裁的方法记录

    # ═══ 证据精炼 ═══
    evidence_gaps: list[dict]                           # 证据缺失列表 [{"finding_id": "F003", "gap_type": "..."}]
    evidence_supplements: Annotated[list, operator.add] # 补充证据（Fan-In 合并）
    evidence_retrieval_round: int                       # 当前证据补充轮次
    MAX_RETRIEVAL_ROUNDS: int                           # 最大证据补充轮次，默认 2

    # ═══ 质量控制 ═══
    evidence_sufficiency: float                         # 证据充分度 (0.0 ~ 1.0)
    confidence_score: float                             # 整体置信度 (0.0 ~ 1.0)

    # ═══ 风险评估 ═══
    risk_tier: str                                      # 风险等级 (RiskTier)，V2.2 增加 INCONCLUSIVE
    critical_findings_count: int                        # 严重发现数量
    has_special_category_data: bool                     # 是否涉及特殊类别数据 (Art.9)
    cross_border_risk_level: str                        # 跨境风险等级: HIGH/MEDIUM/LOW（替代旧 bool）

    # ═══ DPIA Reflection (V2.2 优化②: WP248 量表) ═══
    dpia_report: dict                                   # DPIA 报告内容 {"systematic_description": "...", ...}
    dpia_iteration: int                                 # DPIA 迭代次数
    dpia_quality_score: float                           # DPIA 质量评分 (0.0 ~ 1.0)
    dpia_dimensions_passed: str                         # 及格维度数，如 "5/7"
    dpia_quality_details: dict                          # V2.2 新增：WP248 各维度评分明细
    reflection_feedback: str                            # Reflection 反馈文本
    MAX_DPIA_ITERATIONS: int                            # 最大 DPIA 迭代次数，默认 3

    # ═══ HITL (V2.2 简化: 整份结论级审批) ═══
    needs_human_review: bool                            # 是否需要人审
    dpo_decision: dict                                  # V2.2: DPO 整份结论级决策 {action, original_risk_tier, new_risk_tier, ...}

    # ═══ 法规版本感知 (V2.2 优化③: 新增) ═══
    regulation_versions: dict                           # 使用的法规版本 {regulation_id: {name, version, effective_date}}
    documents_outdated: list[dict]                      # 输入文档时效性 [{warning, date}]
    kb_has_updates: bool                                # 知识库是否有更新

    # ═══ 输出 ═══
    report_text: str                                    # 最终审计报告全文

    # ═══ 错误处理 ═══
    errors: Annotated[list, operator.add]               # 错误列表（Fan-In 合并）
    warnings: Annotated[list, operator.add]             # 警告列表（Fan-In 合并）

    # ═══ 内部标记 (仅路由用，不持久化) ═══
    _dpo_edited: bool                                   # DPO 是否修改了风险等级
    _dpo_rejected: bool                                 # DPO 是否驳回了审计
    _dpia_veto: bool                                    # DPIA 是否触发一票否决
    _dpia_passed: bool                                  # DPIA 质量是否达标


# ═══════════════════════════════════════════════════════════
# 辅助函数：创建初始状态
# ═══════════════════════════════════════════════════════════

def create_initial_state(
    audit_id: str,
    target_name: str,
    target_description: str,
    privacy_documents: list[dict] = None,
    data_schemas: list[dict] = None,
    document_date: str = "",
) -> GDPRPrivacyAuditStateV2_2:
    """
    创建并返回一个初始化的审计状态。

    参数:
        audit_id: 审计唯一 ID
        target_name: 审计目标名称
        target_description: 审计目标描述
        privacy_documents: 隐私文档列表（可选）
        data_schemas: 数据表结构列表（可选）
        document_date: 输入文档日期（可选，用于版本感知）

    返回:
        初始化完成的 GDPRPrivacyAuditStateV2_2
    """
    privacy_documents = privacy_documents or []
    data_schemas = data_schemas or []

    # 自动判断输入类型
    input_types = []
    if privacy_documents:
        input_types.append("privacy_document")
    if data_schemas:
        input_types.append("data_schema")

    return GDPRPrivacyAuditStateV2_2(
        # 阶段控制
        phase=AuditPhase.INIT.value,
        phase_iterations={},
        MAX_ITERATIONS=3,

        # 输入
        audit_id=audit_id,
        target_name=target_name,
        target_description=target_description,
        input_types=input_types,
        privacy_documents=privacy_documents,
        data_schemas=data_schemas,
        document_date=document_date,

        # 证据与发现
        evidence=[],
        findings=[],
        synthesis_summary="",

        # 冲突管理
        conflicts=[],
        conflict_detected=False,
        conflict_resolution_round=0,
        MAX_CONFLICT_ROUNDS=2,
        conflict_resolution_methods=[],

        # 证据精炼
        evidence_gaps=[],
        evidence_supplements=[],
        evidence_retrieval_round=0,
        MAX_RETRIEVAL_ROUNDS=2,

        # 质量控制
        evidence_sufficiency=0.0,
        confidence_score=0.0,

        # 风险评估
        risk_tier=RiskTier.LOW.value,
        critical_findings_count=0,
        has_special_category_data=False,
        cross_border_risk_level="LOW",

        # DPIA Reflection
        dpia_report={},
        dpia_iteration=0,
        dpia_quality_score=0.0,
        dpia_dimensions_passed="0/7",
        dpia_quality_details={},
        reflection_feedback="",
        MAX_DPIA_ITERATIONS=3,

        # HITL
        needs_human_review=False,
        dpo_decision={},

        # 法规版本感知
        regulation_versions={},
        documents_outdated=[],
        kb_has_updates=False,

        # 输出
        report_text="",

        # 错误处理
        errors=[],
        warnings=[],

        # 内部标记
        _dpo_edited=False,
        _dpo_rejected=False,
        _dpia_veto=False,
        _dpia_passed=False,
    )
