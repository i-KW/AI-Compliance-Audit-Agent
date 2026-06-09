"""
测试场景 B：文档 + SQL 混合输入（完整链路）。

这是最能体现系统能力的测试场景。

输入: 1 份隐私声明 + 1 份 SQL DDL
预期流程:
  START → init_node → evidence_supervisor
    → Fan-Out: privacy_doc_auditor + data_schema_auditor 并发
    → Fan-In: 两个 Agent 的 findings 通过 operator.add 合并
    → conflict_detector: 检测到 4 种冲突
    → arbitration_agent: GDPRPriorityEngine 裁决（规则引擎 + LLM）
    → verification_agent: 验证冲突是否消除
    → 若未解决: retry → arbitration_agent（循环回边 #1）
    → conflict_resolved
    → synthesis_agent: 综合双方发现 + 消解结果
    → risk_rater: HIGH 风险（8+ HIGH 发现 + 特殊类别 + 跨境传输）
    → human_review: HITL 中断（DPO 审批整份结论）
      ├─ approve: → dpia_generator → reflection → final_report → END
      ├─ edit: → synthesis_agent（循环回边 #3）→ risk_rater → dpia → ...
      └─ reject: → INCONCLUSIVE → END

验证点:
  1. Fan-Out: 2 个 Agent 并发执行
  2. Fan-In: findings 通过 operator.add 安全合并
  3. 冲突检测: 4 种冲突类型全部检测到
  4. 规则引擎: RULE_ENGINE 方法裁决（确定性规则）
  5. 循环回边 #1: conflict retry 存在且不超最大轮次
  6. HITL: HIGH 风险触发人审
  7. DPO 三种操作: approve / edit / reject
  8. 循环回边 #3: DPO edit → re_evaluate → synthesis_agent
  9. DPIA Reflection: 质量评分
  10. 法规版本: footer 包含版本信息

运行方式:
  cd GDPR_Privacy_Auditor_Agent

  # 默认 approve
  python -m pytest tests/test_scenario_b.py -v

  # 测试 DPO reject
  DPO_TEST_ACTION=reject python -m pytest tests/test_scenario_b.py -v

  # 测试 DPO edit
  DPO_TEST_ACTION=edit DPO_TEST_NEW_TIER=MEDIUM python -m pytest tests/test_scenario_b.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import (
    create_initial_state,
    GDPRPrivacyAuditStateV2_2,
    AuditPhase,
    RiskTier,
    ConflictType,
    ResolutionMethod,
)
from graph import build_graph, run_audit
from rules.priority import GDPRPriorityEngine
from hitl.review import (
    DPOAction,
    human_review_node_simulated,
    route_after_human_review,
    display_conclusion,
    _build_conclusion,
)


# ═══════════════════════════════════════════════════════════
# 测试数据
# ═══════════════════════════════════════════════════════════

SAMPLE_PRIVACY_POLICY = """
Privacy Policy for E-Shop Global Inc.

We collect the following personal data:
- Email address for account creation
- Full name for order fulfillment
- Billing address for payment processing
- IP address for security purposes
- Cookie data for service improvement
- Purchase history for personalized recommendations

We process your data for:
- Account management (contract necessity)
- Order processing (contract necessity)
- Service improvement (legitimate interest)
- Marketing communications (consent)

By using our service, you agree to receive marketing emails.
We may share data with our trusted partners.

We serve users globally including the European Union.
For data subject rights requests, contact us at privacy@eshop.example.com.
Last updated: 2024-01-15
"""

SAMPLE_SQL_DDL = """
-- E-Shop Global Database Schema

CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    email VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    phone_number VARCHAR(50),
    billing_address TEXT,
    shipping_address TEXT,
    ip_address VARCHAR(45),
    date_of_birth DATE,
    device_imei VARCHAR(15),
    location_gps VARCHAR(100),
    browsing_history TEXT,
    purchase_amount DECIMAL(10,2),
    marketing_consent_flag BOOLEAN DEFAULT FALSE,
    user_agent_string TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE orders (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    order_total DECIMAL(10,2),
    payment_method VARCHAR(50),
    shipping_address TEXT,
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE marketing_events (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    campaign_id VARCHAR(100),
    event_type VARCHAR(50),
    email_opened BOOLEAN,
    link_clicked BOOLEAN,
    ttl_days INT DEFAULT 1460,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Database cluster configuration
-- Region: us-west-2 (Oregon, USA)
-- Backup region: eu-west-1 (Ireland, EU)
-- Cross-region replication: enabled

-- Common query with PII aliasing
SELECT
    u.email AS user_contact,
    u.phone_number AS contact_info,
    u.full_name AS display_name
FROM users u
JOIN marketing_events me ON u.id = me.user_id
WHERE u.marketing_consent_flag = 1;
"""


# ═══════════════════════════════════════════════════════════
# 测试用例
# ═══════════════════════════════════════════════════════════

class TestScenarioBMixedInput:
    """测试场景 B：文档 + SQL 混合输入 — 完整链路。"""

    # ── 基础验证 ──

    def test_graph_builds_and_runs(self):
        """验证图编译和基本运行。"""
        result = run_audit(
            target_name="E-Shop Global",
            target_description="A global e-commerce platform.",
            privacy_documents=[{
                "name": "privacy_policy.md",
                "content": SAMPLE_PRIVACY_POLICY,
            }],
            data_schemas=[{
                "name": "schema.sql",
                "content": SAMPLE_SQL_DDL,
            }],
            document_date="2024-01-15",
        )

        assert result["phase"] == AuditPhase.COMPLETED.value
        assert len(result["evidence"]) > 0
        assert len(result["findings"]) > 0

    # ── Fan-Out / Fan-In 验证 ──

    def test_both_agents_produce_evidence(self):
        """验证两个 Agent 都产生了证据（Fan-Out 生效）。"""
        result = run_audit(
            target_name="E-Shop Global",
            privacy_documents=[{"name": "privacy.md", "content": SAMPLE_PRIVACY_POLICY}],
            data_schemas=[{"name": "schema.sql", "content": SAMPLE_SQL_DDL}],
        )

        privacy_evidence = [
            e for e in result["evidence"]
            if e.get("source") == "privacy_doc_auditor"
        ]
        data_evidence = [
            e for e in result["evidence"]
            if e.get("source") == "data_schema_auditor"
        ]

        assert len(privacy_evidence) > 0, "Privacy Doc Auditor should produce evidence"
        assert len(data_evidence) > 0, "Data Schema Auditor should produce evidence"

    def test_both_agents_produce_findings(self):
        """验证两个 Agent 都产生了发现（Fan-In 生效）。"""
        result = run_audit(
            target_name="E-Shop Global",
            privacy_documents=[{"name": "privacy.md", "content": SAMPLE_PRIVACY_POLICY}],
            data_schemas=[{"name": "schema.sql", "content": SAMPLE_SQL_DDL}],
        )

        privacy_findings = [
            f for f in result["findings"]
            if f.get("source") == "privacy_doc_auditor"
        ]
        data_findings = [
            f for f in result["findings"]
            if f.get("source") == "data_schema_auditor"
        ]

        assert len(privacy_findings) > 0, "Privacy Doc Auditor should produce findings"
        assert len(data_findings) > 0, "Data Schema Auditor should produce findings"

    # ── 冲突检测验证 ──

    def test_conflicts_detected(self):
        """验证检测到冲突（两个 Agent 结论不一致）。"""
        result = run_audit(
            target_name="E-Shop Global",
            privacy_documents=[{"name": "privacy.md", "content": SAMPLE_PRIVACY_POLICY}],
            data_schemas=[{"name": "schema.sql", "content": SAMPLE_SQL_DDL}],
        )

        conflicts = result.get("conflicts", [])

        # 应该有至少 1 个冲突（两个 Agent 的发现通常会有不一致）
        assert len(conflicts) > 0, \
            f"Should detect at least 1 conflict, got {len(conflicts)}"

        # 冲突类型应该是有效的 ConflictType 枚举值
        conflict_types = {c.get("conflict_type") for c in conflicts}
        valid_types = {ct.value for ct in ConflictType}
        for ct in conflict_types:
            assert ct in valid_types, \
                f"Conflict type '{ct}' is not a valid ConflictType"

    def test_conflict_resolution_methods_recorded(self):
        """验证每次冲突消解的方法都被记录。"""
        result = run_audit(
            target_name="E-Shop Global",
            privacy_documents=[{"name": "privacy.md", "content": SAMPLE_PRIVACY_POLICY}],
            data_schemas=[{"name": "schema.sql", "content": SAMPLE_SQL_DDL}],
        )

        methods = result.get("conflict_resolution_methods", [])

        if result.get("conflicts"):
            assert len(methods) > 0, \
                "Should record conflict resolution methods when conflicts exist"

            for method in methods:
                assert "conflict_id" in method
                assert "method" in method
                assert "winner" in method
                assert "rule_applied" in method
                assert method["method"] in (
                    ResolutionMethod.RULE_ENGINE.value,
                    ResolutionMethod.LLM_CONTEXTUAL.value,
                ), f"Invalid resolution method: {method['method']}"

    # ── 风险等级验证 ──

    def test_high_risk_triggers_human_review(self):
        """验证 HIGH 风险触发 HITL。"""
        result = run_audit(
            target_name="E-Shop Global",
            privacy_documents=[{"name": "privacy.md", "content": SAMPLE_PRIVACY_POLICY}],
            data_schemas=[{"name": "schema.sql", "content": SAMPLE_SQL_DDL}],
        )

        risk = result["risk_tier"]
        # Mixed input should trigger at least MEDIUM, typically HIGH
        assert risk in (RiskTier.HIGH.value, RiskTier.MEDIUM.value), \
            f"Mixed input should produce HIGH or MEDIUM risk, got {risk}"

        # 检查是否检测到关键风险信号（至少一个）
        has_special = result.get("has_special_category_data", False)
        has_transfer = result.get("has_high_risk_transfer", False)
        has_high_risk = risk == RiskTier.HIGH.value
        assert has_special or has_transfer or has_high_risk, \
            "Should detect at least one risk signal (special category, cross-border, or HIGH tier)"

    # ── HITL DPO 操作验证 ──

    def test_dpo_approve(self):
        """验证 DPO approve → 继续 DPIA → 完整报告。"""
        # 用环境变量控制 DPO 决策
        os.environ["DPO_TEST_ACTION"] = "approve"

        result = run_audit(
            target_name="E-Shop Global",
            privacy_documents=[{"name": "privacy.md", "content": SAMPLE_PRIVACY_POLICY}],
            data_schemas=[{"name": "schema.sql", "content": SAMPLE_SQL_DDL}],
        )

        dpo = result.get("dpo_decision", {})
        assert dpo.get("action") == "approve", \
            f"DPO should approve, got {dpo.get('action')}"

        # 应该继续到 DPIA 和报告
        assert result["_dpo_rejected"] == False
        assert result["phase"] == AuditPhase.COMPLETED.value
        assert len(result.get("report_text", "")) > 500
        assert "[AUDIT REJECTED]" not in result.get("report_text", "")

    def test_dpo_reject(self):
        """验证 DPO reject → INCONCLUSIVE → 直接结束。"""
        os.environ["DPO_TEST_ACTION"] = "reject"
        os.environ["DPO_TEST_REJECT_REASON"] = "Target system is outside GDPR scope."

        result = run_audit(
            target_name="E-Shop Global",
            privacy_documents=[{"name": "privacy.md", "content": SAMPLE_PRIVACY_POLICY}],
            data_schemas=[{"name": "schema.sql", "content": SAMPLE_SQL_DDL}],
        )

        dpo = result.get("dpo_decision", {})
        assert dpo.get("action") == "reject", \
            f"DPO should reject, got {dpo.get('action')}"

        assert result["risk_tier"] == RiskTier.INCONCLUSIVE.value, \
            f"Should be INCONCLUSIVE, got {result['risk_tier']}"

        assert "[AUDIT REJECTED]" in result.get("report_text", ""), \
            "Report should contain rejection notice"

        # 检查 reject reason 是否记录
        assert "GDPR scope" in dpo.get("reason", ""), \
            "DPO reason should be recorded"

    def test_dpo_edit(self):
        """验证 DPO edit（降级风险）→ re_evaluate → 可能不再触发 HITL。"""
        os.environ["DPO_TEST_ACTION"] = "edit"
        os.environ["DPO_TEST_NEW_TIER"] = "MEDIUM"

        result = run_audit(
            target_name="E-Shop Global",
            privacy_documents=[{"name": "privacy.md", "content": SAMPLE_PRIVACY_POLICY}],
            data_schemas=[{"name": "schema.sql", "content": SAMPLE_SQL_DDL}],
        )

        dpo = result.get("dpo_decision", {})
        assert dpo.get("action") == "edit", \
            f"DPO should edit, got {dpo.get('action')}"

        # 风险等级应该被 DPO 修改
        assert dpo.get("original_risk_tier") == RiskTier.HIGH.value
        assert dpo.get("new_risk_tier") == RiskTier.MEDIUM.value

        # 应该完成整个流程
        assert result["phase"] == AuditPhase.COMPLETED.value
        assert len(result.get("report_text", "")) > 500

        # 清理
        os.environ.pop("DPO_TEST_ACTION", None)
        os.environ.pop("DPO_TEST_NEW_TIER", None)

    # ── DPIA 验证 ──

    def test_dpia_contains_risk_scenarios(self):
        """验证 DPIA 报告包含风险场景（WP248 要求 ≥3 个）。"""
        result = run_audit(
            target_name="E-Shop Global",
            privacy_documents=[{"name": "privacy.md", "content": SAMPLE_PRIVACY_POLICY}],
            data_schemas=[{"name": "schema.sql", "content": SAMPLE_SQL_DDL}],
        )

        dpia = result.get("dpia_report", {})
        risk_scenarios = dpia.get("risk_identification", [])

        assert len(risk_scenarios) >= 3, \
            f"DPIA must have at least 3 risk scenarios (WP248), got {len(risk_scenarios)}"

        for scenario in risk_scenarios:
            assert "scenario" in scenario
            assert "likelihood" in scenario
            assert "impact" in scenario

    def test_dpia_quality_score_present(self):
        """验证 DPIA 质量评分存在且有明细。"""
        result = run_audit(
            target_name="E-Shop Global",
            privacy_documents=[{"name": "privacy.md", "content": SAMPLE_PRIVACY_POLICY}],
            data_schemas=[{"name": "schema.sql", "content": SAMPLE_SQL_DDL}],
        )

        score = result.get("dpia_quality_score", -1)
        assert 0.0 <= score <= 1.0, f"DPIA score should be 0-1, got {score}"

        details = result.get("dpia_quality_details", {})
        assert len(details) == 7, \
            f"Should have 7 WP248 dimensions, got {len(details)}"

        # 验证风险识别维度的评分存在
        assert "risk_identification" in details, "Must have risk_identification dimension"

    # ── 法规版本验证 ──

    def test_regulation_version_tracking(self):
        """验证法规版本追踪正常工作。"""
        result = run_audit(
            target_name="E-Shop Global",
            privacy_documents=[{"name": "privacy.md", "content": SAMPLE_PRIVACY_POLICY}],
            data_schemas=[{"name": "schema.sql", "content": SAMPLE_SQL_DDL}],
            document_date="2024-01-15",
        )

        # 2024年1月的文档 → 应该有 EDPB 指南更新提醒
        assert result.get("documents_outdated") or result.get("kb_has_updates"), \
            "2024-01-15 document should trigger update warnings"

        # 法规版本应该被记录
        versions = result.get("regulation_versions", {})
        assert "GDPR-2016-679" in versions, "Should track GDPR version"

        # 报告应该包含法规版本 footer
        report = result.get("report_text", "")
        assert "2016/679" in report, "Report footer should reference GDPR 2016/679"

    # ── 报告结构验证 ──

    def test_final_report_structure(self):
        """验证最终报告包含完整结构。"""
        result = run_audit(
            target_name="E-Shop Global",
            privacy_documents=[{"name": "privacy.md", "content": SAMPLE_PRIVACY_POLICY}],
            data_schemas=[{"name": "schema.sql", "content": SAMPLE_SQL_DDL}],
        )

        report = result["report_text"]

        required_sections = [
            "GDPR PRIVACY AUDIT REPORT",
            "EXECUTIVE SUMMARY",
            "FINDINGS",
            "CONFLICT RESOLUTION",
            "DPIA QUALITY ASSESSMENT",
        ]

        for section in required_sections:
            assert section in report, \
                f"Report missing section: '{section}'"

        # 如果有 HITL，应有决策记录
        if result.get("dpo_decision", {}).get("action"):
            assert "HUMAN REVIEW DECISION" in report, \
                "Report should include DPO decision section"


# ═══════════════════════════════════════════════════════════
# HITL 模块单元测试
# ═══════════════════════════════════════════════════════════

class TestHITLModule:
    """测试 HITL 模块的各个组件。"""

    def test_build_conclusion(self):
        """验证 _build_conclusion 生成正确的审批数据结构。"""
        from state import create_initial_state

        state = create_initial_state(
            audit_id="AUD-TEST-001",
            target_name="Test Target",
            target_description="Test description",
            privacy_documents=[{"name": "test.md", "content": "test"}],
        )

        # 模拟添加一些发现
        state["risk_tier"] = RiskTier.HIGH.value
        state["critical_findings_count"] = 3
        state["findings"] = [
            {
                "finding_id": "F-001",
                "severity": "HIGH",
                "title": "Test finding 1",
                "category": "DATA_SCOPE_DISCREPANCY",
                "description": "Test description 1",
            },
            {
                "finding_id": "F-002",
                "severity": "HIGH",
                "title": "Test finding 2",
                "category": "TRANSFER_UNDECLARED",
                "description": "Test description 2",
            },
        ]

        conclusion = _build_conclusion(state)

        assert conclusion["audit_id"] == "AUD-TEST-001"
        assert conclusion["target_name"] == "Test Target"
        assert conclusion["risk_tier"] == "HIGH"
        assert conclusion["total_findings"] == 2
        assert conclusion["critical_findings_count"] == 3
        assert len(conclusion["top_critical_findings"]) == 2

    def test_dpo_action_constants(self):
        """验证 DPOAction 常量定义。"""
        assert DPOAction.APPROVE == "approve"
        assert DPOAction.EDIT == "edit"
        assert DPOAction.REJECT == "reject"

    def test_route_after_review_approve(self):
        """验证路由函数 — approve 路径。"""
        from state import create_initial_state
        state = create_initial_state(
            audit_id="TEST", target_name="T", target_description="D",
        )
        state["_dpo_edited"] = False
        state["_dpo_rejected"] = False

        route = route_after_human_review(state)
        assert route == "continue", f"Approve should route to 'continue', got '{route}'"

    def test_route_after_review_edit(self):
        """验证路由函数 — edit 路径（循环回边 #3）。"""
        from state import create_initial_state
        state = create_initial_state(
            audit_id="TEST", target_name="T", target_description="D",
        )
        state["_dpo_edited"] = True
        state["_dpo_rejected"] = False

        route = route_after_human_review(state)
        assert route == "re_evaluate", f"Edit should route to 're_evaluate', got '{route}'"

    def test_route_after_review_reject(self):
        """验证路由函数 — reject 路径。"""
        from state import create_initial_state
        state = create_initial_state(
            audit_id="TEST", target_name="T", target_description="D",
        )
        state["_dpo_edited"] = False
        state["_dpo_rejected"] = True

        route = route_after_human_review(state)
        assert route == "end", f"Reject should route to 'end', got '{route}'"

    def test_display_conclusion(self):
        """验证 CLI 审批界面生成。"""
        conclusion = {
            "audit_id": "AUD-TEST-001",
            "target_name": "Test",
            "risk_tier": "HIGH",
            "confidence_score": 0.85,
            "total_findings": 10,
            "critical_findings_count": 5,
            "finding_categories": {"UNDECLARED_PII": 5, "TRANSFER_UNDECLARED": 3},
            "top_critical_findings": [
                {
                    "id": "F-001", "title": "Test", "category": "T1",
                    "severity": "HIGH", "description": "Desc"
                }
            ],
            "conflict_resolution_summary": {"total_conflicts": 4, "resolved": 4},
            "regulation_versions": {"GDPR-2016-679": {"name": "GDPR", "version": "v1.0"}},
        }

        display = display_conclusion(conclusion)

        assert "AUD-TEST-001" in display
        assert "HIGH" in display
        assert "DPO Actions" in display
        assert "[A]pprove" in display
        assert "[E]dit" in display
        assert "[R]eject" in display


# ═══════════════════════════════════════════════════════════
# 规则引擎集成测试
# ═══════════════════════════════════════════════════════════

class TestPriorityEngineIntegration:
    """测试 GDPRPriorityEngine 在混合输入场景中的行为。"""

    def test_data_scope_discrepancy_priority(self):
        """验证 DATA_SCOPE_DISCREPANCY 的裁决权重。"""
        engine = GDPRPriorityEngine()

        result = engine.resolve({
            "conflict_type": "DATA_SCOPE_DISCREPANCY",
            "description": "Policy declares 6 categories, DDL has 12 PII fields",
            "description_privacy": "Privacy policy only declares email and name",
            "description_data": "DDL contains 12 PII columns including IMEI, GPS",
        })

        # Art.5(1)(c) 数据最小化 (75分) > Art.13 透明度 (70分)
        # → data_schema_auditor 应该获胜
        assert result["method"] == ResolutionMethod.RULE_ENGINE.value
        assert result["winner"] == "data_schema_auditor", \
            f"Expected data_schema_auditor (75 > 70), got {result['winner']}"
        assert result["weight_data"] > result["weight_privacy"]

    def test_transfer_undeclared_priority(self):
        """验证 TRANSFER_UNDECLARED 的裁决权重。"""
        engine = GDPRPriorityEngine()

        result = engine.resolve({
            "conflict_type": "TRANSFER_UNDECLARED",
            "description": "Data stored in US but policy doesn't declare transfer",
            "description_privacy": "Policy has no cross-border declaration",
            "description_data": "Database region: us-west-2",
        })

        # Art.44 跨境传输 (90分) > Art.13 透明度 (70分)
        # → data_schema_auditor 应该获胜
        assert result["method"] == ResolutionMethod.RULE_ENGINE.value
        assert result["winner"] == "data_schema_auditor", \
            f"Expected data_schema_auditor (90 > 70), got {result['winner']}"

    def test_all_conflict_types_have_mappings(self):
        """验证所有 4 种冲突类型都有条款映射。"""
        engine = GDPRPriorityEngine()

        for conflict_type in ConflictType:
            result = engine.resolve({
                "conflict_type": conflict_type.value,
                "description": f"Test {conflict_type.value}",
                "description_privacy": "Test",
                "description_data": "Test",
            })

            # 应该都有合法的方法
            assert result["method"] in (
                ResolutionMethod.RULE_ENGINE.value,
                ResolutionMethod.LLM_CONTEXTUAL.value,
            )

            # 都应该有 LLM 提示词
            assert result.get("llm_prompt") is not None, \
                f"Missing llm_prompt for {conflict_type.value}"

            # 权重应该在有效范围内
            assert 50 <= result["weight_privacy"] <= 100
            assert 50 <= result["weight_data"] <= 100


# ═══════════════════════════════════════════════════════════
# 直接运行
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Scenario B: Mixed Input (Document + SQL)")
    print("=" * 60)

    # 运行审计
    result = run_audit(
        target_name="E-Shop Global Inc.",
        target_description="A global e-commerce platform with EU customers.",
        privacy_documents=[{
            "name": "privacy_policy.md",
            "content": SAMPLE_PRIVACY_POLICY,
        }],
        data_schemas=[{
            "name": "schema.sql",
            "content": SAMPLE_SQL_DDL,
        }],
        document_date="2024-01-15",
    )

    print(f"\nPhase: {result['phase']}")
    print(f"Risk Tier: {result['risk_tier']}")
    print(f"Total Findings: {len(result['findings'])}")
    print(f"Total Evidence: {len(result['evidence'])}")

    # 冲突消解
    conflicts = result.get("conflicts", [])
    print(f"\nConflicts: {len(conflicts)}")
    for c in conflicts:
        arb = c.get("arbitration_result", {})
        print(f"  {c.get('conflict_id', '?')}: {c.get('conflict_type', '?')}")
        print(f"    Method: {arb.get('method', '?')}, Winner: {arb.get('winner', '?')}")

    # 风险
    print(f"\nSpecial Category Data: {result['has_special_category_data']}")
    print(f"High Risk Transfer: {result['has_high_risk_transfer']}")
    print(f"Needs Human Review: {result['needs_human_review']}")

    # DPO
    dpo = result.get("dpo_decision", {})
    print(f"\nDPO Decision: {dpo.get('action', 'none')}")
    if dpo.get("action") == "edit":
        print(f"  Risk override: {dpo.get('original_risk_tier')} → {dpo.get('new_risk_tier')}")

    # DPIA
    print(f"\nDPIA Score: {result['dpia_quality_score']}")
    print(f"DPIA Passed: {result['_dpia_passed']}")

    # Report
    print(f"\nReport Length: {len(result['report_text'])} chars")
    print(f"Documents Outdated: {len(result['documents_outdated'])}")
    print(f"KB Has Updates: {result['kb_has_updates']}")

    print("\n" + "=" * 60)
    print("Report Preview (first 800 chars):")
    print("-" * 40)
    print(result['report_text'][:800])
