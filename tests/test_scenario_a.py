"""
测试场景 A：仅文档输入（无 SQL）。

输入: 1 份隐私声明文档
预期流程:
  START → init_node → evidence_supervisor
    → Fan-Out: 仅 privacy_doc_auditor（无 data_schema_auditor）
    → Fan-In: 仅 1 个 Agent 的 findings
    → conflict_detector: 无冲突（只有 1 个 Agent 的输出，无法对比）
    → conflict_resolved
    → synthesis_agent: 综合 1 个 Agent 的发现
    → risk_rater: MEDIUM 风险（有 FAIL 发现但不满足 HIGH 条件）
    → dpia_generator: 生成 DPIA
    → reflection_agent: 评分 DPIA
    → final_report: 生成最终报告（含法规 footer）
    → END

验证点:
  1. 图结构正确 — 仅 1 个 Agent 执行（无 Fan-Out）
  2. 无冲突检测（单 Agent 场景）
  3. 风险等级 MEDIUM（无 HIGH 发现）
  4. DPIA 质量评分 > 0.85
  5. 报告包含法规版本 footer
  6. 无 HITL 触发

运行方式:
  cd GDPR_Privacy_Auditor_Agent
  python -m pytest tests/test_scenario_a.py -v
"""

import os
import sys
import json

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import (
    create_initial_state,
    GDPRPrivacyAuditStateV2_2,
    AuditPhase,
    RiskTier,
)
from graph import build_graph, run_audit


# ═══════════════════════════════════════════════════════════
# 测试数据
# ═══════════════════════════════════════════════════════════

SAMPLE_PRIVACY_POLICY = """
Privacy Policy for TestShop Inc.

We at TestShop Inc. ("we", "our", "us") collect the following personal data:
- Email address for account creation and order confirmation
- Full name for order fulfillment
- Billing address for payment processing and tax compliance

We process your data for the following purposes:
- Account management and authentication (legal basis: contract necessity)
- Order processing and fulfillment (legal basis: contract necessity)
- Service improvement and analytics (legal basis: legitimate interest)

We do NOT sell your personal data. We share data with:
- Payment processors (Stripe, PayPal) for transaction processing
- Cloud hosting providers (AWS eu-west-1, Germany) for infrastructure

Your data is retained for:
- Account data: duration of account + 30 days after closure
- Order data: 7 years for tax compliance (legal obligation)

Your rights under GDPR:
- Right to access your data
- Right to rectification
- Right to erasure ("right to be forgotten")
- Right to data portability
- Right to object to processing

For EU/EEA users: Our lead supervisory authority is the Irish DPC.
We serve users globally, including the European Union.

For questions about your privacy, contact: dpo@testshop.example.com
Last updated: 2025-06-01
"""


# ═══════════════════════════════════════════════════════════
# 测试用例
# ═══════════════════════════════════════════════════════════

class TestScenarioADocumentOnly:
    """测试场景 A：仅文档输入。"""

    def test_graph_builds(self):
        """验证图可以正常编译。"""
        graph = build_graph()
        assert graph is not None
        assert len(graph.nodes) >= 10  # 至少 10 个节点

    def test_run_audit_document_only(self):
        """验证完整审计流程（仅文档输入）。"""
        result = run_audit(
            target_name="TestShop E-Commerce",
            target_description="A small online retail platform.",
            privacy_documents=[{
                "name": "privacy_policy.md",
                "content": SAMPLE_PRIVACY_POLICY,
            }],
            data_schemas=[],  # 无 SQL 输入
            document_date="2025-06-01",
        )

        # 验证阶段完整性
        assert result["phase"] == AuditPhase.COMPLETED.value, \
            f"Expected COMPLETED, got {result['phase']}"

        # 验证有证据产生
        assert len(result["evidence"]) > 0, "Should have evidence from Privacy Auditor"

        # 验证有发现
        assert len(result["findings"]) > 0, "Should have findings from Privacy Auditor"

        # 验证无冲突（单 Agent，无法对比）
        assert result["conflict_detected"] == False, \
            "Single agent should not detect conflicts"

        # 验证风险等级
        risk = result["risk_tier"]
        assert risk in (RiskTier.MEDIUM.value, RiskTier.LOW.value), \
            f"Risk should be MEDIUM or LOW for doc-only input, got {risk}"

        # 验证不需要人审（非 HIGH 风险）
        assert result["needs_human_review"] == False, \
            "Doc-only should not trigger human review"

        # 验证 DPIA 已生成
        dpia = result.get("dpia_report", {})
        assert len(dpia) > 0, "DPIA report should not be empty"

        # 验证 DPIA 质量评分
        score = result.get("dpia_quality_score", 0)
        assert score > 0.0, f"DPIA quality score should be > 0, got {score}"

        # 验证最终报告
        report = result.get("report_text", "")
        assert len(report) > 500, f"Report should be substantial, got {len(report)} chars"
        assert "GDPR PRIVACY AUDIT REPORT" in report
        assert "TestShop E-Commerce" in report or "TestShop" in report

        # 验证法规版本 footer
        assert "2016/679" in report, "Report should include regulation version"

        # 验证无 DPO 决策记录（未触发 HITL）
        dpo = result.get("dpo_decision", {})
        assert dpo.get("action", "none") == "none" or dpo == {}, \
            "Should not have DPO decision when no HITL triggered"

    def test_findings_have_required_fields(self):
        """验证每条 finding 都有必需的字段。"""
        result = run_audit(
            target_name="Test",
            privacy_documents=[{
                "name": "privacy.md",
                "content": SAMPLE_PRIVACY_POLICY,
            }],
        )

        for finding in result["findings"]:
            assert "finding_id" in finding, "Each finding needs a finding_id"
            assert "source" in finding, "Each finding needs a source"
            assert "state" in finding, "Each finding needs a state"
            assert "severity" in finding, "Each finding needs a severity"
            assert "title" in finding, "Each finding needs a title"
            assert "description" in finding, "Each finding needs a description"
            assert "related_articles" in finding, "Each finding needs related_articles"

            # 来源必须来自 Privacy Doc Auditor
            assert finding["source"] == "privacy_doc_auditor", \
                f"Doc-only scenario: all findings from privacy_doc_auditor, got {finding['source']}"

    def test_evidence_has_correct_structure(self):
        """验证每条 evidence 都有正确的结构。"""
        result = run_audit(
            target_name="Test",
            privacy_documents=[{
                "name": "privacy.md",
                "content": SAMPLE_PRIVACY_POLICY,
            }],
        )

        for ev in result["evidence"]:
            assert "source" in ev, "Each evidence needs a source"
            assert "evidence_id" in ev, "Each evidence needs an evidence_id"
            assert "type" in ev, "Each evidence needs a type"

            # 来源必须来自 Privacy Doc Auditor
            assert ev["source"] == "privacy_doc_auditor", \
                f"Evidence source mismatch: {ev['source']}"

    def test_report_contains_all_sections(self):
        """验证最终报告包含所有必需章节。"""
        result = run_audit(
            target_name="TestShop",
            privacy_documents=[{
                "name": "privacy.md",
                "content": SAMPLE_PRIVACY_POLICY,
            }],
        )

        report = result["report_text"]

        required_sections = [
            "GDPR PRIVACY AUDIT REPORT",
            "EXECUTIVE SUMMARY",
            "FINDINGS",
            "DPIA QUALITY ASSESSMENT",
        ]

        for section in required_sections:
            assert section in report, \
                f"Report missing section: '{section}'"


# ═══════════════════════════════════════════════════════════
# 直接运行
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Scenario A: Document-only Input")
    print("=" * 60)

    # 运行审计
    result = run_audit(
        target_name="TestShop E-Commerce",
        target_description="A small online retail platform.",
        privacy_documents=[{
            "name": "privacy_policy.md",
            "content": SAMPLE_PRIVACY_POLICY,
        }],
        data_schemas=[],
        document_date="2025-06-01",
    )

    print(f"\nPhase: {result['phase']}")
    print(f"Risk Tier: {result['risk_tier']}")
    print(f"Total Findings: {len(result['findings'])}")
    print(f"Total Evidence: {len(result['evidence'])}")
    print(f"Conflicts: {len(result['conflicts'])}")
    print(f"Needs Human Review: {result['needs_human_review']}")
    print(f"DPIA Score: {result['dpia_quality_score']}")
    print(f"Report Length: {len(result['report_text'])} chars")
    print()

    # 展示报告摘要
    print("Report Preview (first 600 chars):")
    print("-" * 40)
    print(result['report_text'][:600])
    print("...")
