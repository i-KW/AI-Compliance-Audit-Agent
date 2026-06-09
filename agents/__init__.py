"""
Agent 模块。

包含 2 个 Specialist Agent：
  - Privacy Doc Auditor: 分析隐私声明文档（Art.5-22）
  - Data Schema Auditor: 分析数据表结构 SQL/元数据（Art.5/25/30/32/44-49）

Phase 5: 两种 Agent 均已升级为真实 LLM ReAct 模式（DeepSeek + 工具）。
"""


# ═══════════════════════════════════════════════════════════
# 输出归一化：将 LLM 自由文本输出映射到系统预期的枚举值
# ═══════════════════════════════════════════════════════════

# 类别名归一化映射（LLM 自由输出 → 系统枚举）
CATEGORY_NORMALIZATION = {
    # Privacy Auditor 类别
    "CONSENT_BUNDLED": "CONSENT_LANGUAGE_VAGUE",
    "BUNDLED_CONSENT": "CONSENT_LANGUAGE_VAGUE",
    "捆绑同意": "CONSENT_LANGUAGE_VAGUE",
    "VAGUE_PURPOSE_DESCRIPTION": "VAGUE_PURPOSE",
    "目的不明确": "VAGUE_PURPOSE",
    "INCOMPLETE_PRIVACY_DECLARATION": "INCOMPLETE_DECLARATION",
    "声明不完整": "INCOMPLETE_DECLARATION",
    "INCOMPLETE": "INCOMPLETE_DECLARATION",
    "REGIONAL_DISCLOSURE": "REGIONAL_SCOPE",
    "TRANSFER_INSUFFICIENT": "REGIONAL_SCOPE",
    "跨境传输": "REGIONAL_SCOPE",
    "MARKETING_PURPOSE": "MARKETING_AD_DISCLOSURE",
    "营销目的": "MARKETING_AD_DISCLOSURE",
    "MISSING_DATA_SUBJECT_RIGHTS": "MISSING_RIGHTS",
    "缺少权利声明": "MISSING_RIGHTS",
    # Data Auditor 类别
    "PII_NOT_DECLARED": "UNDECLARED_PII",
    "未声明的PII": "UNDECLARED_PII",
    "UNDECLARED": "UNDECLARED_PII",
    "PII_ALIAS": "PII_OBFUSCATION",
    "PII别名": "PII_OBFUSCATION",
    "TTL_EXCEEDING": "RETENTION_EXCESSIVE",
    "保留期过长": "RETENTION_EXCESSIVE",
    "EXCESSIVE": "RETENTION_EXCESSIVE",
    "CROSS_BORDER": "TRANSFER_UNDECLARED",
    "跨境未声明": "TRANSFER_UNDECLARED",
    "UNPROTECTED_CROSS_BORDER": "TRANSFER_UNDECLARED",
    "SENSITIVE_DATA": "SPECIAL_CATEGORY_DATA",
    "敏感数据": "SPECIAL_CATEGORY_DATA",
    "HIGH_SENSITIVITY_PII": "SPECIAL_CATEGORY_DATA",
    "设备标识符": "SPECIAL_CATEGORY_DATA",
}


def normalize_findings(findings: list[dict], agent_source: str) -> list[dict]:
    """
    将 LLM 输出的 findings 归一化为系统预期的格式。

    处理：
      1. 类别名映射（自由文本 → 枚举值）
      2. 补充 source 字段
      3. 确保必填字段存在
    """
    for f in findings:
        # 归一化类别
        raw_cat = f.get("category", "")
        if raw_cat in CATEGORY_NORMALIZATION:
            f["category"] = CATEGORY_NORMALIZATION[raw_cat]

        # 补充 source
        f.setdefault("source", agent_source)

        # 确保必填字段
        f.setdefault("state", "FAIL")
        f.setdefault("severity", "MEDIUM")
        f.setdefault("title", "未命名发现")
        f.setdefault("description", "")
        f.setdefault("related_articles", [])
        f.setdefault("evidence_refs", [])

    return findings


def ensure_evidence(findings: list[dict], evidence: list[dict], agent_source: str) -> list[dict]:
    """
    如果 LLM 没有输出 evidence，从 findings 自动生成最小 evidence 条目。
    """
    if evidence:
        for e in evidence:
            e.setdefault("source", agent_source)
        return evidence

    # 从 findings 生成证据
    generated = []
    seen_types = set()
    for idx, f in enumerate(findings):
        cat = f.get("category", "UNKNOWN")
        if cat not in seen_types:
            seen_types.add(cat)
            generated.append({
                "source": agent_source,
                "evidence_id": f"EVD-AUTO-{idx + 1:03d}",
                "type": f"llm_generated_{cat.lower()}",
                "summary": f"LLM ReAct audit finding: {f.get('title', '')[:100]}",
            })

    # 兜底
    if not generated:
        generated.append({
            "source": agent_source,
            "evidence_id": "EVD-AUTO-000",
            "type": "llm_generated",
            "summary": "LLM ReAct audit completed (auto-generated evidence)",
        })

    return generated
