"""
Privacy Doc Auditor 工具集（4 个工具）。

这些工具在 Agent 的 ReAct 循环中被 LLM 调用：
  Think: "我需要检查隐私政策中的同意语言是否符合 Art.7"
  Act:   invoke check_consent_language(policy_text)
  Observe: 返回结构化分析结果
  Think: "同意语言模糊，需要查 GDPR Art.7 的具体要求"
  Act:   invoke search_gdpr_knowledge("consent requirements Art.7")
  ...

工具列表：
  1. search_gdpr_knowledge    — RAG 语义搜索（所有 Agent 共用）
  2. analyze_privacy_text     — 隐私声明完整性结构化检查
  3. check_consent_language   — 同意语言 GDPR 合规分析
  4. extract_declared_categories — 提取政策中声明的数据类别清单

LangGraph/LangChain 知识点：
  - 每个工具用 @tool 装饰器声明（LangChain 兼容）
  - 工具在 create_agent 时注入，LLM 自主决定调用顺序
  - 工具返回值被 Agent 的 ReAct 循环用作 Observation
"""

import re
from typing import Optional
from langchain.tools import tool

# RAG 搜索后端
from rag.search import search_gdpr_knowledge as _rag_search


# ═══════════════════════════════════════════════════════════
# Tool 1: RAG 知识搜索（共用）
# ═══════════════════════════════════════════════════════════

@tool
def search_gdpr_knowledge(
    query: str,
    article: Optional[str] = None,
    topic: Optional[str] = None,
) -> str:
    """
    搜索 GDPR 知识库获取法规正文、EDPB 指南和执法案例。

    当需要查询 GDPR 条款具体要求、合规标准或相关案例时调用此工具。
    每次调用自动返回带版本元数据的结果。

    参数:
        query: 自然语言查询，如 "consent requirements for marketing"
        article: 可选，指定条款号如 "7" 只看 Art.7
        topic: 可选，指定主题如 "consent" / "cross_border" / "dpia"

    返回:
        格式化的搜索结果文本（含法规版本信息）
    """
    results = _rag_search(
        query=query,
        filter_article=article,
        filter_topic=topic,
        n_results=5,
    )

    if not results:
        return "No relevant GDPR knowledge found for the query."

    output_parts = []
    for i, r in enumerate(results):
        meta = r.get("metadata", {})
        collection = r.get("collection", "unknown")
        score = r.get("score", 0)

        # 格式化版本信息
        version_info = ""
        if meta.get("article"):
            version_info += f"Art.{meta['article']}"
        if meta.get("version"):
            version_info += f" ({meta['version']})"
        if meta.get("effective_date"):
            version_info += f" [effective: {meta['effective_date']}]"
        if meta.get("regulation_id"):
            version_info += f" | {meta['regulation_id']}"

        output_parts.append(
            f"--- Result {i+1} (score: {score:.2f}, source: {collection}) ---\n"
            f"Version: {version_info}\n"
            f"{r['content'][:1500]}"
        )

    return "\n\n".join(output_parts)


# ═══════════════════════════════════════════════════════════
# Tool 2: 隐私声明完整性分析
# ═══════════════════════════════════════════════════════════

@tool
def analyze_privacy_text(text: str) -> str:
    """
    分析一段隐私声明文本的 GDPR 合规完整性。

    检查隐私声明是否包含以下必要元素：
      - 数据控制者身份和联系方式
      - 处理目的和法律基础
      - 数据接收者或接收者类别
      - 跨境传输声明（如适用）
      - 数据保留期限
      - 数据主体权利（访问/更正/删除/携带）
      - 自动化决策说明（如适用）

    参数:
        text: 隐私声明文本

    返回:
        JSON 格式的结构化分析结果
    """
    import json

    checks = {
        "controller_identity": {
            "found": False,
            "evidence": "",
            "requirement": "Art.13(1)(a) — 控制者身份和联系方式",
        },
        "processing_purposes": {
            "found": False,
            "evidence": "",
            "requirement": "Art.13(1)(c) — 处理目的和法律基础",
        },
        "data_recipients": {
            "found": False,
            "evidence": "",
            "requirement": "Art.13(1)(e) — 数据接收者",
        },
        "cross_border_transfer": {
            "found": False,
            "evidence": "",
            "requirement": "Art.13(1)(f) — 跨境传输声明",
        },
        "retention_period": {
            "found": False,
            "evidence": "",
            "requirement": "Art.13(2)(a) — 数据保留期限",
        },
        "data_subject_rights": {
            "found": False,
            "evidence": "",
            "requirement": "Art.13(2)(b-d) — 数据主体权利",
        },
        "automated_decision": {
            "found": False,
            "evidence": "",
            "requirement": "Art.13(2)(f) — 自动化决策",
        },
    }

    text_lower = text.lower()

    # 检查控制者身份
    if any(kw in text_lower for kw in ["we ", "our ", "company ", "controller", "data controller"]):
        checks["controller_identity"]["found"] = True
        checks["controller_identity"]["evidence"] = "Controller identity mentioned in text"

    # 检查处理目的
    purpose_kw = ["purpose", "use", "collect", "process", "处理", "目的", "用于"]
    if any(kw in text_lower for kw in purpose_kw):
        checks["processing_purposes"]["found"] = True
        checks["processing_purposes"]["evidence"] = "Processing purposes described"

    # 检查法律基础
    legal_basis_kw = ["consent", "同意", "contract", "合同", "legitimate interest", "合法权益",
                      "legal obligation", "法定义务", "vital interest"]
    found_bases = [kw for kw in legal_basis_kw if kw in text_lower]
    if found_bases:
        checks["processing_purposes"]["evidence"] += f" | Legal bases mentioned: {found_bases}"

    # 检查数据接收者
    recipient_kw = ["third party", "第三方", "service provider", "partner", "affiliate", "subsidiary",
                    "recipient", "receiver", "share"]
    if any(kw in text_lower for kw in recipient_kw):
        checks["data_recipients"]["found"] = True
        checks["data_recipients"]["evidence"] = "Data sharing / recipients mentioned"

    # 检查跨境传输
    transfer_kw = ["international", "cross-border", "跨境", "transfer", "outside", "third country",
                   "eu", "us", "united states", "europe", "adequate", "adequacy", "safeguard"]
    if any(kw in text_lower for kw in transfer_kw):
        checks["cross_border_transfer"]["found"] = True
        checks["cross_border_transfer"]["evidence"] = "Cross-border transfer mentioned"

    # 检查保留期限
    retention_kw = ["retain", "保留", "retention", "period", "delete", "删除", "store", "储存",
                    "as long as", "until"]
    if any(kw in text_lower for kw in retention_kw):
        checks["retention_period"]["found"] = True
        checks["retention_period"]["evidence"] = "Data retention mentioned"

    # 检查数据主体权利
    rights_kw = ["right to", "access", "rectif", "eras", "delet", "portab", "object",
                 "权利", "访问", "更正", "删除", "携带", "反对"]
    found_rights = [kw for kw in rights_kw if kw in text_lower]
    if len(found_rights) >= 3:
        checks["data_subject_rights"]["found"] = True
        checks["data_subject_rights"]["evidence"] = f"Data subject rights mentioned: {found_rights}"

    # 检查自动化决策
    auto_kw = ["automated decision", "自动决策", "profiling", "画像", "algorithm"]
    if any(kw in text_lower for kw in auto_kw):
        checks["automated_decision"]["found"] = True
        checks["automated_decision"]["evidence"] = "Automated decision-making mentioned"

    # 计算完整度
    total_items = len(checks)
    found_items = sum(1 for c in checks.values() if c["found"])
    completeness = found_items / total_items if total_items > 0 else 0

    result = {
        "completeness_score": round(completeness, 2),
        "items_found": found_items,
        "items_total": total_items,
        "checks": checks,
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════
# Tool 3: 同意语言分析
# ═══════════════════════════════════════════════════════════

@tool
def check_consent_language(text: str) -> str:
    """
    分析隐私声明中的同意语言是否符合 GDPR Art.7 要求。

    检查以下模式：
      - 捆绑同意：使用服务即同意 ("by using this service...")
      - 预选框默认同意
      - 模糊同意 ("we may use..." / "your data might be...")
      - 不可撤回 ("consent cannot be withdrawn")
      - 区分度不足：同意请求未与其他条款区分

    参数:
        text: 隐私声明文本或同意相关段落

    返回:
        JSON 格式的同意语言分析结果
    """
    import json

    text_lower = text.lower()

    issues = []

    # ── 检查 1: 捆绑同意 ──
    bundled_patterns = [
        (r"by\s+using\s+(this|our|the)\s+(service|site|app|platform|website)",
         "bundled_consent", "HIGH",
         "使用服务即构成同意——违反 Art.7(4) 捆绑同意禁令"),
        (r"continuing\s+to\s+use",
         "bundled_consent", "HIGH",
         "继续使用即同意——未提供真正的选择权"),
        (r"by\s+creating\s+an\s+account",
         "bundled_consent", "HIGH",
         "创建账户即同意——服务访问以同意为条件"),
    ]

    for pattern, category, severity, description in bundled_patterns:
        if re.search(pattern, text_lower):
            issues.append({
                "type": category,
                "severity": severity,
                "pattern_matched": pattern,
                "description": description,
            })

    # ── 检查 2: 模糊语言 ──
    vague_patterns = [
        (r"we\s+may\s+(use|collect|share|process)",
         "vague_language", "MEDIUM",
         "'We may...' 表述模糊，未明确说明具体处理活动"),
        (r"as\s+described\s+(in|below|above|herein)",
         "vague_language", "MEDIUM",
         "笼统引用 'as described' 而非在同意请求中明确说明"),
        (r"improve\s+(our|the|your)\s+(service|experience|product)",
         "vague_purpose", "MEDIUM",
         "'improve our service' 目的过于宽泛，不符合目的明确性要求"),
    ]

    for pattern, category, severity, description in vague_patterns:
        if re.search(pattern, text_lower):
            issues.append({
                "type": category,
                "severity": severity,
                "pattern_matched": pattern,
                "description": description,
            })

    # ── 检查 3: 正面标志 ──
    positive_indicators = []
    positive_patterns = [
        (r"(opt[-\s]?in)|(explicit\s+consent)|(freely\s+given)",
         "明确的 opt-in 或 explicit consent 表述"),
        (r"withdraw\s+(your\s+)?consent|revoke\s+consent",
         "提到了撤回同意权"),
        (r"separate\s+(consent|agreement)|independent\s+of",
         "同意请求与其他事项区分"),
        (r"(do\s+not\s+sell|do\s+not\s+track|do\s+not\s+share)",
         "明确的 opt-out 选项"),
    ]

    for pattern, description in positive_patterns:
        if re.search(pattern, text_lower):
            positive_indicators.append(description)

    # ── 计算风险评分 ──
    high_issues = sum(1 for i in issues if i["severity"] == "HIGH")
    medium_issues = sum(1 for i in issues if i["severity"] == "MEDIUM")

    if high_issues > 0:
        risk_level = "HIGH"
    elif medium_issues > 0:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    result = {
        "risk_level": risk_level,
        "total_issues": len(issues),
        "high_severity_issues": high_issues,
        "medium_severity_issues": medium_issues,
        "positive_indicators": positive_indicators,
        "issues": issues,
        "summary": (
            f"Consent language analysis: {len(issues)} issues found "
            f"({high_issues} HIGH, {medium_issues} MEDIUM). "
            f"{len(positive_indicators)} positive indicators detected."
        ),
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════
# Tool 4: 提取声明的数据类别
# ═══════════════════════════════════════════════════════════

@tool
def extract_declared_categories(text: str) -> str:
    """
    从隐私政策文本中提取声明的个人数据类别。

    识别文本中明确声明的数据收集项，生成结构化清单。
    此清单后续会与 Data Schema Auditor 的 PII 扫描结果做对比（冲突检测）。

    参数:
        text: 隐私声明文本

    返回:
        JSON 格式的已声明数据类别清单
    """
    import json

    # ── 已知的 PII 类别关键词映射 ──
    category_patterns = {
        "email_address": [r"\bemail\b", r"\be-mail\b", r"电子邮件", r"邮箱"],
        "full_name": [r"\bfull\s*name\b", r"\b姓名\b", r"\bname\b"],
        "phone_number": [r"\bphone\b", r"\btelephone\b", r"\bmobile\b", r"\b电话\b", r"\b手机\b"],
        "billing_address": [r"\bbilling\b", r"\b账单地址\b", r"\baddress\b"],
        "shipping_address": [r"\bshipping\b", r"\b送货地址\b", r"\b收货地址\b"],
        "ip_address": [r"\bip\s*address\b", r"\bIP地址\b"],
        "cookie_data": [r"\bcookie\b", r"\bCookie\b"],
        "browsing_history": [r"\bbrowsing\b", r"\b浏览记录\b", r"\bbrowsing\s*history\b"],
        "purchase_history": [r"\bpurchase\b", r"\b购买记录\b", r"\border\s*history\b"],
        "payment_info": [r"\bpayment\b", r"\b支付\b", r"\bcredit\s*card\b", r"\b信用卡\b"],
        "location_data": [r"\blocation\b", r"\b位置\b", r"\bgps\b", r"\bgeolocation\b", r"\bGPS\b"],
        "device_info": [r"\bdevice\b", r"\b设备\b", r"\bIMEI\b"],
        "date_of_birth": [r"\bbirth\b", r"\b出生\b", r"\bdob\b"],
        "social_media": [r"\bsocial\b", r"\b社交\b", r"\b社交媒体\b"],
        "user_content": [r"\bcontent\b.*\bpost\b", r"\bupload\b", r"\b上传\b"],
    }

    text_lower = text.lower()
    declared_categories = {}

    for category, patterns in category_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                declared_categories[category] = {
                    "declared": True,
                    "matched_pattern": pattern,
                }
                break
        else:
            declared_categories[category] = {
                "declared": False,
                "matched_pattern": None,
            }

    # ── 汇总 ──
    declared_list = [
        cat for cat, info in declared_categories.items()
        if info["declared"]
    ]

    result = {
        "total_declared": len(declared_list),
        "declared_categories": declared_list,
        "detail": declared_categories,
        "note": (
            "This list represents categories EXPLICITLY mentioned in the privacy text. "
            "Categories may be referenced by different terms than what the database uses. "
            "Compare with Data Schema Auditor's PII scan for discrepancies."
        ),
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════
# 工具列表（供 create_agent 使用）
# ═══════════════════════════════════════════════════════════

PRIVACY_AUDITOR_TOOLS = [
    search_gdpr_knowledge,
    analyze_privacy_text,
    check_consent_language,
    extract_declared_categories,
]
