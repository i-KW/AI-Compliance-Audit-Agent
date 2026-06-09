"""
GDPR 审计发现验证器 — 防幻觉。

作用：
  1. 验证 LLM 输出的 finding 引用的 GDPR 条款是否真实存在
  2. 验证引用的条款是否与 finding 的类别相关
  3. 标记/修正幻觉引用，输出验证统计

原理：
  LLM 可能编造不存在的条款号（如 "Art.105"）或不相关的条款引用。
  此模块在生成报告前做一道硬检查——不是"希望 LLM 输出正确引用"，
  而是"程序化验证引用是否合法，不合法就不让它进报告"。
"""

import re
from typing import Optional


# ═══════════════════════════════════════════════════════════
# GDPR 全部 99 个条款名录（真实性验证基础）
# ═══════════════════════════════════════════════════════════

VALID_ARTICLES: dict[str, str] = {
    "1": "Subject-matter and objectives",
    "2": "Material scope",
    "3": "Territorial scope",
    "4": "Definitions",
    "5": "Principles relating to processing of personal data",
    "6": "Lawfulness of processing",
    "7": "Conditions for consent",
    "8": "Conditions applicable to child's consent",
    "9": "Processing of special categories of personal data",
    "10": "Processing of personal data relating to criminal convictions",
    "11": "Processing which does not require identification",
    "12": "Transparent information, communication and modalities",
    "13": "Information to be provided where data collected from data subject",
    "14": "Information to be provided where data not obtained from data subject",
    "15": "Right of access by the data subject",
    "16": "Right to rectification",
    "17": "Right to erasure ('right to be forgotten')",
    "18": "Right to restriction of processing",
    "19": "Notification obligation regarding rectification or erasure",
    "20": "Right to data portability",
    "21": "Right to object",
    "22": "Automated individual decision-making, including profiling",
    "23": "Restrictions",
    "24": "Responsibility of the controller",
    "25": "Data protection by design and by default",
    "26": "Joint controllers",
    "27": "Representatives of non-EU controllers",
    "28": "Processor",
    "29": "Processing under the authority of the controller or processor",
    "30": "Records of processing activities",
    "31": "Cooperation with the supervisory authority",
    "32": "Security of processing",
    "33": "Notification of a personal data breach to the authority",
    "34": "Communication of a personal data breach to the data subject",
    "35": "Data protection impact assessment",
    "36": "Prior consultation",
    "37": "Designation of the data protection officer",
    "38": "Position of the data protection officer",
    "39": "Tasks of the data protection officer",
    "40": "Codes of conduct",
    "41": "Monitoring of approved codes of conduct",
    "42": "Certification",
    "43": "Certification bodies",
    "44": "General principle for transfers",
    "45": "Transfers on the basis of an adequacy decision",
    "46": "Transfers subject to appropriate safeguards",
    "47": "Binding corporate rules",
    "48": "Transfers not authorised by Union law",
    "49": "Derogations for specific situations",
    "50": "International cooperation",
    "51": "Supervisory authority",
    "52": "Independence",
    "53": "General conditions for members",
    "54": "Rules on the establishment",
    "55": "Competence",
    "56": "Lead supervisory authority",
    "57": "Tasks",
    "58": "Powers",
    "59": "Activity reports",
    "60": "Cooperation",
    "61": "Mutual assistance",
    "62": "Joint operations",
    "63": "Consistency mechanism",
    "64": "Opinion of the Board",
    "65": "Dispute resolution",
    "66": "Urgency procedure",
    "67": "Exchange of information",
    "68": "European Data Protection Board",
    "69": "Independence of the Board",
    "70": "Tasks of the Board",
    "71": "Reports of the Board",
    "72": "Procedure",
    "73": "Chair",
    "74": "Tasks of the Chair",
    "75": "Secretariat",
    "76": "Confidentiality",
    "77": "Right to lodge a complaint",
    "78": "Right to effective judicial remedy against a authority",
    "79": "Right to effective judicial remedy against a controller",
    "80": "Representation of data subjects",
    "81": "Suspension of proceedings",
    "82": "Right to compensation and liability",
    "83": "General conditions for imposing administrative fines",
    "84": "Penalties",
    "85": "Processing and freedom of expression",
    "86": "Processing and public access to official documents",
    "87": "Processing of the national identification number",
    "88": "Processing in the context of employment",
    "89": "Safeguards and derogations for archiving/research",
    "90": "Obligations of secrecy",
    "91": "Existing data protection rules of churches",
    "92": "Exercise of the delegation",
    "93": "Committee procedure",
    "94": "Repeal of Directive 95/46/EC",
    "95": "Relationship with Directive 2002/58/EC",
    "96": "Relationship with previously concluded Agreements",
    "97": "Commission reports",
    "98": "Review of other Union legal acts",
    "99": "Entry into force and application",
}

# GDPR 条款按章节分组（用于相关性判断）
ARTICLE_RANGES = {
    "CHAPTER_I": (1, 4),       # General provisions
    "CHAPTER_II": (5, 11),     # Principles
    "CHAPTER_III": (12, 22),   # Rights of the data subject
    "CHAPTER_IV": (23, 43),    # Controller and processor
    "CHAPTER_V": (44, 50),     # Transfers to third countries
    "CHAPTER_VI": (51, 59),    # Supervisory authorities
    "CHAPTER_VII": (60, 67),   # Cooperation and consistency
    "CHAPTER_VIII": (68, 76),  # EDPB
    "CHAPTER_IX": (77, 84),    # Remedies and penalties
    "CHAPTER_X": (85, 91),     # Specific processing situations
    "CHAPTER_XI": (92, 99),    # Final provisions
}


# ═══════════════════════════════════════════════════════════
# 类别-条款相关性映射
# ═══════════════════════════════════════════════════════════
# 每种 finding 类别预期涉及哪些条款范围。
# LLM 引用超出此范围的条款时标记为"低相关性"。

CATEGORY_ARTICLE_RANGES: dict[str, list[tuple[int, int, str]]] = {
    # ——— Privacy Auditor 类别 ———
    "CONSENT_LANGUAGE_VAGUE": [
        (7, 8, "consent conditions"),
    ],
    "VAGUE_PURPOSE": [
        (5, 6, "principles and lawfulness"),
        (12, 14, "transparency obligations"),
    ],
    "REGIONAL_SCOPE": [
        (3, 3, "territorial scope"),
        (27, 29, "representatives and processors"),
        (44, 49, "international transfers"),
    ],
    "INCOMPLETE_DECLARATION": [
        (12, 14, "transparency and information obligations"),
        (5, 5, "principles"),
    ],
    "MARKETING_AD_DISCLOSURE": [
        (5, 7, "lawfulness and consent"),
        (21, 22, "right to object and profiling"),
    ],
    "MISSING_RIGHTS": [
        (12, 22, "data subject rights (Chapter III)"),
    ],
    # ——— Data Schema Auditor 类别 ———
    "UNDECLARED_PII": [
        (4, 4, "definitions"),
        (5, 5, "principles"),
        (13, 14, "transparency"),
    ],
    "PII_OBFUSCATION": [
        (5, 5, "data minimisation principle"),
        (25, 25, "data protection by design"),
        (32, 32, "security of processing"),
    ],
    "RETENTION_EXCESSIVE": [
        (5, 5, "storage limitation principle"),
        (13, 13, "retention period disclosure"),
        (17, 19, "right to erasure and restriction"),
    ],
    "TRANSFER_UNDECLARED": [
        (44, 49, "international transfers (Chapter V)"),
        (13, 13, "transfer disclosure obligation"),
    ],
    "SPECIAL_CATEGORY_DATA": [
        (9, 10, "special categories and criminal data"),
        (4, 4, "definition of special categories"),
    ],
    # ——— 通用类别（跨所有范围） ———
    "DEFAULT": [
        (1, 99, "any GDPR article may be referenced"),
    ],
}


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════

def parse_article_number(ref: str) -> Optional[str]:
    """
    从 "Art.7(1)(a)" 或 "Art.44" 或 "Article 7" 中提取条款号 "7"。

    返回:
        条款号字符串，如 "7", "44"；无法解析时返回 None
    """
    # 匹配 "Art.7", "Art.44", "Article 25" 格式
    m = re.search(r'(?:Art\.|Article)\s*(\d+)', ref.strip())
    if m:
        return m.group(1)
    return None


def is_valid_article(ref: str) -> bool:
    """
    验证引用条款是否存在。

    "Art.7" → True (1-99 内)
    "Art.105" → False (不存在)
    """
    num = parse_article_number(ref)
    if num is None:
        return False
    return num in VALID_ARTICLES


def is_article_relevant(ref: str, category: str) -> tuple[bool, str]:
    """
    验证引用条款是否与 finding 类别相关。

    返回:
        (是否相关, 理由)
    """
    num_str = parse_article_number(ref)
    if num_str is None:
        return False, f"无法解析条款号: {ref}"

    num = int(num_str)
    ranges = CATEGORY_ARTICLE_RANGES.get(category, CATEGORY_ARTICLE_RANGES["DEFAULT"])

    for start, end, area in ranges:
        if start <= num <= end:
            return True, f"{ref} 属于 {area} 范围，与 {category} 相关"

    # 不匹配 — 尝试反向判断：章节边界
    for chapter_name, (ch_start, ch_end) in ARTICLE_RANGES.items():
        if ch_start <= num <= ch_end:
            return False, (
                f"{ref} ({VALID_ARTICLES.get(num_str, '?')}) "
                f"属于 {chapter_name}，与 {category} 类发现不相关。"
                f"预期范围: {_format_ranges(ranges)}"
            )

    return False, f"{ref} 不在 GDPR 条款范围内 (1-99)"


def _format_ranges(ranges: list[tuple[int, int, str]]) -> str:
    """格式化预期范围供错误消息使用。"""
    parts = []
    for start, end, area in ranges:
        if start == end:
            parts.append(f"Art.{start} ({area})")
        else:
            parts.append(f"Art.{start}-{end} ({area})")
    return "; ".join(parts)


# ═══════════════════════════════════════════════════════════
# Finding 验证
# ═══════════════════════════════════════════════════════════

def verify_finding(finding: dict) -> dict:
    """
    验证单条 finding 的引用真实性。

    返回:
        加强后的 finding，包含验证结果字段:
        - _verification_issues: list[str]，引用问题描述（空=无问题）
        - _verification_passed: bool
        - related_articles: 清理后的引用列表（移除了不存在的条款）
    """
    issues = []
    articles = finding.get("related_articles", [])
    category = finding.get("category", "")

    cleaned_articles = []
    for ref in articles:
        if not is_valid_article(ref):
            issues.append(f"虚假引用: {ref} — GDPR 中不存在此条款")
            continue  # 移除不存在的条款引用

        relevant, reason = is_article_relevant(ref, category)
        if not relevant:
            issues.append(f"不相关引用: {reason}")
            # 仍然保留，但标记 warning——条款存在但不相关，需要人工判断
            cleaned_articles.append(f"{ref}[需人工确认]")
        else:
            cleaned_articles.append(ref)

    passed = len(issues) == 0

    return {
        **finding,
        "related_articles": cleaned_articles,
        "_verification_issues": issues,
        "_verification_passed": passed,
    }


def verify_all_findings(findings: list[dict]) -> list[dict]:
    """
    批量验证所有 findings。

    每条 finding 加 _verification_passed 和 _verification_issues 字段。
    """
    return [verify_finding(f) for f in findings]


def get_verification_stats(findings: list[dict]) -> dict:
    """
    统计验证结果。

    返回:
        {
            "total": 9,
            "passed": 7,
            "failed": 2,
            "issues": [
                "F-001: 虚假引用: Art.105 — GDPR 中不存在此条款",
                "F-003: 不相关引用: Art.44 ...",
            ]
        }
    """
    total = len(findings)
    passed = sum(1 for f in findings if f.get("_verification_passed", True))
    failed = total - passed

    all_issues = []
    for f in findings:
        fid = f.get("finding_id", "?")
        for issue in f.get("_verification_issues", []):
            all_issues.append(f"{fid}: {issue}")

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "has_issues": failed > 0,
        "issues": all_issues,
    }
