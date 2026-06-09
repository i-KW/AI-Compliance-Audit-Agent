"""
Data Schema Auditor 工具集（5 个工具）。

这些工具在 Agent 的 ReAct 循环中被 LLM 调用：
  Think: "数据库里有哪些 PII 字段？我需要逐个扫描"
  Act:   invoke scan_pii_columns("CREATE TABLE users (...)")
  Observe: 返回 [email, phone, IMEI, GPS, ...]
  Think: "发现了敏感字段 IMEI 和 GPS，需要查 GDPR 特殊类别要求"
  Act:   invoke search_gdpr_knowledge("special category data Art.9")

工具列表：
  1. search_gdpr_knowledge    — RAG 语义搜索（共用）
  2. scan_pii_columns         — PII 字段正则 + 语义扫描
  3. parse_sql_lineage        — SQL 数据血缘追踪 (SELECT...AS)
  4. check_retention_ttl      — 保留期 TTL 合规验证
  5. detect_cross_border_risk — 跨境传输风险检测

LangGraph/LangChain 知识点：
  - 工具是 Agent ReAct 循环中的 Action 层
  - scan_pii_columns 先用 regex 快速扫描，再用 RAG 做语义确认
  - 每个工具返回结构化结果，供 LLM 在下一步 Think 中分析
"""

import re
import json
from typing import Optional
from langchain.tools import tool

# RAG 搜索后端
from rag.search import search_gdpr_knowledge as _rag_search
from rag.search import search_keywords as _rag_keywords
from rag.collections import COLLECTION_PII_PATTERNS, COLLECTION_RETENTION_GUIDELINES


# ═══════════════════════════════════════════════════════════
# Tool 1: RAG 知识搜索（共用 — 与 Privacy Doc Auditor 相同）
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

    参数:
        query: 自然语言查询，如 "cross-border transfer requirements Art.44"
        article: 可选，指定条款号如 "44" 只看 Art.44
        topic: 可选，指定主题如 "cross_border" / "retention" / "security"

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

        version_info = ""
        if meta.get("article"):
            version_info += f"Art.{meta['article']}"
        if meta.get("version"):
            version_info += f" ({meta['version']})"
        if meta.get("effective_date"):
            version_info += f" [effective: {meta['effective_date']}]"

        output_parts.append(
            f"--- Result {i+1} (score: {score:.2f}, source: {collection}) ---\n"
            f"Version: {version_info}\n"
            f"{r['content'][:1500]}"
        )

    return "\n\n".join(output_parts)


# ═══════════════════════════════════════════════════════════
# Tool 2: PII 字段扫描
# ═══════════════════════════════════════════════════════════

# ── 内置 PII 正则模式（独立于 RAG，保证离线可用）──
BUILTIN_PII_PATTERNS = {
    "email_address": {
        "regex": r".*(email|e_mail|e-mail|mail).*",
        "sensitivity": "medium",
        "category": "contact",
    },
    "phone_number": {
        "regex": r".*(phone|mobile|tel|cell|telephone|contact_number).*",
        "sensitivity": "medium",
        "category": "contact",
    },
    "full_name": {
        "regex": r".*(full_name|first_name|last_name|surname|given_name|display_name|real_name).*",
        "sensitivity": "medium",
        "category": "identity",
    },
    "physical_address": {
        "regex": r".*(address|billing_address|shipping_address|postal_address|street|city|state|zip|postcode).*",
        "sensitivity": "medium",
        "category": "contact",
    },
    "ip_address": {
        "regex": r".*(ip_address|ipaddr|ip_addr|client_ip|remote_addr|user_ip).*",
        "sensitivity": "medium",
        "category": "network",
    },
    "date_of_birth": {
        "regex": r".*(date_of_birth|birth_date|birthday|dob|age).*",
        "sensitivity": "medium",
        "category": "identity",
    },
    "device_identifier": {
        "regex": r".*(imei|device_id|device_imei|udid|device_identifier|mac_address|serial_number).*",
        "sensitivity": "high",
        "category": "device",
    },
    "gps_location": {
        "regex": r".*(location|gps|lat|lng|longitude|latitude|geo|geolocation|coords|coordinates).*",
        "sensitivity": "high",
        "category": "location",
    },
    "browsing_behavior": {
        "regex": r".*(browsing|browser_history|page_view|clickstream|session|referrer).*",
        "sensitivity": "medium",
        "category": "behavioral",
    },
    "purchase_data": {
        "regex": r".*(purchase|order|transaction|payment|amount|price|cart|checkout).*",
        "sensitivity": "medium",
        "category": "financial",
    },
    "user_agent": {
        "regex": r".*(user_agent|ua_string|browser_string|agent).*",
        "sensitivity": "medium",
        "category": "device_fingerprint",
    },
    "consent_record": {
        "regex": r".*(consent|opt_in|opt_out|marketing_consent|gdpr_consent|agreed).*",
        "sensitivity": "medium",
        "category": "consent",
    },
    "national_id": {
        "regex": r".*(ssn|social_security|national_id|passport|tax_id|id_number).*",
        "sensitivity": "high",
        "category": "identity",
    },
    "health_data": {
        "regex": r".*(health|medical|diagnosis|prescription|patient|hospital|doctor).*",
        "sensitivity": "high",
        "category": "special_category",
    },
    "biometric_data": {
        "regex": r".*(fingerprint|face|retina|iris|voice_print|biometric|dna).*",
        "sensitivity": "high",
        "category": "special_category",
    },
}


@tool
def scan_pii_columns(sql_text: str) -> str:
    """
    扫描 SQL DDL 文本中的所有 PII（个人数据）字段。

    使用两层检测：
      1. 正则模式匹配 — 快速扫描列名
      2. RAG 语义确认 — 对于不确定的列名，查询 PII 模式知识库

    参数:
        sql_text: SQL DDL 文本（CREATE TABLE 语句等）

    返回:
        JSON 格式的 PII 扫描结果，包含每个检测到的字段的类型、敏感度和类别
    """
    # Step 1: 提取所有列名
    column_pattern = re.compile(
        r'(?:CREATE\s+TABLE|ALTER\s+TABLE).*?\((.*?)\);',
        re.IGNORECASE | re.DOTALL
    )

    # 提取 CREATE TABLE 中的列定义
    column_defs = []
    for match in column_pattern.finditer(sql_text):
        body = match.group(1)
        # 提取列名（简化：匹配 identifier 后跟 空格 + 类型）
        for line in body.split(","):
            line = line.strip()
            col_match = re.match(r'(\w+)\s+\w+', line)
            if col_match:
                column_defs.append(col_match.group(1).lower())

    if not column_defs:
        # 尝试更松散的匹配
        column_defs = re.findall(
            r'^\s*(\w+)\s+(?:VARCHAR|TEXT|INT|INTEGER|BOOLEAN|DATE|TIMESTAMP|DECIMAL|FLOAT|BIGINT|CHAR|BLOB)',
            sql_text, re.IGNORECASE | re.MULTILINE
        )
        column_defs = [c.lower() for c in column_defs]

    # Step 2: 用内置正则扫描
    pii_results = []
    pii_detected = set()

    for col_name in column_defs:
        for pii_type, config in BUILTIN_PII_PATTERNS.items():
            if re.match(config["regex"], col_name, re.IGNORECASE):
                if col_name not in pii_detected:
                    pii_results.append({
                        "column": col_name,
                        "pii_type": pii_type,
                        "sensitivity": config["sensitivity"],
                        "category": config["category"],
                        "detection_method": "regex",
                    })
                    pii_detected.add(col_name)
                break  # 一个列只匹配第一个命中的 PII 类型

    # Step 3: RAG 语义确认（对已检测到的列）
    # 用关键词搜索 PII 模式知识库，获取更详细的 PII 类型描述
    try:
        rag_results = _rag_keywords(
            keywords=pii_detected if pii_detected else ["email", "phone", "name"],
            collection_name=COLLECTION_PII_PATTERNS,
            n_results=3,
        )
    except Exception:
        rag_results = []

    # Step 4: 汇总
    high_sensitivity = [p for p in pii_results if p["sensitivity"] == "high"]
    total_pii = len(pii_results)

    result = {
        "total_columns_scanned": len(column_defs),
        "total_pii_columns": total_pii,
        "high_sensitivity_count": len(high_sensitivity),
        "pii_columns": pii_results,
        "non_pii_columns": [
            c for c in column_defs if c not in pii_detected
        ],
        "rag_confirmation": (
            f"RAG knowledge base returned {len(rag_results)} relevant PII patterns"
            if rag_results else "RAG knowledge base empty — using built-in patterns only"
        ),
        "note": (
            "PII detection based on column name patterns. Semantic analysis of "
            "column content (actual data values) may reveal additional PII. "
            "Aliased columns (SELECT email AS user_contact) may obscure PII — "
            "use parse_sql_lineage for detection."
        ),
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════
# Tool 3: SQL 数据血缘追踪
# ═══════════════════════════════════════════════════════════

@tool
def parse_sql_lineage(sql_text: str) -> str:
    """
    追踪 SQL 查询中的数据血缘——检测 SELECT...AS 重命名和 JOIN 传递。

    为什么重要：
      当 SELECT email AS user_contact 将 PII 字段重命名为非明显名称时，
      下游分析可能无法识别 user_contact 包含个人数据。
      这违反了 Art.25（设计保护）和 Art.30（处理记录）的要求。

    参数:
        sql_text: SQL 查询或 DDL 文本

    返回:
        JSON 格式的血缘追踪结果
    """
    # ── 检测 SELECT...AS 别名 ──
    alias_pattern = re.compile(
        r'SELECT\s+.*?(\w+)\s+AS\s+(\w+)',
        re.IGNORECASE | re.DOTALL
    )

    aliases = []
    for match in alias_pattern.finditer(sql_text):
        original = match.group(1).lower()
        aliased = match.group(2).lower()

        # 检查原始列是否是 PII
        is_pii = False
        for pii_type, config in BUILTIN_PII_PATTERNS.items():
            if re.match(config["regex"], original, re.IGNORECASE):
                is_pii = True
                break

        # 检查别名是否模糊了 PII 性质
        alias_is_innocuous = not any(
            re.match(config["regex"], aliased, re.IGNORECASE)
            for config in BUILTIN_PII_PATTERNS.values()
        )

        aliases.append({
            "original_column": original,
            "aliased_column": aliased,
            "original_is_pii": is_pii,
            "alias_obscures_pii": is_pii and alias_is_innocuous,
            "risk": (
                "HIGH" if (is_pii and alias_is_innocuous)
                else "MEDIUM" if is_pii
                else "LOW"
            ),
        })

    # ── 检测 JOIN 传递 ──
    join_pattern = re.compile(
        r'JOIN\s+(\w+)\s+ON\s+\w+\.(\w+)\s*=\s*\w+\.(\w+)',
        re.IGNORECASE
    )
    joins = []
    for match in join_pattern.finditer(sql_text):
        joined_table = match.group(1)
        left_key = match.group(2).lower()
        right_key = match.group(3).lower()
        joins.append({
            "joined_table": joined_table,
            "left_key": left_key,
            "right_key": right_key,
            "note": "JOIN may propagate PII across tables",
        })

    # ── 检测 subquery/CTE 中的 PII 传递 ──
    cte_pattern = re.compile(
        r'WITH\s+(\w+)\s+AS\s*\((.*?)\)',
        re.IGNORECASE | re.DOTALL
    )
    ctes = []
    for match in cte_pattern.finditer(sql_text):
        cte_name = match.group(1)
        cte_body = match.group(2)[:200]
        ctes.append({
            "cte_name": cte_name,
            "body_preview": cte_body,
        })

    # ── 汇总 ──
    high_risk_aliases = [a for a in aliases if a["risk"] == "HIGH"]
    total_aliases = len(aliases)

    result = {
        "total_aliases": total_aliases,
        "high_risk_aliases": len(high_risk_aliases),
        "aliases": aliases,
        "joins": joins,
        "ctes": ctes,
        "summary": (
            f"Lineage analysis: {total_aliases} column aliases detected, "
            f"{len(high_risk_aliases)} obscuring PII nature. "
            f"{len(joins)} JOINs, {len(ctes)} CTEs analyzed."
        ),
        "gdpr_relevance": {
            "art_25": "Obscured PII in data lineage violates data protection by design",
            "art_30": "Controller must maintain records including categories of data — obscured PII impedes this",
        },
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════
# Tool 4: 保留期 TTL 验证
# ═══════════════════════════════════════════════════════════

@tool
def check_retention_ttl(
    table_name: str = None,
    ttl_days: int = None,
    data_category: str = None,
    sql_text: str = None,
) -> str:
    """
    验证数据库表/字段的保留期 TTL 是否符合 GDPR 存储限制原则。

    可以传入具体的表名和 TTL，或传入 SQL DDL 文本自动提取。

    参数:
        table_name: 表名（可选）
        ttl_days: 当前 TTL 天数（可选）
        data_category: 数据类别如 "marketing_data"（可选）
        sql_text: SQL DDL 文本，自动提取 TTL 配置（可选）

    返回:
        JSON 格式的保留期合规评估
    """
    results = []

    # ── 如果传入了 SQL 文本，自动提取 TTL ──
    if sql_text:
        # 提取 TTL 相关配置
        ttl_patterns = re.findall(
            r'(?:ttl|retention|retain|expire|delete_after).*?(\d+)',
            sql_text, re.IGNORECASE
        )
        table_patterns = re.findall(
            r'CREATE\s+TABLE\s+(\w+)',
            sql_text, re.IGNORECASE
        )

        if ttl_patterns:
            for i, ttl_str in enumerate(ttl_patterns):
                table = table_patterns[i] if i < len(table_patterns) else "unknown"
                results.append({
                    "table": table,
                    "ttl_days": int(ttl_str),
                    "extracted_from": "sql_text",
                })

    # ── 如果传入了具体参数 ──
    if table_name and ttl_days:
        results.append({
            "table": table_name,
            "ttl_days": ttl_days,
            "data_category": data_category or "unknown",
            "extracted_from": "parameters",
        })

    if not results:
        return json.dumps({
            "error": "No TTL information provided. Pass sql_text for auto-extraction or table_name+ttl_days directly.",
        }, ensure_ascii=False)

    # ── 对每个 TTL 配置，查询保留期指南并评估 ──
    assessments = []
    for item in results:
        table = item["table"]
        actual_ttl = item["ttl_days"]

        # 查询 RAG 保留期指南
        try:
            guidelines = _rag_search(
                query=f"retention period for {item.get('data_category', table)} data",
                n_results=3,
                collections=[COLLECTION_RETENTION_GUIDELINES],
            )
        except Exception:
            guidelines = []

        # 从指南中提取建议最大保留期
        guideline_max = None
        guideline_source = "unknown"
        if guidelines:
            for g in guidelines:
                meta = g.get("metadata", {})
                max_days = meta.get("max_retention_days")
                if max_days:
                    guideline_max = int(max_days)
                    guideline_source = meta.get("guideline_source", "unknown")
                    break

        # 如果没有指南，使用内置默认值
        if guideline_max is None:
            builtin_defaults = {
                "marketing": 365,
                "financial": 2555,
                "session": 180,
                "account": 730,
            }
            for key, default in builtin_defaults.items():
                if key in table.lower() or key in (data_category or "").lower():
                    guideline_max = default
                    guideline_source = "builtin_default"
                    break

        if guideline_max is None:
            guideline_max = 365  # 保守默认
            guideline_source = "conservative_default"

        # 评估合规性
        exceeds = actual_ttl > guideline_max
        excess_factor = round(actual_ttl / guideline_max, 1) if guideline_max > 0 else 0

        if exceeds:
            if excess_factor > 3:
                severity = "HIGH"
            elif excess_factor > 1.5:
                severity = "MEDIUM"
            else:
                severity = "LOW"
        else:
            severity = "PASS"

        assessments.append({
            "table": table,
            "actual_ttl_days": actual_ttl,
            "guideline_max_days": guideline_max,
            "guideline_source": guideline_source,
            "exceeds_guideline": exceeds,
            "excess_factor": excess_factor,
            "severity": severity,
            "recommendation": (
                f"Reduce TTL from {actual_ttl} to {guideline_max} days"
                if exceeds
                else f"TTL of {actual_ttl} days is within guidelines"
            ),
        })

    # ── 汇总 ──
    has_violations = any(a["severity"] in ("HIGH", "MEDIUM") for a in assessments)

    result = {
        "assessments": assessments,
        "has_violations": has_violations,
        "total_tables_checked": len(assessments),
        "summary": (
            f"Retention TTL check: {len(assessments)} tables analyzed. "
            f"{sum(1 for a in assessments if a['severity'] != 'PASS')} violations found."
        ),
        "gdpr_basis": {
            "art_5": "Storage limitation — data shall be kept no longer than necessary",
            "art_25": "Data protection by design — implement automatic deletion mechanisms",
            "art_30": "Records must include envisaged time limits for erasure",
        },
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════
# Tool 5: 跨境传输检测
# ═══════════════════════════════════════════════════════════

@tool
def detect_cross_border_risk(sql_text: str = None, metadata: str = None) -> str:
    """
    检测数据是否存在跨境传输风险。

    检测方式：
      1. SQL 中的 region/cluster/location 配置
      2. 表元数据中的数据中心位置
      3. 备份/复制目标地域

    参数:
        sql_text: SQL DDL 文本（可选）
        metadata: 额外的元数据/配置文本（可选），如 "region: us-west-2, backup: eu-central-1"

    返回:
        JSON 格式的跨境传输风险评估
    """
    detections = []

    # ── 在 SQL 文本中搜索地域相关配置 ──
    if sql_text:
        sql_lower = sql_text.lower()

        # 地域关键词
        region_patterns = [
            (r"(us[_-]?(east|west|central|north|south)[_-]?\d*)", "USA", "HIGH"),
            (r"(eu[_-]?(west|central|north|south)[_-]?\d*)", "EU", "LOW"),
            (r"(ap[_-]?(southeast|northeast|south|east)[_-]?\d*)", "Asia-Pacific", "MEDIUM"),
            (r"(cn[_-]?\w*)", "China", "HIGH"),
            (r"(ru[_-]?\w*)", "Russia", "HIGH"),
        ]

        for pattern, region, risk in region_patterns:
            if re.search(pattern, sql_lower):
                detections.append({
                    "region": region,
                    "matched_pattern": pattern,
                    "risk_level": risk,
                    "source": "sql_text",
                    "adequacy_decision": region == "EU",  # 简化判断
                })

        # 检查是否有显式的跨境标记
        cross_border_keywords = [
            "cross_border", "跨境", "international", "foreign",
            "replica", "replication", "backup_region", "dr_region",
        ]
        for kw in cross_border_keywords:
            if kw in sql_lower:
                detections.append({
                    "region": "unknown",
                    "matched_pattern": kw,
                    "risk_level": "MEDIUM",
                    "source": "sql_text",
                    "adequacy_decision": False,
                    "note": f"Keyword '{kw}' suggests potential cross-border setup",
                })

    # ── 在元数据文本中搜索 ──
    if metadata:
        meta_lower = metadata.lower()

        # 明确的地域标记
        region_match = re.search(
            r'(?:region|location|datacenter|zone|cluster)[\s:=]+(\w+[\w-]*)',
            meta_lower
        )
        if region_match:
            region = region_match.group(1)
            # 判断风险等级
            if any(cn in region.lower() for cn in ["us", "usa", "american", "oregon", "virginia", "california"]):
                risk = "HIGH"
                adequacy = False
            elif any(eu in region.lower() for eu in ["eu", "europe", "frankfurt", "ireland", "paris", "london"]):
                risk = "LOW"
                adequacy = True
            elif any(asia in region.lower() for asia in ["cn", "china", "beijing", "shanghai", "hongkong"]):
                risk = "HIGH"
                adequacy = False
            else:
                risk = "MEDIUM"
                adequacy = False

            detections.append({
                "region": region,
                "matched_pattern": f"metadata: region={region}",
                "risk_level": risk,
                "source": "metadata",
                "adequacy_decision": adequacy,
            })

    # ── 如果没有任何检测结果 ──
    if not detections:
        return json.dumps({
            "detections": [],
            "has_cross_border_risk": False,
            "risk_level": "UNKNOWN",
            "summary": "No cross-border data transfer indicators found in the provided text. This does not guarantee absence of transfers — manual review recommended.",
            "recommendation": "Confirm data storage locations with infrastructure team.",
        }, ensure_ascii=False, indent=2)

    # ── 汇总 ──
    high_risks = [d for d in detections if d["risk_level"] == "HIGH"]
    has_high_risk = len(high_risks) > 0

    result = {
        "detections": detections,
        "total_detections": len(detections),
        "high_risk_detections": len(high_risks),
        "has_cross_border_risk": True,
        "overall_risk_level": "HIGH" if has_high_risk else "MEDIUM",
        "summary": (
            f"Cross-border analysis: {len(detections)} potential transfer indicators found. "
            f"{len(high_risks)} HIGH risk — destinations without EU adequacy decision."
        ),
        "gdpr_requirements": {
            "art_44": "Transfers only if controller/processor comply with Chapter V",
            "art_45": "Transfers on basis of adequacy decision",
            "art_46": "Transfers subject to appropriate safeguards (SCCs, BCRs)",
            "art_49": "Derogations for specific situations (explicit consent, contract necessity)",
        },
        "recommendation": (
            "For HIGH risk destinations: Verify EU adequacy decision, "
            "implement Standard Contractual Clauses (SCCs), and conduct "
            "Transfer Impact Assessment (TIA) per Schrems II requirements."
        ),
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════
# 工具列表（供 create_agent 使用）
# ═══════════════════════════════════════════════════════════

DATA_AUDITOR_TOOLS = [
    search_gdpr_knowledge,
    scan_pii_columns,
    parse_sql_lineage,
    check_retention_ttl,
    detect_cross_border_risk,
]
