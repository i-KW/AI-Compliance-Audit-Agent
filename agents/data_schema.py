"""
Data Schema Auditor — 数据表结构审计 Agent（真实 LLM ReAct 模式）。

职责范围（Art.5/25/30/32/44-49）：
  1. PII 字段扫描 — 识别 email, phone, IMEI 等个人数据列
  2. 数据血缘追踪 — 检测 SELECT...AS 重命名模糊 PII 性质
  3. 保留期 TTL 验证 — 对比 GDPR 存储限制和行业指南
  4. 跨境传输检测 — 通过 region/CLUSTER 等检测跨境数据流向
  5. 提取实际 PII 清单 — 供冲突检测对比

工具：
  - search_gdpr_knowledge: RAG 语义搜索 GDPR 法规知识库
  - scan_pii_columns: PII 字段正则 + 语义扫描
  - parse_sql_lineage: SQL 血缘追踪 (SELECT...AS / JOIN)
  - check_retention_ttl: 保留期 TTL 合规验证
  - detect_cross_border_risk: 跨境传输风险检测

LangGraph 知识点：
  - 与 Privacy Doc Auditor 并行执行（Fan-Out → Send()）
  - 内部 ReAct 循环：LLM Think → 选工具 → 调用 → Observe → Think
  - 输出的 findings 通过 operator.add 与 Privacy Auditor 合并
"""

import json
import re
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from config import get_llm
from tools.data import DATA_AUDITOR_TOOLS
from state import GDPRPrivacyAuditStateV2_2
from agents import normalize_findings, ensure_evidence


# ═══════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════

DATA_AUDITOR_PROMPT = """你是一个 GDPR 数据表结构审计专家（Data Schema Auditor），专门分析 SQL DDL、表元数据和数据库配置。

## 你的职责范围（GDPR Art.5/25/30/32/44-49）

1. **PII 字段扫描** — 识别所有包含个人数据的列（email, phone, name, IMEI, GPS 等）
2. **数据血缘追踪** — 检测 SELECT...AS 重命名是否模糊了 PII 字段性质
3. **保留期 TTL 验证** — 对比每类数据的实际保留期与 GDPR 存储限制原则
4. **跨境传输检测** — 检测数据库地域配置（region/CLUSTER），识别跨境数据流
5. **提取实际 PII 清单** — 生成完整的实际 PII 列清单

## 可用工具

| 工具 | 用途 | 何时使用 |
|------|------|---------|
| `search_gdpr_knowledge` | 搜索 GDPR 法条、EDPB 指南、执法案例 | 需要确认法律依据或了解违规后果时 |
| `scan_pii_columns` | 扫描 SQL 中的 PII 列 | 第一步必调 |
| `parse_sql_lineage` | 追踪 SELECT...AS 别名 | scan 完成后调用 |
| `check_retention_ttl` | 验证 TTL 合规性 | 发现 TTL 配置后调用 |
| `detect_cross_border_risk` | 检测跨境传输 | 发现 region/CLUSTER 配置后调用 |

## 审计方法论

1. **先扫描**：调用 scan_pii_columns 识别所有 PII 列
2. **再追踪**：调用 parse_sql_lineage 检查是否有别名模糊 PII
3. **查 TTL**：如果发现 TTL 配置，调用 check_retention_ttl 验证
4. **查跨境**：如果发现 region/CLUSTER 配置，调用 detect_cross_border_risk
5. **查法条**：对每个 HIGH/MEDIUM 发现，调用 search_gdpr_knowledge 确认条款

**重要**：每个工具调用 1 次即可。完成所有工具调用后，综合分析结果并输出最终结论。

## 输出格式

所有工具调用完成后，你的最后一条消息必须是以下 JSON 格式（用 ```json 代码块包裹）：

```json
{
    "evidence": [
        {
            "source": "data_schema_auditor",
            "evidence_id": "EVD-DATA-001",
            "type": "pii_scan|lineage_tracking|retention_ttl_check|cross_border_detection|schema_metadata",
            "schema_name": "SQL文件名",
            "summary": "证据摘要（中文）"
        }
    ],
    "findings": [
        {
            "finding_id": "F-AUDITID-DATA-001",
            "source": "data_schema_auditor",
            "state": "FAIL|PASS",
            "category": "UNDECLARED_PII|RETENTION_EXCESSIVE|TRANSFER_UNDECLARED|PII_OBFUSCATION|SPECIAL_CATEGORY_DATA",
            "severity": "HIGH|MEDIUM|LOW",
            "title": "发现标题（中文，简洁）",
            "description": "详细描述（中文，3-5句，包含具体字段名和数值）",
            "related_articles": ["Art.5(1)(e)", "Art.44"]
        }
    ]
}
```

**规则**：
- 至少 3 条 evidence（pii_scan + lineage + ttl 或跨境），至少 3 条 findings
- severity 判定：设备标识/位置数据未声明 → HIGH；TTL 超 3 倍 → HIGH；别名模糊 → MEDIUM
- related_articles 必须引用具体的 GDPR 条款号
- 检测到高危 PII 类型（IMEI, GPS, health, biometric）→ 标注 SPECIAL_CATEGORY_DATA
- 所有文本用中文
"""


# ═══════════════════════════════════════════════════════════
# Agent 缓存（模块级单例）
# ═══════════════════════════════════════════════════════════

_data_agent = None


def _get_data_agent():
    """获取或创建 Data Schema Agent（惰性缓存）。"""
    global _data_agent
    if _data_agent is None:
        _data_agent = create_agent(
            model=get_llm(),
            tools=DATA_AUDITOR_TOOLS,
            system_prompt=DATA_AUDITOR_PROMPT,
            name="data_schema_auditor",
        )
    return _data_agent


# ═══════════════════════════════════════════════════════════
# 节点函数
# ═══════════════════════════════════════════════════════════

def data_schema_auditor_node(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    Data Schema Auditor 节点 — 真实 LLM ReAct 模式。

    使用 DeepSeek LLM + 5 个工具，在 ReAct 循环中自主完成数据表审计。
    如果 LLM 或工具调用失败，回退到模拟输出。

    参数:
        state: 完整审计状态

    返回:
        dict — 含 evidence 和 findings 的部分状态更新
    """
    data_schemas = state.get("data_schemas", [])
    audit_id = state.get("audit_id", "UNKNOWN")
    target_name = state.get("target_name", "UNKNOWN")

    # ── 无输入时跳过 ──
    if not data_schemas:
        return {
            "evidence": [{
                "source": "data_schema_auditor",
                "type": "no_input",
                "summary": "无数据表结构输入，跳过审计。",
            }],
            "findings": [],
        }

    try:
        # 构建任务描述
        schema_descriptions = []
        for schema in data_schemas:
            name = schema.get("name", "unknown")
            content = schema.get("content", "")
            schema_descriptions.append(
                f"### SQL/DDL 文件: {name}\n```sql\n{content[:4000]}\n```"
            )

        schema_texts = "\n\n".join(schema_descriptions)

        task = f"""请审计以下数据库 schema。

目标系统: {target_name}
审计 ID: {audit_id}

{schema_texts}

请按以下步骤执行：
1. 先调用 scan_pii_columns 扫描所有 PII 列
2. 调用 parse_sql_lineage 检测 SQL 别名
3. 如果发现 TTL 配置，调用 check_retention_ttl 验证
4. 如果发现 region/CLUSTER 配置，调用 detect_cross_border_risk
5. 对 HIGH 等级的发现，调用 search_gdpr_knowledge 确认相关 GDPR 条款
6. 输出 JSON 格式的 findings 和 evidence

注意: finding_id 格式为 F-{audit_id}-DATA-XXX (如 F-{audit_id}-DATA-001)
"""

        # 调用 ReAct Agent
        agent = _get_data_agent()
        result = agent.invoke({
            "messages": [HumanMessage(content=task)]
        })

        # 解析输出
        parsed = _parse_agent_output(result, audit_id, data_schemas)

        return parsed

    except Exception as e:
        print(f"[Data Schema Auditor] LLM 调用失败，使用模拟输出: {e}")
        return _fallback_output(data_schemas, audit_id)


# ═══════════════════════════════════════════════════════════
# 输出解析
# ═══════════════════════════════════════════════════════════

def _parse_agent_output(agent_result: dict, audit_id: str, fallback_docs: list = None) -> dict:
    """
    从 Agent 的最终消息中提取 findings 和 evidence JSON。
    兼容 markdown code block 和裸 JSON。
    """
    messages = agent_result.get("messages", [])

    for msg in reversed(messages):
        content = ""
        if hasattr(msg, "content"):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get("content", "")

        if not content or not isinstance(content, str):
            continue

        json_str = _extract_json_block(content)
        if not json_str:
            continue

        try:
            parsed = json.loads(json_str)
            findings = parsed.get("findings", [])
            evidence = parsed.get("evidence", [])

            findings = normalize_findings(findings, "data_schema_auditor")
            for f in findings:
                if "finding_id" not in f:
                    f["finding_id"] = f"F-{audit_id}-DATA-AUTO"

            evidence = ensure_evidence(findings, evidence, "data_schema_auditor")

            if findings or evidence:
                return {"findings": findings, "evidence": evidence}
        except json.JSONDecodeError:
            continue

    return _fallback_output(fallback_docs or [], audit_id)


def _extract_json_block(text: str) -> str | None:
    """从文本中提取 JSON 块。"""
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r'\{[\s\S]*"findings"[\s\S]*\}', text)
    if match:
        return match.group(0)

    return None


# ═══════════════════════════════════════════════════════════
# 回退（LLM 调用失败时）
# ═══════════════════════════════════════════════════════════

def _fallback_output(data_schemas: list, audit_id: str) -> dict:
    """
    LLM 调用失败时的模拟输出。
    """
    evidence_items = []
    findings = []

    for schema_idx, schema in enumerate(data_schemas):
        schema_name = schema.get("name", f"schema_{schema_idx}")
        schema_content = schema.get("content", "")

        evidence_items.append({
            "source": "data_schema_auditor",
            "evidence_id": f"EVD-DATA-{schema_idx + 1:03d}-FB",
            "type": "fallback_analysis",
            "schema_name": schema_name,
            "content_length": len(schema_content),
            "summary": f"[回退模式] 已加载 schema: {schema_name} ({len(schema_content)} 字)",
        })

        findings.append({
            "finding_id": f"F-{audit_id}-DATA-FB-{schema_idx + 1:03d}",
            "source": "data_schema_auditor",
            "state": "NEEDS_MANUAL_REVIEW",
            "category": "INCONCLUSIVE",
            "severity": "MEDIUM",
            "title": "[回退模式] LLM 审计不可用，需人工审核",
            "description": (
                "Data Schema Auditor 的 LLM 调用失败，无法自动分析数据表结构。"
                "请检查 API key 配置或网络连接后重试。"
            ),
            "related_articles": [],
        })

    return {"findings": findings, "evidence": evidence_items}
