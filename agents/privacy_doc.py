"""
Privacy Doc Auditor — 隐私文档审计 Agent（真实 LLM ReAct 模式）。

职责范围（Art.5-22）：
  1. 隐私声明完整性检查
  2. 同意语言 GDPR 合规分析
  3. 地区适用范围审查
  4. 广告/营销数据声明分析
  5. 提取声明的数据类别（供冲突检测对比）

工具：
  - search_gdpr_knowledge: RAG 语义搜索 GDPR 法规知识库
  - analyze_privacy_text: 隐私声明完整性结构化检查
  - check_consent_language: 同意语言合规检查
  - extract_declared_categories: 提取声明数据类别清单

LangGraph 知识点：
  - create_react_agent 创建 ReAct 子图作为 Agent
  - 节点函数隐私 Agent，LLM 在 Think→Act→Observe 循环中自主调用工具
  - 输出 findings + evidence 通过 Annotated[list, operator.add] Fan-In 合并
"""

import json
import re
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from config import get_llm
from tools.privacy import PRIVACY_AUDITOR_TOOLS
from state import GDPRPrivacyAuditStateV2_2
from agents import normalize_findings, ensure_evidence


# ═══════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════

PRIVACY_AUDITOR_PROMPT = """你是一个 GDPR 隐私文档审计专家（Privacy Doc Auditor），专门分析隐私政策和数据声明文件。

## 你的职责范围（GDPR Art.5-22）

1. **隐私声明完整性检查** — 是否声明了必要的数据类别、处理目的、法律基础、接收者、跨境传输、保留期、数据主体权利
2. **同意语言分析** — 同意语言是否明确、自由、知情；是否有捆绑同意（"使用即同意"模式）
3. **地区适用范围审查** — 是否针对不同地区用户有差异化说明
4. **广告/营销数据声明分析** — 是否清晰区分核心服务目的与广告/营销目的
5. **提取声明的数据类别** — 列出政策中明确声明的所有数据类别

## 可用工具

| 工具 | 用途 | 何时使用 |
|------|------|---------|
| `search_gdpr_knowledge` | 搜索 GDPR 法条、EDPB 指南、执法案例 | 需要确认法律依据时 |
| `analyze_privacy_text` | 分析隐私声明完整度 | 第一步必调 |
| `check_consent_language` | 检查同意语言合规性 | 分析完文本后 |
| `extract_declared_categories` | 提取声明的数据类别 | 最后一步，用于对比 |

## 审计方法论

1. **先理解**：快速阅读文档，了解数据控制者是谁、处理什么数据、为什么处理
2. **再检查**：用工具逐项检查隐私声明的完整性
3. **查法条**：对发现问题，调用 search_gdpr_knowledge 确认相关 GDPR 条款
4. **最后提取**：调用 extract_declared_categories 生成数据类别清单

**重要**：每个工具只调用 1-2 次，不要重复。完成所有工具调用后，输出最终结论。

## 输出格式

所有工具调用完成后，你的最后一条消息必须是以下 JSON 格式（用 ```json 代码块包裹）：

```json
{
    "evidence": [
        {
            "source": "privacy_doc_auditor",
            "evidence_id": "EVD-PRIV-001",
            "type": "document_metadata|completeness_analysis|consent_check|category_extraction",
            "document_name": "文档名",
            "summary": "证据摘要（中文）"
        }
    ],
    "findings": [
        {
            "finding_id": "F-AUDITID-PRIV-001",
            "source": "privacy_doc_auditor",
            "state": "FAIL|PASS",
            "category": "CONSENT_LANGUAGE_VAGUE|VAGUE_PURPOSE|REGIONAL_SCOPE|INCOMPLETE_DECLARATION|MARKETING_AD_DISCLOSURE|MISSING_RIGHTS",
            "severity": "HIGH|MEDIUM|LOW",
            "title": "发现标题（中文，简洁）",
            "description": "详细描述（中文，3-5句，包含具体证据）",
            "related_articles": ["Art.7(1)", "Art.13(2)"]
        }
    ]
}
```

**规则**：
- 至少 2 条 evidence，至少 2 条 findings
- severity 判定：涉及特殊数据/跨境/同意失效 → HIGH；声明缺失但可补救 → MEDIUM；格式/措辞建议 → LOW
- related_articles 必须引用具体的 GDPR 条款号
- 所有文本用中文
"""


# ═══════════════════════════════════════════════════════════
# Agent 缓存（模块级单例）
# ═══════════════════════════════════════════════════════════

_privacy_agent = None


def _get_privacy_agent():
    """获取或创建 Privacy Doc Agent（惰性缓存）。"""
    global _privacy_agent
    if _privacy_agent is None:
        _privacy_agent = create_agent(
            model=get_llm(),
            tools=PRIVACY_AUDITOR_TOOLS,
            system_prompt=PRIVACY_AUDITOR_PROMPT,
            name="privacy_doc_auditor",
        )
    return _privacy_agent


# ═══════════════════════════════════════════════════════════
# 节点函数
# ═══════════════════════════════════════════════════════════

def privacy_doc_auditor_node(state: GDPRPrivacyAuditStateV2_2) -> dict:
    """
    Privacy Doc Auditor 节点 — 真实 LLM ReAct 模式。

    使用 DeepSeek LLM + 4 个工具，在 ReAct 循环中自主完成隐私文档审计。
    如果 LLM 或工具调用失败，回退到模拟输出（保证图不中断）。

    参数:
        state: 完整审计状态

    返回:
        dict — 含 evidence 和 findings 的部分状态更新
    """
    privacy_docs = state.get("privacy_documents", [])
    audit_id = state.get("audit_id", "UNKNOWN")
    target_name = state.get("target_name", "UNKNOWN")

    # ── 无输入时跳过 ──
    if not privacy_docs:
        return {
            "evidence": [{
                "source": "privacy_doc_auditor",
                "type": "no_input",
                "summary": "无隐私文档输入，跳过审计。",
            }],
            "findings": [],
        }

    try:
        # 构建任务描述
        docs_descriptions = []
        for doc in privacy_docs:
            name = doc.get("name", "unknown")
            content = doc.get("content", "")
            docs_descriptions.append(
                f"### 文档: {name}\n```\n{content[:4000]}\n```"
            )

        document_texts = "\n\n".join(docs_descriptions)

        task = f"""请审计以下隐私政策文档。

目标系统: {target_name}
审计 ID: {audit_id}

{document_texts}

请按以下步骤执行：
1. 先调用 analyze_privacy_text 分析完整度
2. 再调用 check_consent_language 检查同意语言
3. 对发现的问题，调用 search_gdpr_knowledge 查相关法条
4. 最后调用 extract_declared_categories 提取数据类别
5. 输出 JSON 格式的 findings 和 evidence

注意: finding_id 格式为 F-{audit_id}-PRIV-XXX (如 F-{audit_id}-PRIV-001)
"""

        # 调用 ReAct Agent
        agent = _get_privacy_agent()
        result = agent.invoke({
            "messages": [HumanMessage(content=task)]
        })

        # 解析 Agent 输出
        parsed = _parse_agent_output(result, audit_id, privacy_docs)

        return parsed

    except Exception as e:
        # 回退到模拟输出
        print(f"[Privacy Doc Auditor] LLM 调用失败，使用模拟输出: {e}")
        return _fallback_output(privacy_docs, audit_id)


# ═══════════════════════════════════════════════════════════
# 输出解析
# ═══════════════════════════════════════════════════════════

def _parse_agent_output(agent_result: dict, audit_id: str, fallback_docs: list = None) -> dict:
    """
    从 Agent 的最终消息中提取 findings 和 evidence JSON。

    容忍 JSON 被 markdown code block 包裹的情况。
    """
    messages = agent_result.get("messages", [])

    # 从最后往前找 AI 消息中的 JSON
    for msg in reversed(messages):
        content = ""
        if hasattr(msg, "content"):
            content = msg.content
        elif isinstance(msg, dict):
            content = msg.get("content", "")

        if not content or not isinstance(content, str):
            continue

        # 提取 JSON 块
        json_str = _extract_json_block(content)
        if not json_str:
            continue

        try:
            parsed = json.loads(json_str)
            findings = parsed.get("findings", [])
            evidence = parsed.get("evidence", [])

            # 补充 source 和 finding_id
            findings = normalize_findings(findings, "privacy_doc_auditor")
            for f in findings:
                if "finding_id" not in f:
                    f["finding_id"] = f"F-{audit_id}-PRIV-AUTO"

            evidence = ensure_evidence(findings, evidence, "privacy_doc_auditor")

            if findings or evidence:
                return {"findings": findings, "evidence": evidence}
        except json.JSONDecodeError:
            continue

    # 解析失败 → 回退（传入原始文档以生成 evidence）
    return _fallback_output(fallback_docs or [], audit_id)


def _extract_json_block(text: str) -> str | None:
    """从文本中提取 JSON 块（支持 markdown code block 和裸 JSON）。"""
    # 尝试 ```json ... ``` 包裹
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 尝试裸 JSON 对象
    match = re.search(r'\{[\s\S]*"findings"[\s\S]*\}', text)
    if match:
        return match.group(0)

    return None


# ═══════════════════════════════════════════════════════════
# 回退（LLM 调用失败时）
# ═══════════════════════════════════════════════════════════

def _fallback_output(privacy_docs: list, audit_id: str) -> dict:
    """
    LLM 调用失败时的模拟输出。
    功能与 Phase 2 的模拟输出等价，但标注为回退模式。
    """
    evidence_items = []
    findings = []

    for doc_idx, doc in enumerate(privacy_docs):
        doc_name = doc.get("name", f"document_{doc_idx}")
        doc_content = doc.get("content", "")

        evidence_items.append({
            "source": "privacy_doc_auditor",
            "evidence_id": f"EVD-PRIV-{doc_idx + 1:03d}-FB",
            "type": "fallback_analysis",
            "document_name": doc_name,
            "content_length": len(doc_content),
            "summary": f"[回退模式] 已加载隐私文档: {doc_name} ({len(doc_content)} 字)",
        })

        findings.append({
            "finding_id": f"F-{audit_id}-PRIV-FB-{doc_idx + 1:03d}",
            "source": "privacy_doc_auditor",
            "state": "NEEDS_MANUAL_REVIEW",
            "category": "INCONCLUSIVE",
            "severity": "MEDIUM",
            "title": "[回退模式] LLM 审计不可用，需人工审核",
            "description": (
                "Privacy Doc Auditor 的 LLM 调用失败，无法自动分析隐私文档。"
                "请检查 API key 配置或网络连接后重试。"
            ),
            "related_articles": [],
        })

    return {"findings": findings, "evidence": evidence_items}
