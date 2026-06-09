"""
GDPRPriorityEngine — 冲突消解双层架构的 Layer 1（规则引擎）。

设计原则（面试重点）：
  1. 规则引擎硬编码 GDPR 罚款梯度权重 → 80% 常规冲突直接裁决
  2. LLM 只在同权重条款冲突时做情境推断 → 20% 复杂冲突
  3. 每次裁决记录方法 (RULE_ENGINE / LLM_CONTEXTUAL) → 可审计、可复现

权重来源：
  - 100 分: Art.83(5) — 最高全球营收 4% 罚款（特殊类别数据、DPIA 义务）
  - 85-90 分: Art.83(4) — 最高全球营收 2% 罚款（跨境传输、同意）
  - 70-80 分: Art.83(4) — 核心义务（透明度、安全、数据最小化）
  - 60-65 分: Art.83(4) — 程序性义务（记录、保留期）

参考：
  - GDPR 第 83 条（罚款的一般条件）
  - V2.2 架构文档 Section 3
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ═══════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════

class ResolutionMethod(str, Enum):
    """冲突消解方法枚举（与 state.py 中的定义一致）。"""
    RULE_ENGINE = "RULE_ENGINE"         # 规则引擎直接裁决
    LLM_CONTEXTUAL = "LLM_CONTEXTUAL"   # LLM 情境推断


@dataclass
class ArbitrationRecord:
    """
    单次仲裁的完整记录。

    每条记录可审计——面试官问"仲裁怎么做的"时，直接展示此记录。
    """
    conflict_id: str                          # 冲突 ID，如 "C001"
    method: ResolutionMethod                  # 裁决方法
    rule_applied: str                         # 应用的规则，如 "PRIORITY_ART_9_OVER_ART_30"
    winner_agent: str                         # 胜方 Agent: "privacy_doc_auditor" | "data_schema_auditor"
    explanation: str                          # LLM 生成的解释文本
    weight_privacy: int                       # Privacy Doc Auditor 引用条款的权重
    weight_data: int                          # Data Schema Auditor 引用条款的权重
    llm_context: str = ""                     # LLM 推断依据（仅 LLM_CONTEXTUAL 时填充）


# ═══════════════════════════════════════════════════════════
# GDPRPriorityEngine — 核心规则引擎
# ═══════════════════════════════════════════════════════════

class GDPRPriorityEngine:
    """
    GDPR 条款优先级规则引擎。

    核心逻辑：
      1. 根据冲突类型查到两个 Agent 各自依据的 GDPR 条款
      2. 查 ARTICLE_WEIGHTS 获取每条条款的权重分
      3. 权重不同 → RULE_ENGINE 直接裁决（高权重者优先）
      4. 权重相同 → 标记为 LLM_CONTEXTUAL，交给 LLM 做情境推断

    使用方式：
      engine = GDPRPriorityEngine()
      result = engine.resolve(conflict_dict)
      # result["method"] → "RULE_ENGINE" 或 "LLM_CONTEXTUAL"
      # result["winner"] → "privacy_doc_auditor" 或 "data_schema_auditor"
    """

    # ═══ GDPR 条款 → 权重分 ═══
    # 权重越高 = 违反后罚款越重 = 合规优先级越高
    ARTICLE_WEIGHTS = {
        # ── Tier 1 (100): 特殊类别数据 + DPIA — 违反直接触发最高罚款 ──
        "art_9_special_category": 100,          # 特殊类别个人数据处理
        "art_35_dpia_required": 95,             # 数据保护影响评估义务

        # ── Tier 2 (85-90): 跨境传输 + 同意 — Schrems II 后执法活跃度最高 ──
        "art_44_transfer_mechanism": 90,        # 跨境数据传输的一般原则
        "art_46_safeguards": 88,                # 适当保障措施
        "art_7_consent_conditions": 85,         # 同意的条件
        "art_7_4_bundled_consent": 85,          # 捆绑同意禁令

        # ── Tier 3 (70-80): 核心处理原则 — 透明度 + 安全 + 最小化 ──
        "art_5_1_a_lawfulness": 80,             # 合法性、正当性、透明性
        "art_5_1_c_data_minimization": 75,      # 数据最小化
        "art_32_security": 75,                  # 处理安全性
        "art_13_transparency": 70,              # 从数据主体收集信息时的透明度
        "art_25_data_protection_by_design": 70, # 数据保护设计及默认

        # ── Tier 4 (60-65): 程序性义务 ──
        "art_30_records_of_processing": 60,     # 处理活动记录
        "art_5_1_e_storage_limitation": 65,     # 存储限制

        # ── Default ──
        "default": 50,                          # 未明确映射的条款
    }

    # ═══ 冲突类型 → 双方引用条款的映射 ═══
    # 为什么每个冲突类型只映射了 2 个条款？
    #   因为 V2.2 只有 2 个 Agent（按输入介质分），每个 Agent 在特定冲突中
    #   引用一个"主要相关"的条款。这是简化设计——实际审计中可能涉及多个条款，
    #   但规则引擎只需要"最相关"的那条来做优先级比较。
    CONFLICT_TO_ARTICLES = {
        # 声明 vs 实际数据范围不一致
        "DATA_SCOPE_DISCREPANCY": {
            "privacy_doc_auditor": "art_13_transparency",         # 政策声明的 → 透明度
            "data_schema_auditor": "art_5_1_c_data_minimization", # 实际采集的 → 最小化
        },
        # 声明保留期 vs 实际 TTL 不匹配
        "RETENTION_MISMATCH": {
            "privacy_doc_auditor": "art_13_transparency",            # 政策声明的
            "data_schema_auditor": "art_5_1_e_storage_limitation",   # 实际 TTL 的
        },
        # 实际跨境传输但政策未声明
        "TRANSFER_UNDECLARED": {
            "privacy_doc_auditor": "art_13_transparency",         # 政策未声明
            "data_schema_auditor": "art_44_transfer_mechanism",   # 实际跨境传输
        },
        # 同意范围与实际使用不匹配
        "CONSENT_SCOPE_GAP": {
            "privacy_doc_auditor": "art_7_consent_conditions",    # 同意语言
            "data_schema_auditor": "art_7_4_bundled_consent",     # 捆绑同意
        },
    }

    # ═══ 公共方法 ═══

    def get_article_weight(self, article_key: str) -> int:
        """
        查询某个条款的优先级权重。

        参数:
            article_key: 条款键名，如 "art_9_special_category"

        返回:
            权重分 (50 ~ 100)
        """
        return self.ARTICLE_WEIGHTS.get(
            article_key,
            self.ARTICLE_WEIGHTS["default"]
        )

    def resolve(self, conflict: dict) -> dict:
        """
        解决一个冲突。

        参数:
            conflict: 冲突字典，必须包含:
                - conflict_type: ConflictType 的字符串值
                - description: 冲突描述文本

        返回:
            {
                "method": "RULE_ENGINE" | "LLM_CONTEXTUAL",
                "winner": "privacy_doc_auditor" | "data_schema_auditor" | None,
                "weight_privacy": int,
                "weight_data": int,
                "rule_applied": str,
                "needs_llm": bool,          # 是否需要 LLM 参与
                "llm_prompt": str | None,   # LLM 提示词
            }

        示例:
            >>> engine = GDPRPriorityEngine()
            >>> result = engine.resolve({
            ...     "conflict_type": "DATA_SCOPE_DISCREPANCY",
            ...     "description": "政策声明收集6类数据，但DDL有12个PII字段",
            ...     "description_privacy": "隐私政策仅声明email和name",
            ...     "description_data": "实际表结构包含email/name/phone/IMEI/location/IP等",
            ... })
            >>> result["method"]
            'RULE_ENGINE'
            >>> result["winner"]
            'data_schema_auditor'  # 数据最小化(75分) > 透明度(70分)
        """
        conflict_type = conflict.get("conflict_type", "")
        mapping = self.CONFLICT_TO_ARTICLES.get(conflict_type, {})

        # 查询双方 Agent 引用的条款
        article_privacy = mapping.get("privacy_doc_auditor", "default")
        article_data = mapping.get("data_schema_auditor", "default")

        # 查权重分
        weight_privacy = self.get_article_weight(article_privacy)
        weight_data = self.get_article_weight(article_data)

        # ── 情况 1: 权重不同 → 规则引擎直接裁决 ──
        if weight_privacy != weight_data:
            winner = (
                "privacy_doc_auditor" if weight_privacy > weight_data
                else "data_schema_auditor"
            )
            rule = (
                f"PRIORITY_{article_privacy.upper()}_OVER_{article_data.upper()}"
                if weight_privacy > weight_data
                else f"PRIORITY_{article_data.upper()}_OVER_{article_privacy.upper()}"
            )
            return {
                "method": ResolutionMethod.RULE_ENGINE.value,
                "winner": winner,
                "weight_privacy": weight_privacy,
                "weight_data": weight_data,
                "rule_applied": rule,
                "needs_llm": True,  # 仍然需要 LLM 生成解释文本
                "llm_prompt": self._build_explanation_prompt(
                    conflict, winner,
                    article_privacy if winner == "privacy_doc_auditor" else article_data,
                    article_data if winner == "privacy_doc_auditor" else article_privacy,
                    max(weight_privacy, weight_data),
                    min(weight_privacy, weight_data),
                ),
            }

        # ── 情况 2: 同权重 → LLM 情境推断 ──
        return {
            "method": ResolutionMethod.LLM_CONTEXTUAL.value,
            "winner": None,  # LLM 决定
            "weight_privacy": weight_privacy,
            "weight_data": weight_data,
            "rule_applied": "SAME_WEIGHT_CONTEXTUAL",
            "needs_llm": True,
            "llm_prompt": self._build_contextual_prompt(
                conflict, article_privacy, article_data, weight_privacy
            ),
        }

    def resolve_batch(self, conflicts: list[dict]) -> list[dict]:
        """
        批量解决冲突。

        参数:
            conflicts: 冲突字典列表

        返回:
            每个冲突的裁决结果列表（与输入顺序一致）
        """
        return [self.resolve(c) for c in conflicts]

    # ═══ 私有方法 ═══

    def _build_explanation_prompt(
        self,
        conflict: dict,
        winner: str,
        article_high: str,
        article_low: str,
        weight_high: int,
        weight_low: int,
    ) -> str:
        """
        构建"解释型"提示词（规则引擎已裁决，LLM 只负责解释）。

        关键设计：此提示词明确要求 LLM "解释"而非"重新裁决"——
        防止 LLM 推翻规则引擎的决定。
        """
        return f"""你是一个 GDPR 合规解释器。你的任务是**解释**已经做出的裁决，而不是重新裁决。

## 裁决结果
{winner} 的结论优先采纳。

## 裁决依据
- {article_high.upper()}（权重 {weight_high} 分）> {article_low.upper()}（权重 {weight_low} 分）

## 权重来源
GDPR 第 83 条罚款梯度：
- 违反 {article_high.upper()} 属于较高级别义务
- 违反 {article_low.upper()} 属于较低级别的程序性/补充性义务

## 冲突内容
{conflict.get('description', '无描述')}

## 要求
请生成一段 3-5 句的解释，说明为什么这个裁决在 GDPR 框架下是正确的。
引用具体法条。**不要质疑裁决，只做解释。**"""

    def _build_contextual_prompt(
        self,
        conflict: dict,
        article_a: str,
        article_b: str,
        weight: int,
    ) -> str:
        """
        构建"推断型"提示词（同权重，LLM 做情境推断）。

        关键设计：要求 LLM 明确输出 WINNER + REASONING，
        以便程序解析结果。
        """
        return f"""你是一个 GDPR 合规仲裁员。两个条款具有相同的优先级权重（各 {weight} 分），需要你做情境推断。

## 条款信息
- 条款 A ({article_a}): 由 Privacy Doc Auditor 引用（分析隐私政策文本）
- 条款 B ({article_b}): 由 Data Schema Auditor 引用（分析数据库表结构）

## 冲突内容
{conflict.get('description', '无描述')}

## 两个 Agent 的具体陈述

### Privacy Doc Auditor 的观点：
{conflict.get('description_privacy', '未提供')}

### Data Schema Auditor 的观点：
{conflict.get('description_data', '未提供')}

## 要求
请判断哪一个 Agent 的结论应该优先采纳。你必须：
1. 明确说出 WINNER（必须是 "privacy_doc_auditor" 或 "data_schema_auditor"）
2. 引用 GDPR 的具体条文说明为什么
3. 说明为什么另一个 Agent 的视角在此冲突中是次要的

## 输出格式
WINNER: <agent_name>
REASONING: <3-5 句推理>"""


# ═══════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════

def parse_llm_winner(llm_response: str) -> Optional[str]:
    """
    从 LLM 响应中解析 WINNER 字段。

    参数:
        llm_response: LLM 的原始文本响应

    返回:
        "privacy_doc_auditor" | "data_schema_auditor" | None（解析失败时）
    """
    for line in llm_response.strip().split("\n"):
        if line.upper().startswith("WINNER:"):
            winner = line.split(":", 1)[1].strip().lower()
            if "privacy_doc_auditor" in winner:
                return "privacy_doc_auditor"
            elif "data_schema_auditor" in winner:
                return "data_schema_auditor"
    return None
