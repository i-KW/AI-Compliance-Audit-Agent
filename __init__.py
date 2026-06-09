"""
GDPR Privacy Auditor Agent — 基于 LangGraph 的 GDPR 隐私合规审计系统。

输入隐私声明文档和/或数据表结构（SQL/元数据），
自动分析 GDPR 合规风险并生成审计报告。

架构版本: V2.2 MVP
核心设计:
  - 2 个 Specialist Agent (Privacy Doc / Data Schema) Fan-Out 并发审计
  - 4 条循环回边（冲突消解重试、证据补充、DPO edit 重评估、DPIA Reflection）
  - 冲突消解 = GDPRPriorityEngine 规则引擎 + LLM 双层架构
  - DPIA 质量 = EDPB WP248 7 维度结构化量表 + 风险识别一票否决
  - 法规版本感知全链路追踪
  - HITL = 整份结论级 DPO 审批
"""

__version__ = "0.1.0"
__architecture_version__ = "V2.2-MVP"
