"""
HITL（Human-in-the-Loop）人审模块。

V2.2 简化设计：
  - DPO 审批整份结论，不是逐条发现
  - 三种操作：Approve（继续）/ Edit（修改风险等级）/ Reject（驳回）
  - Edit 触发重评估（回到 Synthesis Agent）
  - Reject 标记 INCONCLUSIVE 并结束审计
"""
