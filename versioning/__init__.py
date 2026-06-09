"""
法规版本感知模块。

包含：
  - RegulationVersionTracker: 法规版本追踪器
    - 检查输入文档时效性（超过 2 年 → 警告）
    - 生成报告法规版本标注（footer）
    - 检查知识库更新（旧审计结论自动标记 NEEDS_RECHECK）
"""
