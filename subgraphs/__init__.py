"""
子图模块。

包含：
  - Conflict Resolution 子图: 冲突检测 → 仲裁 → 验证 → 重试/升级
    - 集成 GDPRPriorityEngine 规则引擎 + LLM 双层架构
    - 内含 2 条循环回边（仲裁重试、证据补充）
"""
