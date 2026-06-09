# GDPR Privacy Auditor Agent — 项目状态与已知问题

> 最后更新: 2026-06-06
> 当前阶段: Phase 4 完成 待办如下：下一步是配置 OpenAI API Key 激活 ChromaDB 语义搜索

最新情况：完成了 GDPR 隐私合规审计系统的全部 4 个阶段开发，27 个测试全部通过。下一步是配置 OpenAI API Key 激活 ChromaDB 语义搜索，或讨论后续改进方向。

---

## 当前状态总览

| 阶段 | 状态 | 关键产出 |
|------|------|---------|
| Phase 1: State + 规则引擎 + 版本追踪 | ✅ 完成 | state.py, rules/priority.py, rules/rubric.py, versioning/tracker.py |
| Phase 2: LangGraph 图骨架 + Agent 节点 | ✅ 完成 | graph.py (10 nodes), subgraphs/conflict.py, agents/ |
| Phase 3: RAG + 工具层 | ✅ 完成（工具层）/ ⚠️ ChromaDB 嵌入待激活 | 9 个 LangChain Tool, 24 条种子数据, 混合搜索框架 |
| Phase 4: HITL + 端到端测试 | ✅ 完成 | hitl/review.py, tests/test_scenario_a.py, tests/test_scenario_b.py, tests/conftest.py |

---

## 已知问题

### Issue #1: ChromaDB 嵌入模型下载极慢

**严重程度**: ✅ 已解决（2026-06-06）

**解决方案**: 使用阿里云 DashScope `text-embedding-v3` API（1024维，中文 ✅）
+ `config.py` 中 `tiktoken_enabled=False` 修复 langchain-openai 发送 token ID 的问题

**验证结果**: 24 条种子数据全部用 text-embedding-v3 向量化，中文查询正常检索英文文档（跨语言 RAG ✅）

---

### Issue #2: Phase 2 Agent 节点使用模拟输出

**严重程度**: 低（预期行为，Phase 4 中不修复）

**现象**:
- `agents/privacy_doc.py` 和 `agents/data_schema.py` 的节点函数返回硬编码的模拟 findings
- 模拟数据用于验证图结构，不反映真实 LLM 审计能力

**解决方案**（后续阶段）:
- 将 Phase 3 的工具注入 `create_agent()`，让 LLM 在 ReAct 循环中自主调用工具
- 需要 `config.py` 中配置有效的 LLM API key

**相关文件**:
- `agents/privacy_doc.py` — `privacy_doc_auditor_node()` 末尾的 "Phase 3 升级指南"
- `agents/data_schema.py` — `data_schema_auditor_node()` 末尾的 "Phase 3 升级指南"

---

### Issue #3: LangGraph 版本差异（1.2.2 vs 架构文档假设的 0.2.x）

**严重程度**: 低（已适配）

**已适配的变更**:
- `SqliteSaver` → `InMemorySaver`（LangGraph 1.x 移除了 SQLite 内置支持）
- `Send`, `interrupt`, `Command` API 保持一致
- `StateGraph` 编译 API 保持一致

**未适配的**:
- 持久化 checkpointer 需要用 LangGraph API Server 或自行实现（Phase 4 暂不需要）

---

## Phase 4 待做事项

- [ ] `hitl/review.py` — 真实 HITL 人审节点（interrupt + Command resume）
- [ ] `tests/test_scenario_a.py` — 仅文档输入端到端测试
- [ ] `tests/test_scenario_b.py` — 文档+SQL 混合输入端到端测试（完整 Fan-Out → Conflict → HITL → DPIA 链路）

---

## 环境信息

- **操作系统**: Windows 11 Home China 10.0.26200
- **Python**: 3.11.8
- **LangGraph**: 1.2.2
- **ChromaDB**: 0.5.x
- **安装的包**: langgraph, langchain, langchain-core, langchain-openai, chromadb, sqlparse, pydantic, python-dotenv, pytest
