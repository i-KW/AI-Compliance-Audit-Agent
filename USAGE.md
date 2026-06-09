# GDPR Privacy Auditor — 使用指南

## 前置条件

1. 安装依赖：`pip install -r requirements.txt`
2. 配置 `.env` 文件（已配好 DeepSeek + 阿里云嵌入）

## 三种使用方式

### 方式 1: 只审隐私文档

```python
from graph import run_audit

result = run_audit(
    target_name="某电商平台",
    privacy_documents=[{
        "name": "隐私政策.md",
        "content": "我们收集您的电子邮件、姓名和账单地址用于订单处理。使用本服务即表示您同意接收营销邮件..."
    }],
    data_schemas=[],  # 不传 SQL
)
# → 输出 MEDIUM/LOW 风险 + Privacy Auditor 的发现
print(result["report_text"])
```

### 方式 2: 只审数据表结构

```python
from graph import run_audit

result = run_audit(
    target_name="用户数据库",
    privacy_documents=[],  # 不传隐私文档
    data_schemas=[{
        "name": "users.sql",
        "content": "CREATE TABLE users (email VARCHAR, phone VARCHAR, device_imei VARCHAR) CLUSTER=us-west-2;"
    }],
)
# → 输出 PII 扫描 + TTL + 跨境检测结果
print(result["report_text"])
```

### 方式 3: 混合审计（完整链路 — 最能展示系统能力）

```python
from graph import run_audit

result = run_audit(
    target_name="某全球电商平台",
    target_description="服务欧盟用户，数据库部署在美西和欧洲",
    privacy_documents=[{
        "name": "privacy_policy_2024.md",
        "content": "你的隐私政策完整文本...",
    }],
    data_schemas=[{
        "name": "schema.sql",
        "content": "CREATE TABLE users (...); CREATE TABLE orders (...);",
    }],
    document_date="2024-01-15",  # 隐私政策更新日期（用于版本感知）
)
# → Fan-Out 2 个 Agent → 冲突检测 → 仲裁 → HITL → DPIA → 报告

# 查看完整报告
print(result["report_text"])

# 查看具体发现
for f in result["findings"]:
    print(f"[{f['severity']}] {f['title']}")

# 查看冲突消解记录
for c in result["conflicts"]:
    arb = c.get("arbitration_result", {})
    print(f"{c['conflict_type']} → winner={arb['winner']}, method={arb['method']}")
```

### HITL 人审（DPO 模拟审批）

系统检测到 HIGH 风险时会暂停等待 DPO 决策。测试时用环境变量模拟：

```bash
# 默认自动审批
python your_script.py

# 模拟 DPO 驳回
DPO_TEST_ACTION=reject DPO_TEST_REJECT_REASON="超出GDPR管辖范围" python your_script.py

# 模拟 DPO 降级风险
DPO_TEST_ACTION=edit DPO_TEST_NEW_TIER=MEDIUM python your_script.py
```

### 运行测试

```bash
python -m pytest tests/ -v
# 27 个测试，覆盖：
#   场景 A: 仅文档输入
#   场景 B: 文档+SQL 完整链路（Fan-Out → Conflict → HITL → DPIA）
```

### 查看 ChromaDB 知识库状态

```python
from rag.collections import get_collection_stats
stats = get_collection_stats()
for name, info in stats.items():
    print(f"{name}: {info['count']} 条文档")

# 语义搜索
from rag.search import search_gdpr_knowledge
results = search_gdpr_knowledge("数据保护影响评估的要求是什么")
for r in results:
    print(f"[{r['collection']}] {r['content'][:100]}...")
```

### 运行单个工具

```python
from tools.privacy import analyze_privacy_text, check_consent_language
from tools.data import scan_pii_columns

# 检查隐私声明完整度
print(analyze_privacy_text.invoke({"text": "你的隐私政策..."}))

# 扫描 PII 列
sql = "CREATE TABLE users (email VARCHAR, device_imei VARCHAR, ttl_days INT DEFAULT 1460);"
print(scan_pii_columns.invoke({"sql_text": sql}))
```
