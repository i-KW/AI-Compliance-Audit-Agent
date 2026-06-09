"""
工具模块。

每个 Specialist Agent 有 4-5 个专属工具：

Privacy Doc Auditor 工具：
  - search_gdpr_knowledge: RAG 语义搜索
  - analyze_privacy_text: 隐私声明完整性检查
  - check_consent_language: 同意语言分析
  - extract_declared_categories: 提取声明的数据类别

Data Schema Auditor 工具：
  - search_gdpr_knowledge: RAG 语义搜索
  - scan_pii_columns: PII 字段扫描 (regex + 语义)
  - parse_sql_lineage: 数据血缘追踪 (SELECT...AS)
  - check_retention_ttl: 保留期 TTL 验证
  - detect_cross_border_risk: 跨境传输检测
"""
