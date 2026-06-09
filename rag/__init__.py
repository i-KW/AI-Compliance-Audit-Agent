"""
RAG（检索增强生成）模块。

基于 ChromaDB 的向量数据库，包含 5 个 Collection：
  - gdpr_legal_text: GDPR 法规正文
  - edpb_guidelines: EDPB 指南
  - enforcement_cases: 执法案例
  - pii_patterns: PII 识别模式
  - retention_guidelines: 保留期指南

每个 chunk 带版本元数据（regulation_id, article, version, effective_date）。
检索方式：语义搜索 + 关键词搜索混合，可选的 Cross-Encoder Reranker。
"""
