"""
RAG 搜索服务 — 混合搜索（语义 + 关键词）+ 版本元数据过滤。

这是两个 Agent 共用的核心工具 `search_gdpr_knowledge` 的后端实现。

搜索策略：
  1. 语义搜索：用嵌入向量在 ChromaDB 中找最相似的 chunk
  2. 关键词过滤：在元数据中匹配 article、topic、regulation_id 等
  3. 混合排序：语义相似度 × 关键词匹配度 → 最终排序
  4. 版本感知：返回结果带版本元数据，供 RegulationVersionTracker 使用

LangGraph 知识点：
  - 此模块被 Agent 的 ReAct 循环中的 Tool 调用
  - Agent 决定"何时搜索"和"搜索什么"，search_gdpr_knowledge 负责"怎么搜"
  - 返回的结果带版本元数据 → 报告可标注使用了哪些法规版本
"""

import re
from typing import Optional
from rag.collections import (
    get_collection,
    COLLECTION_GDPR_LEGAL_TEXT,
    COLLECTION_EDPB_GUIDELINES,
    COLLECTION_ENFORCEMENT_CASES,
    COLLECTION_PII_PATTERNS,
    COLLECTION_RETENTION_GUIDELINES,
)
from rag.embed import embed_text, get_embedding_dimension


# ═══════════════════════════════════════════════════════════
# 搜索配置
# ═══════════════════════════════════════════════════════════

# 默认返回结果数
DEFAULT_N_RESULTS = 5

# 各个 Collection 的搜索权重（用于跨 Collection 搜索）
COLLECTION_WEIGHTS = {
    COLLECTION_GDPR_LEGAL_TEXT: 1.0,        # 法规正文 — 最高权重
    COLLECTION_EDPB_GUIDELINES: 0.95,       # EDPB 指南 — 很高权重
    COLLECTION_ENFORCEMENT_CASES: 0.7,      # 执法案例 — 参考
    COLLECTION_PII_PATTERNS: 0.8,           # PII 模式 — 实用
    COLLECTION_RETENTION_GUIDELINES: 0.8,   # 保留期指南 — 实用
}


# ═══════════════════════════════════════════════════════════
# 公共接口
# ═══════════════════════════════════════════════════════════

def search_gdpr_knowledge(
    query: str,
    collections: list[str] = None,
    n_results: int = DEFAULT_N_RESULTS,
    filter_article: str = None,
    filter_topic: str = None,
    filter_regulation_id: str = None,
    include_metadata: bool = True,
) -> list[dict]:
    """
    GDPR 知识库混合搜索。

    这是两个 Agent 共用的核心 RAG 工具。被 LangChain Tool 包装后，
    在 Agent 的 ReAct 循环中由 LLM 决定何时调用、传什么参数。

    参数:
        query: 搜索查询文本（自然语言）
        collections: 要搜索的 Collection 列表（默认全部 5 个）
        n_results: 返回结果数（默认 5）
        filter_article: 按条款号过滤（如 "7" 只看 Art.7）
        filter_topic: 按主题过滤（如 "consent", "cross_border"）
        filter_regulation_id: 按法规 ID 过滤（如 "GDPR-2016-679"）
        include_metadata: 是否返回版本元数据（默认 True）

    返回:
        [
            {
                "content": "法规/指南文本...",
                "collection": "gdpr_legal_text",
                "score": 0.85,                    # 相似度分数
                "metadata": {                     # 版本元数据
                    "article": "7",
                    "version": "v1.0",
                    "effective_date": "2018-05-25",
                    ...
                }
            },
            ...
        ]

    用法:
        results = search_gdpr_knowledge(
            query="consent requirements for marketing emails",
            filter_topic="consent",
            filter_article="7",
        )
        for r in results:
            print(f"[{r['metadata'].get('article', '?')}] {r['content'][:100]}...")
    """
    # 默认搜索所有 Collection
    if collections is None:
        collections = list(COLLECTION_WEIGHTS.keys())

    # 构建元数据过滤条件
    where_filter = _build_metadata_filter(
        filter_article,
        filter_topic,
        filter_regulation_id,
    )

    all_results = []

    for collection_name in collections:
        try:
            col = get_collection(collection_name)

            # 如果 Collection 为空，跳过
            if col.count() == 0:
                continue

            # 语义搜索
            collection_results = _semantic_search(
                col, query, n_results, where_filter
            )

            # 应用权重
            weight = COLLECTION_WEIGHTS.get(collection_name, 0.5)
            for r in collection_results:
                r["score"] = r.get("score", 0.5) * weight
                r["collection"] = collection_name

            all_results.extend(collection_results)

        except Exception as e:
            # 单个 Collection 搜索失败不影响其他 Collection
            all_results.append({
                "content": f"[Search error in {collection_name}: {str(e)}]",
                "collection": collection_name,
                "score": 0.0,
                "metadata": {},
                "error": str(e),
            })

    # 按分数排序，取 top-N
    all_results.sort(key=lambda r: r.get("score", 0), reverse=True)

    # 去除 metadata（如果不需要）
    if not include_metadata:
        for r in all_results:
            r.pop("metadata", None)

    return all_results[:n_results]


def search_keywords(
    keywords: list[str],
    collection_name: str = COLLECTION_PII_PATTERNS,
    n_results: int = DEFAULT_N_RESULTS,
) -> list[dict]:
    """
    关键词精确搜索（用于 PII 模式匹配等场景）。

    不同于语义搜索——直接用关键词匹配元数据中的字段名。
    用于 Data Schema Auditor 的 PII 扫描工具。

    参数:
        keywords: 关键词列表（如 ["email", "phone", "imei"]）
        collection_name: 目标 Collection
        n_results: 返回结果数

    返回:
        匹配到的文档列表
    """
    try:
        col = get_collection(collection_name)
        if col.count() == 0:
            return []

        results = []

        for keyword in keywords:
            # 用关键词作为查询做语义搜索
            # 同时也尝试直接匹配元数据
            try:
                keyword_results = _semantic_search(col, keyword, n_results=2)
                results.extend(keyword_results)
            except Exception:
                continue

        # 去重 + 排序
        seen = set()
        unique = []
        for r in sorted(results, key=lambda x: x.get("score", 0), reverse=True):
            content_key = r["content"][:100]
            if content_key not in seen:
                seen.add(content_key)
                unique.append(r)

        return unique[:n_results]

    except Exception as e:
        return [{
            "content": f"[Keyword search error: {str(e)}]",
            "collection": collection_name,
            "score": 0.0,
            "metadata": {},
        }]


# ═══════════════════════════════════════════════════════════
# 内部辅助
# ═══════════════════════════════════════════════════════════

def _semantic_search(
    collection,
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    where_filter: dict = None,
) -> list[dict]:
    """
    在单个 Collection 中执行语义搜索。

    参数:
        collection: ChromaDB Collection
        query: 查询文本
        n_results: 返回结果数
        where_filter: ChromaDB where 过滤条件

    返回:
        搜索结果列表
    """
    try:
        # 生成查询嵌入向量
        query_embedding = embed_text(query)

        # ChromaDB 搜索
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, collection.count()),
        }

        if where_filter:
            query_kwargs["where"] = where_filter

        chroma_result = collection.query(**query_kwargs)
    except Exception:
        # 嵌入失败（如无 API key）→ 回退到关键词匹配
        return _keyword_fallback_search(collection, query, n_results, where_filter)

    # 格式化结果
    formatted = []
    if chroma_result and chroma_result["ids"] and chroma_result["ids"][0]:
        for i in range(len(chroma_result["ids"][0])):
            result = {
                "content": chroma_result["documents"][0][i]
                if chroma_result.get("documents") and chroma_result["documents"][0]
                else "",
                "metadata": chroma_result["metadatas"][0][i]
                if chroma_result.get("metadatas") and chroma_result["metadatas"][0]
                else {},
                "score": chroma_result["distances"][0][i]
                if chroma_result.get("distances") and chroma_result["distances"][0]
                else 0.5,
            }
            # ChromaDB 返回的是距离（越小越相关），转为相似度分数
            result["score"] = 1.0 / (1.0 + result["score"])
            formatted.append(result)

    return formatted


def _keyword_fallback_search(
    collection,
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    where_filter: dict = None,
) -> list[dict]:
    """
    关键词回退搜索 — 当嵌入不可用时使用。

    简单策略：在文档和元数据中匹配查询关键词。
    精度低于语义搜索，但保证无 API key 时仍可运行。
    """
    try:
        all_data = collection.get()
    except Exception:
        return []

    if not all_data or not all_data.get("ids"):
        return []

    query_lower = query.lower()
    query_keywords = set(query_lower.split())
    results = []

    for i in range(len(all_data["ids"])):
        doc = all_data["documents"][i] if all_data.get("documents") else ""
        meta = all_data["metadatas"][i] if all_data.get("metadatas") else {}

        # 检查 where 过滤
        if where_filter:
            if not _match_filter(meta, where_filter):
                continue

        # 计算关键词匹配分
        doc_lower = doc.lower()
        score = 0.0
        for kw in query_keywords:
            if kw in doc_lower:
                score += 1.0

        if score > 0:
            results.append({
                "content": doc,
                "metadata": meta,
                "score": score / max(len(query_keywords), 1),
            })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:n_results]


def _match_filter(metadata: dict, where_filter: dict) -> bool:
    """检查文档元数据是否匹配 ChromaDB where 过滤条件。"""
    if not where_filter:
        return True

    if "$and" in where_filter:
        return all(_match_filter(metadata, cond) for cond in where_filter["$and"])

    for key, condition in where_filter.items():
        if key.startswith("$"):
            continue
        if isinstance(condition, dict) and "$eq" in condition:
            if str(metadata.get(key, "")) != str(condition["$eq"]):
                return False
        elif metadata.get(key) != condition:
            return False

    return True


def _build_metadata_filter(
    filter_article: str = None,
    filter_topic: str = None,
    filter_regulation_id: str = None,
) -> Optional[dict]:
    """
    构建 ChromaDB 元数据 where 过滤条件。

    ChromaDB 的 where 语法支持：
      - {"key": "value"}      — 精确匹配
      - {"key": {"$eq": "value"}} — 等于
      - {"$and": [...]}       — 多条件与

    参数:
        filter_article: 条款号
        filter_topic: 主题
        filter_regulation_id: 法规 ID

    返回:
        ChromaDB where dict 或 None（无过滤条件时）
    """
    conditions = []

    if filter_article:
        conditions.append({"article": {"$eq": filter_article}})

    if filter_topic:
        conditions.append({"topic": {"$eq": filter_topic}})

    if filter_regulation_id:
        conditions.append({"regulation_id": {"$eq": filter_regulation_id}})

    if not conditions:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


# ═══════════════════════════════════════════════════════════
# 便捷函数：获取法规版本信息
# ═══════════════════════════════════════════════════════════

def get_knowledge_versions() -> dict:
    """
    获取当前知识库中各法规的版本汇总。

    用于报告中的"法规版本标注"章节。

    返回:
        {regulation_id: {"latest_version": str, "latest_date": str, "chunks": int}, ...}
    """
    version_info = {}

    try:
        col = get_collection(COLLECTION_GDPR_LEGAL_TEXT)
        if col.count() > 0:
            # 获取所有 chunk 的元数据
            all_data = col.get()
            if all_data and all_data.get("metadatas"):
                seen_regs = {}
                for meta in all_data["metadatas"]:
                    reg_id = meta.get("regulation_id", "unknown")
                    if reg_id not in seen_regs:
                        seen_regs[reg_id] = {
                            "latest_version": meta.get("version", "?"),
                            "latest_date": meta.get("last_amended", meta.get("effective_date", "?")),
                            "chunks": 1,
                        }
                    else:
                        seen_regs[reg_id]["chunks"] += 1
                version_info.update(seen_regs)
    except Exception:
        pass

    return version_info
