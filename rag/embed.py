"""
嵌入服务封装。

将 config.py 中的嵌入模型配置封装为可注入的服务接口。
支持 LangChain 兼容的嵌入模型（OpenAI / 本地模型）。

设计原则：
  - 单例模式：全局共享一个嵌入模型实例（避免重复初始化）
  - 惰性加载：首次调用 embed() 时才初始化模型
  - 批量嵌入：支持批量调用以提升性能
"""

from typing import Optional, List
from config import get_embeddings, EMBEDDING_DIMENSION

# 单例持有
_embeddings = None


def get_embedding_model():
    """
    获取或创建嵌入模型实例（惰性加载单例）。

    返回:
        OpenAIEmbeddings 实例
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embeddings()
    return _embeddings


def embed_text(text: str) -> list[float]:
    """
    对单段文本生成嵌入向量。

    参数:
        text: 输入文本

    返回:
        嵌入向量 (长度 = EMBEDDING_DIMENSION)
    """
    model = get_embedding_model()
    return model.embed_query(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    对多段文本批量生成嵌入向量。

    参数:
        texts: 输入文本列表

    返回:
        嵌入向量列表
    """
    model = get_embedding_model()
    return model.embed_documents(texts)


def get_embedding_dimension() -> int:
    """
    获取嵌入向量的维度。

    返回:
        维度数（如 text-embedding-v3 的 1024）
    """
    return EMBEDDING_DIMENSION
