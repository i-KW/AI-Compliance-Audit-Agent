"""
LLM 配置模块 — 用户可自由配置 API 调用方式。

使用方式：
  1. 复制 .env.example 为 .env，填入你的 API key
  2. 或通过环境变量覆盖

支持的 LLM 提供商（任何 OpenAI 兼容 API）：
  - 通义千问 (DashScope)      ← 当前推荐：中文最强
  - DeepSeek                  ← 国产，便宜
  - OpenAI 官方
  - Ollama / vLLM 本地部署    ← 免费，离线，隐私
  - 智谱 GLM / 百度文心 等

当前方案（推荐）：
  LLM:    通义千问 qwen-plus  (DashScope API)
  嵌入:   通义千问 text-embedding-v3 (DashScope API)
  中文支持: ✅ 原生
"""

import os
from dotenv import load_dotenv

# 自动加载 .env 文件（如果存在）
load_dotenv()


# ═══════════════════════════════════════════════════════════
# LLM 配置（推理模型 — 审计 Agent / 仲裁 / DPIA 生成）
# ═══════════════════════════════════════════════════════════

# API 密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-your-api-key-here")

# API 基础 URL
# 通义千问: https://dashscope.aliyuncs.com/compatible-mode/v1
# DeepSeek: https://api.deepseek.com/v1
# Ollama:   http://localhost:11434/v1
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 主模型 — 用于审计 Agent、仲裁、DPIA 生成等核心推理任务
# 通义千问: qwen-plus（推荐）/ qwen-turbo（更快）/ qwen-max（最强）
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")

# 轻量模型 — 用于简单分类、路由判断、维度评分等低复杂度任务
LLM_MODEL_LIGHT = os.getenv("LLM_MODEL_LIGHT", "qwen-turbo")

# 生成参数
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))  # 低温度保证审计一致性
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))  # 秒


# ═══════════════════════════════════════════════════════════
# 嵌入模型配置（RAG 向量化 — 不影响文本生成）
# ═══════════════════════════════════════════════════════════

# 嵌入 API — 默认复用 LLM 的 API key 和 Base URL
# 如果嵌入和 LLM 用不同服务，可单独设置
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", OPENAI_API_KEY)
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", OPENAI_BASE_URL)

# 嵌入模型名
# 通义千问: text-embedding-v3（维度可调 1024/768/512，中文 ✅）
# Ollama:   nomic-embed-text (137MB, 英文) / bge-m3 (1.2GB, 中英 ✅)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

# 嵌入向量维度
# text-embedding-v3: 1024（默认）/ 768 / 512
# bge-m3: 1024
# nomic-embed-text: 768
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))


# ═══════════════════════════════════════════════════════════
# ChromaDB 配置
# ═══════════════════════════════════════════════════════════

CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    os.path.join(os.path.dirname(__file__), "chroma_db")
)

# 5 个 Collection 名称（对应 V2.2 架构）
COLLECTION_GDPR_LEGAL_TEXT = "gdpr_legal_text"
COLLECTION_EDPB_GUIDELINES = "edpb_guidelines"
COLLECTION_ENFORCEMENT_CASES = "enforcement_cases"
COLLECTION_PII_PATTERNS = "pii_patterns"
COLLECTION_RETENTION_GUIDELINES = "retention_guidelines"


# ═══════════════════════════════════════════════════════════
# LangGraph 配置
# ═══════════════════════════════════════════════════════════

# 状态持久化数据库
CHECKPOINT_DB_PATH = os.path.join(os.path.dirname(__file__), "audit_sessions.db")

# 最大循环轮次（防止死循环）
MAX_ITERATIONS = 3
MAX_CONFLICT_ROUNDS = 2
MAX_RETRIEVAL_ROUNDS = 2
MAX_DPIA_ITERATIONS = 3

# 并发控制
MAX_CONCURRENT_AGENTS = 2  # 2 个 Specialist Agent 并发


# ═══════════════════════════════════════════════════════════
# 工具函数：获取 LLM 实例
# ═══════════════════════════════════════════════════════════

def get_llm(temperature: float = None, max_tokens: int = None):
    """
    获取主 LLM 实例。

    参数:
        temperature: 温度参数，默认使用全局配置
        max_tokens: 最大 token 数，默认使用全局配置

    返回:
        ChatOpenAI 实例（兼容任何 OpenAI 格式 API）
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else LLM_MAX_TOKENS,
        timeout=LLM_TIMEOUT,
    )


def get_llm_light():
    """
    获取轻量 LLM 实例（用于简单任务，如维度评分、路由判断）。
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=LLM_MODEL_LIGHT,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.0,  # 简单任务用 0 温度保证一致性
        max_tokens=1024,
        timeout=LLM_TIMEOUT,
    )


def get_embeddings():
    """
    获取嵌入模型实例。

    支持独立的 API key 和 Base URL（与 LLM 可不同）。
    默认复用 LLM 配置。

    返回:
        OpenAIEmbeddings 实例（兼容任何 OpenAI 格式嵌入 API）
    """
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL,
        dimensions=EMBEDDING_DIMENSION,
        # 关键：阿里云等非 OpenAI 嵌入 API 不支持 token ID 格式输入
        # tiktoken_enabled=False → 发送原始文本而非 OpenAI token ID
        # check_embedding_ctx_length=False → 跳过 OpenAI 特有的 token 计数
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )
