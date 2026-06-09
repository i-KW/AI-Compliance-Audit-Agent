"""
RAG 知识库入库流水线 — Phase A: 分层 Collection 方案。

┌─ 架构总览 ────────────────────────────────────────────────────────────┐
│                                                                       │
│  rag_docs/ 中的 8 份 PDF                                              │
│       │                                                               │
│       ▼                                                               │
│  PDF 解析 (PyMuPDF)                                                   │
│       │                                                               │
│  ┌────┴────────────────────────────────────────────────────────────┐  │
│  │                        分派器 (dispatcher)                       │  │
│  │  根据文件名判断文档类型 → 分配到不同的解析和分块策略               │  │
│  └────┬────────────────────────────────────────────────────────────┘  │
│       │                                                               │
│  ┌────┴──────────┐      ┌──────────────────┐                         │
│  │  GDPR 法规     │      │  EDPB 指南        │                         │
│  │  结构化分块     │      │  章节/语义分块     │                         │
│  │  (按 Article)  │      │  (按 Section)     │                         │
│  └────┬──────────┘      └──────┬───────────┘                         │
│       │                        │                                      │
│       ▼                        ▼                                      │
│  ┌────────────────────────────────────┐                              │
│  │  元数据生成                          │                              │
│  │  - article / chapter / version      │                              │
│  │  - guideline_id / topic / date      │                              │
│  └──────────────┬─────────────────────┘                              │
│                 │                                                     │
│                 ▼                                                     │
│  ┌────────────────────────────────────┐                              │
│  │  批量嵌入 (Batch Embedding)          │                              │
│  │  DashScope text-embedding-v3        │                              │
│  │  chunk_size=16 避免限频              │                              │
│  └──────────────┬─────────────────────┘                              │
│                 │                                                     │
│                 ▼                                                     │
│  ┌────────────────────────────────────┐                              │
│  │  ChromaDB 分层入库                   │                              │
│  │  gdpr_legal_text (1 份 PDF)         │                              │
│  │  edpb_guidelines (7 份 PDF)         │                              │
│  │  + 同类文档覆蓋最新版本               │                              │
│  └────────────────────────────────────┘                              │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘

面试知识点：
  1. 结构化分块 vs 语义分块的选择依据 — 有天然边界的文档按边界分，没有的按语义分
  2. 元数据设计如何支持版本感知 — article/chapter/version 让 RegulationVersionTracker 可以追踪
  3. ChromaDB 多 Collection 架构 — 按知识类型分库，搜索时加权融合
  4. 批量嵌入 vs 逐条嵌入 — 减少 API 调用次数，提升吞吐量
  5. 幂等性检查 — 重复运行不产生重复数据（collection.count() 判断）
  6. 错误隔离 — 单份 PDF 解析失败不影响其他 PDF 的入库


用法:
    # 从命令行运行（完整流程）:
    python -m rag.ingest

    # 从 Python 代码中调用:
    from rag.ingest import ingest_all_pdfs
    stats = ingest_all_pdfs()
"""

import os
import re
import sys
from typing import Optional
from datetime import datetime

# 确保项目根目录在 path 中（单独运行时需要）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import fitz  # PyMuPDF — PDF 文本提取库

from rag.collections import (
    get_collection,
    get_chroma_client,
    COLLECTION_GDPR_LEGAL_TEXT,
    COLLECTION_EDPB_GUIDELINES,
)
from rag.embed import embed_texts


# ═══════════════════════════════════════════════════════════════════════
# 1. 常量与配置
# ═══════════════════════════════════════════════════════════════════════

# PDF 源文件目录
RAG_DOCS_DIR = os.path.join(_project_root, "rag_docs")

# Chunking 参数
GDPR_CHUNK_MIN_LENGTH = 200      # 最短 Article chunk（字符数，太短说明是空壳，跳过）
EDPB_CHUNK_SIZE = 800            # EDPB 指南的分块大小（token 近似 = 字符数/4）
EDPB_CHUNK_OVERLAP = 100         # 重叠部分
EMBEDDING_BATCH_SIZE = 10        # 批量嵌入大小（DashScope 限制 ≤10）

# 文档类型标识（从文件名推断）
GDPR_KEYWORDS = ["gdpr_regulation", "GDPR_Regulation"]
EDPB_KEYWORDS = ["EDPB", "WP248", "wp248"]


# ═══════════════════════════════════════════════════════════════════════
# 2. PDF 解析
# ═══════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    用 PyMuPDF 提取 PDF 全文本。

    面试知识点：
      PyMuPDF (fitz) 是目前最快的 Python PDF 解析库。
      它是 C 语言 MuPDF 库的 Python 绑定——比 pdfminer 快 10x，比 PyPDF2 快 5x。
      适用于文本型 PDF（GDPR 法规这类数字原生 PDF）。
      对于扫描件（图片型 PDF）需要用 OCR，PyMuPDF 不适用。

    参数:
        pdf_path: PDF 文件路径

    返回:
        全部文本内容，每页用换行符分隔
    """
    doc = fitz.open(pdf_path)
    pages_text = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        pages_text.append(text)
    doc.close()
    return "\n".join(pages_text)


def detect_document_type(filename: str) -> str:
    """
    根据文件名判断文档类型，决定使用哪种分块策略。

    面试知识点：
      这是"路由"模式——不同类型走不同处理流水线。
      在 RAG 架构中，"文档分类 → 策略选择"是比"统一处理"更优的方案。
      因为不同类型有不同的最佳分块策略。

    返回:
        "gdpr" | "edpb_guideline"
    """
    if any(kw in filename for kw in GDPR_KEYWORDS):
        return "gdpr"
    elif any(kw in filename for kw in EDPB_KEYWORDS):
        return "edpb_guideline"
    else:
        return "edpb_guideline"  # 默认走 EDPB 处理


# ═══════════════════════════════════════════════════════════════════════
# 3. Chunking 策略
# ═══════════════════════════════════════════════════════════════════════

# ─── 3.1 GDPR 结构化分块 ─────────────────────────────────────────────

def extract_article_chunks(full_text: str) -> list[dict]:
    """
    GDPR 法规的结构化分块——按 Article 边界切。

    面试知识点：
      这是"结构化分块"（Structured Chunking）的典型应用。
      核心思想是：如果文档在语义上已经有天然边界（Article / Chapter / Section），
      就按这些边界切，不要用固定长度硬切。

      优势：
        1. 语义完整——每条 chunk 是一条完整的法律条款
        2. 检索精准——搜"Art.44"能精确返回 Art.44 全文
        3. 元数据丰富——可以记录 article/chapter/version

      局限性：
        1. 依赖文档的格式一致性——GDPR 的标题格式一致，可以这样做
        2. 某些 Article 很长（如 Art.4 Definitions 有 26 条子项）→ 需要 sub-chunk
        3. 不适用于没有层级结构的文档

    返回:
        [
            {
                "text": "Article 4\ndefinitions\nFor the purposes of this Regulation...",
                "article": "4",
                "title": "Definitions",
                "chapter": "Chapter I",
                "chunk_type": "article_full"
            },
            ...
        ]
    """
    chunks = []

    # ── Step 1: 定位 "Article X" 边界 ──
    # 模式: "Article" + 空格 + 数字(1-99) + 换行
    # 使用正向前瞻 (?=Article) 保持分隔符在前一个 chunk 中
    article_split_pattern = r'(?=Article\s+(\d+)\s*\n)'
    raw_segments = re.split(article_split_pattern, full_text)

    # re.split 的行为：如果 pattern 有捕获组，捕获内容会作为分隔元素出现
    # 所以 raw_segments 是 ["前缀文本", "Article 7\n", "完整文本..."]
    # 需要重新配对

    segments = []
    i = 0
    while i < len(raw_segments):
        if raw_segments[i].startswith("Article "):
            # 当前元素是 "Article X\n"
            segments.append(raw_segments[i])
        elif segments:
            # 追加到上一个 Article 后面
            segments[-1] += raw_segments[i]
        else:
            # 前缀文本（recitals / 正文之前的 preamble）
            segments.append(raw_segments[i])
        i += 1

    # ── 处理 preamble（前缀文本，通常是 recitals）──
    preamble = segments[0] if segments else ""

    # ── 遍历每个 Article ──
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        art_match = re.match(r'Article\s+(\d+)', seg)
        if not art_match:
            continue  # 跳过 preamble 和其他非 Article 部分

        article_num = art_match.group(1)
        article_title = _extract_article_title(seg)

        # ── 判断所属 Chapter ──
        # 检测 seg 中的 "CHAPTER X" 标记（可能出现在前一个 segment 的末尾）
        # 实际做法：根据 Article 编号查预定义的 Chapter 映射
        chapter = _get_chapter_for_article(int(article_num))

        # ── Chunk 长度检查：太长的 Article 需要 sub-chunk ──
        if len(seg) > 2500:
            # 用 paragraph boundaries 进一步细分
            sub_chunks = _sub_chunk_long_article(seg, article_num, article_title, chapter)
            chunks.extend(sub_chunks)
        elif len(seg) >= GDPR_CHUNK_MIN_LENGTH:
            chunks.append({
                "text": seg,
                "article": article_num,
                "title": article_title,
                "chapter": chapter,
                "chunk_type": "article_full",
            })

    # 处理无 Article 覆盖的前言部分（preamble / recitals）
    if preamble.strip() and len(preamble.strip()) > GDPR_CHUNK_MIN_LENGTH:
        chunks.insert(0, {
            "text": preamble.strip()[:3000],  # 截断过长的 preamble
            "article": "0",
            "title": "Preamble and Recitals",
            "chapter": "Preamble",
            "chunk_type": "preamble",
        })

    return chunks


def _extract_article_title(seg: str) -> str:
    """从 Article 文本中提取标题。"""
    # 格式: "Article 7\nConditions for consent\n..."
    # 标题在 "Article X\n" 之后第一个非空行
    lines = seg.split("\n")
    for line in lines:
        line = line.strip()
        if line and not re.match(r'Article\s+\d+', line):
            return line[:120]
    return ""


def _get_chapter_for_article(article_num: int) -> str:
    """根据 Article 编号返回对应的 Chapter 名称。

    面试知识点：
      这是"元数据关联"的简单形式——不是从文档中提取，而是从预定义的知识中查找。
      在 RAG 中，不是所有元数据都需要从原文提取。
      利用领域知识（GDPR 的 Chapter 结构是固定的）可以简化提取逻辑。
    """
    chapter_map = [
        (1, 4, "Chapter I (Art.1-4) — General provisions"),
        (5, 11, "Chapter II (Art.5-11) — Principles"),
        (12, 22, "Chapter III (Art.12-22) — Rights of the data subject"),
        (23, 43, "Chapter IV (Art.23-43) — Controller and processor"),
        (44, 50, "Chapter V (Art.44-50) — Transfers of personal data to third countries"),
        (51, 59, "Chapter VI (Art.51-59) — Independent supervisory authorities"),
        (60, 67, "Chapter VII (Art.60-67) — Cooperation and consistency"),
        (68, 76, "Chapter VIII (Art.68-76) — European Data Protection Board"),
        (77, 84, "Chapter IX (Art.77-84) — Remedies, liability and penalties"),
        (85, 91, "Chapter X (Art.85-91) — Specific processing situations"),
        (92, 99, "Chapter XI (Art.92-99) — Final provisions"),
    ]
    for start, end, name in chapter_map:
        if start <= article_num <= end:
            return name
    return "Unknown Chapter"


def _sub_chunk_long_article(
    seg: str,
    article_num: str,
    article_title: str,
    chapter: str,
) -> list[dict]:
    """
    对过长（>2500字符）的 Article 按段落细分。

    面试知识点：
      结构化分块的边界情况处理——"一个 Article 塞不下怎么办？"
      不是所有 chunk 都要一样大。对于超长 Article：
      1. 按段落（double newline）分块
      2. 仍然保留 Article 元数据（方便过滤）
      3. 加 sub_chunk_index 标识第几块
    """
    paragraphs = [p.strip() for p in seg.split("\n\n") if p.strip()]
    sub_chunks = []
    current_chunk = ""
    chunk_index = 0

    for para in paragraphs:
        if len(current_chunk) + len(para) > 1200 and current_chunk:
            chunk_index += 1
            sub_chunks.append({
                "text": current_chunk.strip(),
                "article": article_num,
                "title": article_title,
                "chapter": chapter,
                "chunk_type": f"article_sub_{chunk_index}",
            })
            current_chunk = para + "\n\n"
        else:
            current_chunk += para + "\n\n"

    if current_chunk.strip():
        chunk_index += 1
        sub_chunks.append({
            "text": current_chunk.strip(),
            "article": article_num,
            "title": article_title,
            "chapter": chapter,
            "chunk_type": f"article_sub_{chunk_index}",
        })

    return sub_chunks


# ─── 3.2 EDPB 指南分块 ──────────────────────────────────────────────

def extract_edpb_section_chunks(full_text: str, filename: str) -> list[dict]:
    """
    EDPB 指南的分块——按 Section 边界切，降级为固定大小分块。

    面试知识点：
      这不是纯粹的结构化分块，也不是纯粹的固定大小分块，而是"尝试结构化，
      不行再回退"的混合策略。

      EDPB 指南的结构化程度不如 GDPR：
        - 有些指南有清晰的 "1." "2." 编号 → 可以按编号切
        - 有些指南结构不规律 → 只好用 RecursiveCharacterTextSplitter

      这里的实现：
        1. 尝试检测 Section 编号模式
        2. 如果成功 → 按 Section 边界切（结构化）
        3. 如果失败 → 用 RecursiveCharacterTextSplitter 语义分块

    返回:
        [{"text": "...", "section": "1.1"}, ...]
    """
    # ── 尝试检测 section 编号模式 ──
    # EDPB 指南常见的格式：换行后 "1." "2." ... 或 "1.1" "2.3"
    section_pattern = r'(?:\n|^)\s*(\d+\.(?:\d+\.?)?)\s+'

    # 如果检测到 section 编号，尝试用编号作为边界
    section_matches = list(re.finditer(section_pattern, full_text))

    if len(section_matches) >= 6:  # 至少有 6 个 section，说明结构良好
        return _extract_by_section_boundaries(full_text, section_matches)
    else:
        # 结构不清晰 → 用递归字符分块器（RecursiveCharacterTextSplitter）
        return _extract_by_fixed_chunks(full_text)


def _extract_by_section_boundaries(full_text: str, matches: list) -> list[dict]:
    """
    按 Section 编号切分。

    面试知识点：
      "按编号切" 的优点：
        - 每个 chunk 语义内聚（一个 section 讲一个主题）
        - 编号本身就带有层级信息
      "按编号切" 的缺点：
        - 前处理（显式声明）和后处理（被遗忘）等部分没有编号 → 丢失
        - section 长度差异大（短的 100 字符，长的 4000 字符）
    """
    chunks = []

    # 提取所有匹配的位置
    boundaries = [(m.start(), m.group(1)) for m in matches]

    for i, (start_pos, section_id) in enumerate(boundaries):
        # 找到当前 section 的结束位置（下一个 section 的开始）
        if i + 1 < len(boundaries) and i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(full_text)

        section_text = full_text[start_pos:end_pos].strip()
        if len(section_text) < 50:
            continue

        # 太长（>2000）的子 section 再按段落切
        if len(section_text) > 2000:
            sub_paras = [p.strip() for p in section_text.split("\n\n") if p.strip()]
            part_idx = 0
            for para in sub_paras:
                if len(para) < 50:
                    continue
                part_idx += 1
                chunk_text = para[:2000]  # 安全截断
                chunks.append({
                    "text": chunk_text,
                    "section": f"{section_id}-part{part_idx}",
                })
        else:
            chunks.append({
                "text": section_text,
                "section": section_id,
            })

    return chunks


def _extract_by_fixed_chunks(full_text: str) -> list[dict]:
    """
    固定大小分块（回退方案）。

    面试知识点：
      RecursiveCharacterTextSplitter 是 LangChain 的默认分块器。
      它的策略：优先在段落边界切，其次是句子边界，最后是字符边界。
      比硬切（CharacterTextSplitter）更智能，但不是结构化分块。

      参数选择：
        chunk_size=800（字符，约 200 token）— 够短以保持精确性，但够长以保持上下文
        overlap=100 — 前后 chunk 重叠，避免信息在边界处丢失
        分隔符顺序: \n\n > \n > . > 空格 — 从语义粒度最大的开始尝试

    返回:
        [{"text": "...", "section": "auto_fixed"}, ...]
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=EDPB_CHUNK_SIZE,
        chunk_overlap=EDPB_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks_text = splitter.split_text(full_text)
    return [
        {"text": chunk, "section": "auto_fixed", "chunk_index": i}
        for i, chunk in enumerate(chunks_text)
        if len(chunk) >= 50  # 过滤太短的块
    ]


# ═══════════════════════════════════════════════════════════════════════
# 4. 元数据生成
# ═══════════════════════════════════════════════════════════════════════

# ─── 4.1 GDPR 元数据模板 ─────────────────────────────────────────────

GDPR_METADATA_TEMPLATE = {
    "regulation_id": "GDPR-2016-679",
    "version": "v1.0",
    "effective_date": "2018-05-25",
    "last_amended": "2018-05-25",
    "source_url": "https://eur-lex.europa.eu/eli/reg/2016/679",
    "doc_id": "gdpr_regulation_2016_679",
}


def build_gdpr_metadata(chunk: dict) -> dict:
    """
    构建 GDPR chunk 的完整元数据。

    面试知识点：
      元数据是 RAG 中和文本本身同样重要的资产。
      好的元数据设计：
        1. 支持精确过滤（"只看 Art.7 的 chunk"）
        2. 支持版本感知（effective_date 告诉 LLM 这条法规什么时候生效）
        3. 支持来源追溯（source_url 指向官方原文）
        4. 支持法规版本比较（version 字段）
    """
    return {
        **GDPR_METADATA_TEMPLATE,
        "article": chunk.get("article", ""),
        "article_title": chunk.get("title", ""),
        "chapter": chunk.get("chapter", ""),
        "chunk_type": chunk.get("chunk_type", "article_full"),
    }


# ─── 4.2 EDPB 元数据模板 ─────────────────────────────────────────────

# EDPB 指南的元数据配置（从文件名推断）
# 键：文件名包含的关键词 → 值：metadata 模板
EDPB_DOC_METADATA: list[dict] = [
    {
        "keyword": "WP248",
        "guideline_id": "WP248-rev01",
        "title": "Guidelines on Data Protection Impact Assessment (DPIA)",
        "topic": "dpia",
        "related_articles": "Art.35, Art.36",
        "date": "2017-10-04",
    },
    {
        "keyword": "04-2019",
        "guideline_id": "Guidelines-04-2019",
        "title": "Guidelines on Article 25 — Data Protection by Design and by Default",
        "topic": "data_protection_by_design",
        "related_articles": "Art.25",
        "date": "2020-10-20",
    },
    {
        "keyword": "05-2020",
        "guideline_id": "Guidelines-05-2020",
        "title": "Guidelines on consent under Regulation 2016/679",
        "topic": "consent",
        "related_articles": "Art.7, Art.8",
        "date": "2020-05-04",
    },
    {
        "keyword": "01-2021",
        "guideline_id": "Guidelines-01-2021",
        "title": "Guidelines on Examples regarding Personal Data Breach Notification (v2.0)",
        "topic": "breach_notification",
        "related_articles": "Art.33, Art.34",
        "date": "2021-12-14",
    },
    {
        "keyword": "07-2022",
        "guideline_id": "Guidelines-07-2022",
        "title": "Guidelines on certification as a tool for transfers",
        "topic": "cross_border_certification",
        "related_articles": "Art.42, Art.43, Art.46",
        "date": "2023-02-14",
    },
    {
        "keyword": "09-2022",
        "guideline_id": "Guidelines-09-2022",
        "title": "Guidelines on personal data breach notification",
        "topic": "breach_notification",
        "related_articles": "Art.33, Art.34",
        "date": "2023-04-04",
    },
    {
        "keyword": "01-2020",
        "guideline_id": "Recommendations-01-2020",
        "title": "Recommendations on measures that supplement transfer tools (Schrems II)",
        "topic": "cross_border_transfer",
        "related_articles": "Art.44, Art.46",
        "date": "2021-06-18",
    },
]


def _find_edpb_metadata(filename: str) -> dict:
    """根据文件名查找对应的 EDPB 元数据模板。"""
    for tmpl in EDPB_DOC_METADATA:
        if tmpl["keyword"] in filename:
            return tmpl
    # 兜底
    return {
        "keyword": "",
        "guideline_id": f"UNKNOWN-{filename[:20]}",
        "title": filename.replace(".pdf", "").replace("_", " "),
        "topic": "general",
        "related_articles": "",
        "date": "unknown",
    }


def build_edpb_metadata(chunk: dict, filename: str, chunk_index: int) -> dict:
    """
    构建 EDPB chunk 的完整元数据。

    元数据字段设计：
      - edpb_guideline_id: 唯一标识该指南（如 Guidelines-05-2020）
      - title: 指南全称
      - topic: 主题分类（用于 filter_topic 精确过滤）
      - related_articles: 关联的 GDPR 条款（用于跨 Collection 关联）
      - section: 该 chunk 对应的章节编号
      - doc_id: 来源文档标识
      - chunk_index: 文档内第几个 chunk（用于排序/追溯）
    """
    base = _find_edpb_metadata(filename)
    return {
        "edpb_guideline_id": base["guideline_id"],
        "title": base["title"],
        "topic": base["topic"],
        "related_articles": base["related_articles"],
        "date": base["date"],
        "section": chunk.get("section", f"chunk_{chunk_index}"),
        "doc_id": os.path.splitext(filename)[0],
        "chunk_index": chunk_index,
    }


# ═══════════════════════════════════════════════════════════════════════
# 5. ChromaDB 入库
# ═══════════════════════════════════════════════════════════════════════

def ingest_pdf(
    pdf_path: str,
    collection,
    collection_name: str,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> dict:
    """
    将单份 PDF 解析、分块、嵌入后存入 ChromaDB。

    面试知识点：
      整个入库流程按 RAG indexing pipeline 的 4 个步骤展开：
        1. Load（加载 PDF）→ extract_text_from_pdf
        2. Chunk（分块）→ extract_article_chunks / extract_edpb_section_chunks
        3. Embed（向量化）→ embed_texts
        4. Index（索引）→ collection.add

    参数:
        pdf_path: PDF 文件路径
        collection: ChromaDB Collection 实例
        collection_name: Collection 名称（用于元数据模板选择）
        batch_size: 批量嵌入大小

    返回:
        {"filename": ..., "chunks": N, "stored": N, "errors": ...}
    """
    filename = os.path.basename(pdf_path)
    doc_type = detect_document_type(filename)
    stats = {"filename": filename, "type": doc_type, "chunks": 0, "stored": 0, "errors": []}

    try:
        # ── Step 1: 提取文本 ──
        print(f"  [extract] {filename}")
        full_text = extract_text_from_pdf(pdf_path)
        if not full_text.strip():
            stats["errors"].append("Empty text extracted")
            return stats

        # ── Step 2: 分块 ──
        print(f"  [chunk] ...")
        if doc_type == "gdpr":
            chunks = extract_article_chunks(full_text)
        else:
            chunks = extract_edpb_section_chunks(full_text, filename)

        if not chunks:
            stats["errors"].append("No chunks generated")
            return stats

        stats["chunks"] = len(chunks)

        # ── Step 3: 构造 ChromaDB 输入 ──
        documents = []
        metadatas = []
        ids = []

        for idx, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            if len(text) < GDPR_CHUNK_MIN_LENGTH:
                continue

            # 生成元数据
            if collection_name == COLLECTION_GDPR_LEGAL_TEXT:
                meta = build_gdpr_metadata(chunk)
            else:
                meta = build_edpb_metadata(chunk, filename, idx)

            # 生成唯一 ID（幂等：相同内容生成相同 ID）
            chunk_id = f"{meta.get('doc_id', filename)}-{meta.get('article', meta.get('section', str(idx)))}-{idx:03d}"

            documents.append(text)
            metadatas.append(meta)
            ids.append(chunk_id)

        # ── Step 4: 批量嵌入 ──
        print(f"  [embed] {len(documents)} chunks (batch={batch_size})...")
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch_texts = documents[i:i + batch_size]
            batch_embeds = embed_texts(batch_texts)
            all_embeddings.extend(batch_embeds)
            print(f"     batch {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1} done")

        # ── Step 5: 存入 ChromaDB ──
        print(f"  [store] -> {collection_name}...")
        collection.add(
            documents=documents,
            embeddings=all_embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        stats["stored"] = len(documents)

    except Exception as e:
        import traceback
        stats["errors"].append(f"{type(e).__name__}: {str(e)}")
        traceback.print_exc()

    return stats


# ═══════════════════════════════════════════════════════════════════════
# 5.5 RAG 构建日期标记
# ═══════════════════════════════════════════════════════════════════════

def _record_rag_build_date():
    """
    记录 RAG 知识库最后构建日期。

    写入 rag/.rag_build_date 标记文件，供 versioning/tracker.py 的
    get_rag_build_date() 读取，在审计报告 footer 中展示。

    只记录日期，不判断法规版本是否过时——是否过时由 DPO 自行判断。
    """
    marker_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".rag_build_date"
    )
    today = datetime.now().strftime("%Y-%m-%d")
    with open(marker_path, "w") as f:
        f.write(today)


def ingest_all_pdfs(force: bool = False) -> dict:
    """
    入口函数：将 rag_docs/ 下所有 PDF 按分层 Collection 方案入库。

    参数:
        force: 如果为 True，重置已有数据重新入库；否则跳过已有数据的 Collection

    返回:
        {"gdpr_legal_text": {...}, "edpb_guidelines": {...}}
        — 每个 Collection 的入库统计

    面试知识点：
      这是 RAG indexing pipeline 的总控函数。
      幂等性（idempotency）设计：
        - 首次运行 → 解析 PDF + 嵌入 + 存入
        - 再次运行 → collection.count() != 0 → 跳过，不产生重复数据
        - 这样脚本可以在 CI/CD 中安全重复执行

      分层 Collection 的分派逻辑：
        - GDPR 法规 → gdpr_legal_text（1 份 PDF）
        - EDPB 指南 → edpb_guidelines（7 份 PDF）
        - 执法案例 / PII 模式 / 保留期指南 → 保持不变（这 8 份 PDF 不涉及）
    """
    pdf_files = sorted([
        f for f in os.listdir(RAG_DOCS_DIR)
        if f.endswith(".pdf") and not f.startswith("_")
    ])
    print(f"\n{'='*60}")
    print(f"  GDPR/EDPB PDF -> ChromaDB分层入库")
    print(f"  发现 {len(pdf_files)} 份PDF")
    print(f"{'='*60}\n")

    # ── 分配 PDF 到 Collection ──
    pdf_assignments: dict[str, list[str]] = {
        COLLECTION_GDPR_LEGAL_TEXT: [],
        COLLECTION_EDPB_GUIDELINES: [],
    }
    for fname in pdf_files:
        doc_type = detect_document_type(fname)
        if doc_type == "gdpr":
            pdf_assignments[COLLECTION_GDPR_LEGAL_TEXT].append(fname)
        else:
            pdf_assignments[COLLECTION_EDPB_GUIDELINES].append(fname)

    all_stats = {}

    for collection_name, assigned_files in pdf_assignments.items():
        if not assigned_files:
            continue

        # ── 获取 Collection ──
        collection = get_collection(collection_name)

        # ── 幂等检查：已有数据时跳过，除非 force ──
        existing_count = collection.count()
        if existing_count > 0 and not force:
            print(f"[SKIP] {collection_name} 已有 {existing_count} 条数据。")
            print(f"       如需重新入库: ingest_all_pdfs(force=True)")
            print()
            all_stats[collection_name] = {
                "status": "skipped",
                "existing": existing_count,
            }
            continue

        # ── 清空旧数据 ──
        if force and existing_count > 0:
            print(f"[RESET] 清空 {collection_name} ({existing_count} 条旧数据)...")
            client = get_chroma_client()
            client.delete_collection(collection_name)
            # 清除内部缓存后重新获取
            from rag.collections import clear_collection_cache
            clear_collection_cache(collection_name)
            collection = get_collection(collection_name)

        # ── 逐份 PDF 入库 ──
        collection_stats = []
        for fname in assigned_files:
            pdf_path = os.path.join(RAG_DOCS_DIR, fname)
            print(f"\n  ── {fname} ──")
            stats = ingest_pdf(pdf_path, collection, collection_name)
            collection_stats.append(stats)
            status = "+" if stats["stored"] > 0 else "!"
            print(f"  {status} {fname}: {stats['chunks']} chunks -> {stats['stored']} stored")
            if stats["errors"]:
                for err in stats["errors"]:
                    print(f"     ❌ {err}")

        # ── 汇总 ──
        total_stored = sum(s["stored"] for s in collection_stats)
        total_chunks = sum(s["chunks"] for s in collection_stats)
        all_stats[collection_name] = {
            "status": "ingested",
            "files": len(assigned_files),
            "chunks": total_chunks,
            "stored": total_stored,
            "details": collection_stats,
        }
        print(f"\n  [{collection_name}] 合计: {total_chunks} chunks → {total_stored} stored\n")

    # ── 最终统计 ──
    print(f"{'='*60}")
    print(f"  入库完成")
    print(f"{'='*60}")
    total_stored = sum(
        s.get("stored", 0) for s in all_stats.values()
        if isinstance(s, dict)
    )
    print(f"  共处理 {len(pdf_files)} 份PDF")
    print(f"  生成并存入ChromaDB: {total_stored} 条chunk")
    print(f"  Collection分布:")
    for coll_name, stats in all_stats.items():
        if isinstance(stats, dict):
            stored = stats.get("stored", stats.get("existing", 0))
            status = stats.get("status", "?")
            print(f"    - {coll_name}: {stored} chunks ({status})")
    print()

    # ── 记录 RAG 构建日期（供审计报告 footer 使用）──
    _record_rag_build_date()
    print(f"  RAG 构建日期已记录")

    return all_stats


# ═══════════════════════════════════════════════════════════════════════
# 6. CLI 入口
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    命令行用法:
        # 首次入库
        python -m rag.ingest

        # 重新入库（清空旧数据）
        python -m rag.ingest --force

    面试知识点：
      这种设计（__main__ 入口 + 函数式 API）的好处：
        - python -m rag.ingest → 命令行运行（运维友好）
        - from rag.ingest import ingest_all_pdfs → Python API（开发者友好）
        - 同时支持两种使用方式，不改代码结构
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="GDPR/EDPB PDF → ChromaDB 分层入库流水线"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新入库（清空已有数据）"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅检查 PDF 文件和分配方案，不实际入库"
    )
    args = parser.parse_args()

    if args.dry_run:
        pdf_files = sorted([
            f for f in os.listdir(RAG_DOCS_DIR)
            if f.endswith(".pdf") and not f.startswith("_")
        ])
        print(f"\nDRY RUN - 文件分配方案:\n")
        for fname in pdf_files:
            dtype = detect_document_type(fname)
            target = COLLECTION_GDPR_LEGAL_TEXT if dtype == "gdpr" else COLLECTION_EDPB_GUIDELINES
            print(f"  {fname}")
            print(f"    -> {target}")
        print(f"\n  共 {len(pdf_files)} 份PDF")
        sys.exit(0)

    ingest_all_pdfs(force=args.force)
