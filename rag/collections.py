"""
ChromaDB Collection 定义与初始化。

V2.2 架构定义了 5 个 Collection，按知识类型分库：

  1. gdpr_legal_text      — GDPR 法规正文（Art.1-99）
  2. edpb_guidelines      — EDPB 官方指南（WP248 等）
  3. enforcement_cases    — 执法案例和罚款记录
  4. pii_patterns         — PII 识别模式（regex + 语义描述）
  5. retention_guidelines  — 数据保留期行业指南

每个 chunk 的元数据：
  - regulation_id:     法规 ID（如 "GDPR-2016-679"）
  - article:           条款号（如 "7"）
  - version:           版本号（如 "v1.0"）
  - effective_date:    生效日期（如 "2018-05-25"）
  - edpb_guideline_id: EDPB 指南 ID（如适用）

LangGraph 知识点：
  - RAG 是 Agent 工具链中的"知识检索"环节
  - ChromaDB 是外部依赖，通过工具函数与 LangGraph 节点解耦
  - 版本元数据支持 RegulationVersionTracker 的时效性检查
"""

import os
import chromadb
from chromadb.config import Settings
from typing import Optional

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_GDPR_LEGAL_TEXT,
    COLLECTION_EDPB_GUIDELINES,
    COLLECTION_ENFORCEMENT_CASES,
    COLLECTION_PII_PATTERNS,
    COLLECTION_RETENTION_GUIDELINES,
    EMBEDDING_DIMENSION,
)


# ═══════════════════════════════════════════════════════════
# 单例
# ═══════════════════════════════════════════════════════════

_client: Optional[chromadb.PersistentClient] = None
_collections: dict[str, chromadb.Collection] = {}


# ═══════════════════════════════════════════════════════════
# Collection 元数据定义
# ═══════════════════════════════════════════════════════════

COLLECTION_DEFINITIONS = {
    COLLECTION_GDPR_LEGAL_TEXT: {
        "name": COLLECTION_GDPR_LEGAL_TEXT,
        "description": "GDPR 法规正文 — Regulation (EU) 2016/679 全文，按条款分 chunk",
        "metadata_schema": {
            "regulation_id": "str",       # "GDPR-2016-679"
            "article": "str",             # "5", "7", "44" ...
            "article_title": "str",       # "Principles relating to processing of personal data"
            "chapter": "str",             # "Chapter II"
            "version": "str",             # "v1.0"
            "effective_date": "str",      # "2018-05-25"
            "last_amended": "str",        # "2018-05-25"
            "source_url": "str",          # EUR-Lex URL
        },
    },
    COLLECTION_EDPB_GUIDELINES: {
        "name": COLLECTION_EDPB_GUIDELINES,
        "description": "EDPB 指南 — WP248 DPIA 指南、同意指南、设计保护指南等",
        "metadata_schema": {
            "edpb_guideline_id": "str",   # "Guidelines-05-2020"
            "title": "str",               # "Guidelines on consent under Regulation 2016/679"
            "version": "str",             # "v2.1"
            "date": "str",                # "2024-05-15"
            "replaces": "str",            # "v1.0 (2020-05-04)"
            "topic": "str",               # "consent", "dpia", "transparency" ...
            "related_articles": "str",    # "Art.7, Art.8"
        },
    },
    COLLECTION_ENFORCEMENT_CASES: {
        "name": COLLECTION_ENFORCEMENT_CASES,
        "description": "执法案例 — GDPR 罚款和执法记录，用于风险评估参考",
        "metadata_schema": {
            "case_id": "str",             # "C-311/18" (Schrems II)
            "case_name": "str",           # "DPC v Facebook Ireland"
            "court": "str",               # "CJEU", "DPA" ...
            "date": "str",                # "2020-07-16"
            "fine_amount": "str",         # "€1.2 billion"
            "articles_violated": "str",   # "Art.44, Art.46"
            "relevance": "str",           # "cross-border transfer"
        },
    },
    COLLECTION_PII_PATTERNS: {
        "name": COLLECTION_PII_PATTERNS,
        "description": "PII 识别模式 — 字段名正则 + 语义描述，用于数据库列扫描",
        "metadata_schema": {
            "pii_type": "str",            # "email_address", "phone_number", "device_id" ...
            "category": "str",            # "contact", "financial", "behavioral", "sensitive"
            "regex_pattern": "str",       # r'.*email.*'
            "sensitivity": "str",         # "low", "medium", "high", "special_category"
            "gdpr_article": "str",        # "Art.9" for special categories
        },
    },
    COLLECTION_RETENTION_GUIDELINES: {
        "name": COLLECTION_RETENTION_GUIDELINES,
        "description": "保留期指南 — 各行业和各数据类别的建议最长保留期",
        "metadata_schema": {
            "data_category": "str",       # "marketing_data", "financial_records" ...
            "industry": "str",            # "ecommerce", "healthcare", "finance" ...
            "max_retention_days": "str",  # "365"
            "legal_basis": "str",         # "tax_law", "contract", "consent" ...
            "guideline_source": "str",    # "EDPB", "nationalDPA", "industry_standard"
        },
    },
}


# ═══════════════════════════════════════════════════════════
# 公共接口
# ═══════════════════════════════════════════════════════════

def get_chroma_client() -> chromadb.PersistentClient:
    """
    获取或创建 ChromaDB 持久化客户端（单例模式）。

    客户端使用持久化存储，数据保存在 CHROMA_PERSIST_DIR。
    首次调用时创建客户端，后续调用返回同一实例。

    返回:
        chromadb.PersistentClient
    """
    global _client
    if _client is None:
        # 确保持久化目录存在
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

        _client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(
                anonymized_telemetry=False,  # 禁用遥测
                allow_reset=True,            # 开发阶段允许重置
            ),
        )
    return _client


def get_collection(name: str) -> chromadb.Collection:
    """
    获取或创建指定名称的 Collection。

    如果 Collection 不存在，自动创建并配置元数据。
    使用缓存避免重复创建。

    参数:
        name: Collection 名称（如 "gdpr_legal_text"）

    返回:
        chromadb.Collection — 已初始化且可操作的集合
    """
    global _collections

    if name in _collections:
        return _collections[name]

    client = get_chroma_client()
    definition = COLLECTION_DEFINITIONS.get(name)

    try:
        # 尝试获取已有的 Collection
        collection = client.get_collection(name=name)
    except Exception:
        # Collection 不存在 → 创建
        if definition and "metadata_schema" in definition:
            # 将元数据 schema 作为 Collection 的 metadata 存储
            collection_metadata = {
                "description": definition["description"],
                **{f"schema_{k}": v for k, v in definition["metadata_schema"].items()},
            }
        else:
            collection_metadata = {"description": name}

        collection = client.create_collection(
            name=name,
            metadata=collection_metadata,
        )

    _collections[name] = collection
    return collection


def get_all_collections() -> dict[str, chromadb.Collection]:
    """
    获取所有 5 个 Collection 的字典。

    用于批量操作（如初始化时填充种子数据、清空重建等）。

    返回:
        {collection_name: Collection, ...}
    """
    return {
        name: get_collection(name)
        for name in COLLECTION_DEFINITIONS
    }


def reset_all_collections() -> None:
    """
    重置所有 Collection（清空数据，保留 Collection 结构）。

    用途：开发阶段清空测试数据，重新填充。
    ⚠️ 此操作不可逆，生产环境慎用。
    """
    global _collections
    client = get_chroma_client()

    for name in COLLECTION_DEFINITIONS:
        try:
            client.delete_collection(name=name)
        except Exception:
            pass  # Collection 可能不存在

    _collections = {}  # 清空缓存


def clear_collection_cache(name: str = None) -> None:
    """
    清除 _collections 内部缓存。

    用于在 delete_collection 后强制重新从磁盘加载（避免返回过期引用）。
    如果传入 name，只清除指定 collection 的缓存；否则清除全部。

    参数:
        name: Collection 名称（可选）
    """
    global _collections
    if name:
        _collections.pop(name, None)
    else:
        _collections = {}


def get_collection_stats() -> dict:
    """
    获取各 Collection 的统计信息。

    返回:
        {collection_name: {"count": int, "description": str}, ...}

    用于确认知识库填充状态。
    """
    stats = {}
    for name, definition in COLLECTION_DEFINITIONS.items():
        try:
            collection = get_collection(name)
            count = collection.count()
        except Exception:
            count = 0

        stats[name] = {
            "count": count,
            "description": definition["description"],
        }
    return stats


def seed_sample_data() -> dict:
    """
    填充示例种子数据到各 Collection。

    用途：让 Phase 3 的 RAG 搜索有内容可搜。
    在真实部署时，这些数据应被正式的法规知识库替代。

    每个 Collection 填充 3-7 条代表性文档，覆盖主要的 GDPR 条款和场景。

    嵌入策略：
      1. 优先使用 OpenAI embeddings（需 API key）
      2. 无 API key 时使用内置 dummy embeddings（仅用于验证结构）

    返回:
        {collection_name: count_added}
    """
    # 尝试获取嵌入向量
    embeddings_map = _compute_seed_embeddings()
    use_explicit_embeddings = embeddings_map is not None

    results = {}

    # 获取所有种子文档
    all_docs = _get_all_seed_docs()

    # ── 1. GDPR 法规正文 ──
    try:
        col = get_collection(COLLECTION_GDPR_LEGAL_TEXT)
        if col.count() == 0:
            embeds = embeddings_map.get(COLLECTION_GDPR_LEGAL_TEXT) if use_explicit_embeddings else None
            _add_seed_collection(col, all_docs[COLLECTION_GDPR_LEGAL_TEXT], embeds)
        results[COLLECTION_GDPR_LEGAL_TEXT] = col.count()
    except Exception as e:
        results[COLLECTION_GDPR_LEGAL_TEXT] = f"Error: {e}"

    # ── 2. EDPB 指南 ──
    try:
        col = get_collection(COLLECTION_EDPB_GUIDELINES)
        if col.count() == 0:
            embeds = embeddings_map.get(COLLECTION_EDPB_GUIDELINES) if use_explicit_embeddings else None
            _add_seed_collection(col, all_docs[COLLECTION_EDPB_GUIDELINES], embeds)
        results[COLLECTION_EDPB_GUIDELINES] = col.count()
    except Exception as e:
        results[COLLECTION_EDPB_GUIDELINES] = f"Error: {e}"

    # ── 3. 执法案例 ──
    try:
        col = get_collection(COLLECTION_ENFORCEMENT_CASES)
        if col.count() == 0:
            embeds = embeddings_map.get(COLLECTION_ENFORCEMENT_CASES) if use_explicit_embeddings else None
            _add_seed_collection(col, all_docs[COLLECTION_ENFORCEMENT_CASES], embeds)
        results[COLLECTION_ENFORCEMENT_CASES] = col.count()
    except Exception as e:
        results[COLLECTION_ENFORCEMENT_CASES] = f"Error: {e}"

    # ── 4. PII 模式 ──
    try:
        col = get_collection(COLLECTION_PII_PATTERNS)
        if col.count() == 0:
            embeds = embeddings_map.get(COLLECTION_PII_PATTERNS) if use_explicit_embeddings else None
            _add_seed_collection(col, all_docs[COLLECTION_PII_PATTERNS], embeds)
        results[COLLECTION_PII_PATTERNS] = col.count()
    except Exception as e:
        results[COLLECTION_PII_PATTERNS] = f"Error: {e}"

    # ── 5. 保留期指南 ──
    try:
        col = get_collection(COLLECTION_RETENTION_GUIDELINES)
        if col.count() == 0:
            embeds = embeddings_map.get(COLLECTION_RETENTION_GUIDELINES) if use_explicit_embeddings else None
            _add_seed_collection(col, all_docs[COLLECTION_RETENTION_GUIDELINES], embeds)
        results[COLLECTION_RETENTION_GUIDELINES] = col.count()
    except Exception as e:
        results[COLLECTION_RETENTION_GUIDELINES] = f"Error: {e}"

    return results


# ═══════════════════════════════════════════════════════════
# 内部辅助：种子数据嵌入预计算
# ═══════════════════════════════════════════════════════════

def _compute_seed_embeddings() -> dict | None:
    """
    预计算所有种子文档的嵌入向量。

    尝试使用 OpenAI embeddings，失败时返回 None。
    ChromaDB 在 add() 时收到显式 embeddings 参数就不会触发默认模型下载。

    返回:
        {collection_name: [embedding_vectors]} 或 None（失败时）
    """
    try:
        from rag.embed import embed_texts

        # 收集所有种子文档及其所属 collection
        all_docs_map = _get_all_seed_docs()

        embeddings_map = {}
        for coll_name, docs in all_docs_map.items():
            texts = [d["text"] for d in docs]
            try:
                embeddings = embed_texts(texts)
                embeddings_map[coll_name] = embeddings
            except Exception:
                # 单个 collection 失败不影响其他
                continue

        return embeddings_map if embeddings_map else None
    except Exception:
        return None


def _get_all_seed_docs() -> dict[str, list[dict]]:
    """
    返回所有种子文档，供 _compute_seed_embeddings 使用。

    集中管理种子数据，避免重复定义。
    """
    return {
        COLLECTION_GDPR_LEGAL_TEXT: [
            {"text": "Personal data shall be processed lawfully, fairly and in a transparent manner in relation to the data subject (lawfulness, fairness, transparency). Personal data shall be collected for specified, explicit and legitimate purposes and not further processed in a manner that is incompatible with those purposes. Personal data shall be adequate, relevant and limited to what is necessary in relation to the purposes for which they are processed (data minimisation). Personal data shall be kept in a form which permits identification of data subjects for no longer than is necessary (storage limitation). Personal data shall be processed in a manner that ensures appropriate security (integrity and confidentiality).",
             "id": "gdpr_art_5", "meta": {"regulation_id": "GDPR-2016-679", "article": "5", "article_title": "Principles relating to processing of personal data", "chapter": "Chapter II", "version": "v1.0", "effective_date": "2018-05-25", "last_amended": "2018-05-25", "source_url": "https://eur-lex.europa.eu/eli/reg/2016/679"}},
            {"text": "Where processing is based on consent, the controller shall be able to demonstrate that the data subject has consented to processing of his or her personal data. If the data subject's consent is given in the context of a written declaration which also concerns other matters, the request for consent shall be presented in a manner which is clearly distinguishable from the other matters, in an intelligible and easily accessible form, using clear and plain language. When assessing whether consent is freely given, utmost account shall be taken of whether the performance of a contract is conditional on consent to processing of personal data that is not necessary for the performance of that contract (bundled consent prohibition).",
             "id": "gdpr_art_7", "meta": {"regulation_id": "GDPR-2016-679", "article": "7", "article_title": "Conditions for consent", "chapter": "Chapter II", "version": "v1.0", "effective_date": "2018-05-25", "last_amended": "2018-05-25", "source_url": "https://eur-lex.europa.eu/eli/reg/2016/679"}},
            {"text": "Where personal data relating to a data subject are collected from the data subject, the controller shall provide the data subject with: the identity and contact details of the controller; the purposes of the processing and the legal basis; the recipients or categories of recipients of the personal data; where applicable, the fact that the controller intends to transfer personal data to a third country or international organisation and the existence or absence of an adequacy decision.",
             "id": "gdpr_art_13", "meta": {"regulation_id": "GDPR-2016-679", "article": "13", "article_title": "Information to be provided where personal data are collected from the data subject", "chapter": "Chapter III", "version": "v1.0", "effective_date": "2018-05-25", "last_amended": "2018-05-25", "source_url": "https://eur-lex.europa.eu/eli/reg/2016/679"}},
            {"text": "The controller shall implement appropriate technical and organisational measures designed to implement data-protection principles in an effective manner and to integrate the necessary safeguards into the processing. The controller shall implement appropriate measures for ensuring that, by default, only personal data which are necessary for each specific purpose are processed.",
             "id": "gdpr_art_25", "meta": {"regulation_id": "GDPR-2016-679", "article": "25", "article_title": "Data protection by design and by default", "chapter": "Chapter IV", "version": "v1.0", "effective_date": "2018-05-25", "last_amended": "2018-05-25", "source_url": "https://eur-lex.europa.eu/eli/reg/2016/679"}},
            {"text": "Each controller shall maintain a record of processing activities under its responsibility. That record shall contain: the name and contact details of the controller; the purposes of the processing; a description of the categories of data subjects and categories of personal data; the categories of recipients; where applicable, transfers of personal data to a third country; where possible, the envisaged time limits for erasure of the different categories of data; where possible, a general description of the technical and organisational security measures.",
             "id": "gdpr_art_30", "meta": {"regulation_id": "GDPR-2016-679", "article": "30", "article_title": "Records of processing activities", "chapter": "Chapter IV", "version": "v1.0", "effective_date": "2018-05-25", "last_amended": "2018-05-25", "source_url": "https://eur-lex.europa.eu/eli/reg/2016/679"}},
            {"text": "The controller and the processor shall implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk, including: pseudonymisation and encryption of personal data; the ability to ensure ongoing confidentiality, integrity, availability and resilience of processing systems; the ability to restore availability and access to personal data in a timely manner in the event of a physical or technical incident; a process for regularly testing, assessing and evaluating the effectiveness of technical and organisational measures.",
             "id": "gdpr_art_32", "meta": {"regulation_id": "GDPR-2016-679", "article": "32", "article_title": "Security of processing", "chapter": "Chapter IV", "version": "v1.0", "effective_date": "2018-05-25", "last_amended": "2018-05-25", "source_url": "https://eur-lex.europa.eu/eli/reg/2016/679"}},
            {"text": "Any transfer of personal data which are undergoing processing or are intended for processing after transfer to a third country or to an international organisation shall take place only if the controller and processor comply with the conditions laid down in this Chapter. All provisions in this Chapter shall be applied in order to ensure that the level of protection of natural persons guaranteed by this Regulation is not undermined.",
             "id": "gdpr_art_44", "meta": {"regulation_id": "GDPR-2016-679", "article": "44", "article_title": "General principle for transfers", "chapter": "Chapter V", "version": "v1.0", "effective_date": "2018-05-25", "last_amended": "2018-05-25", "source_url": "https://eur-lex.europa.eu/eli/reg/2016/679"}},
        ],
        COLLECTION_EDPB_GUIDELINES: [
            {"text": "A Data Protection Impact Assessment (DPIA) is required where processing is likely to result in a high risk to the rights and freedoms of natural persons. The DPIA shall contain: a systematic description of the envisaged processing operations and the purposes; an assessment of the necessity and proportionality of the processing; an assessment of the risks to the rights and freedoms of data subjects; the measures envisaged to address the risks, including safeguards, security measures and mechanisms to ensure the protection of personal data. A single DPIA may address a set of similar processing operations that present similar high risks.",
             "id": "edpb_wp248", "meta": {"edpb_guideline_id": "WP248", "title": "Guidelines on Data Protection Impact Assessment", "version": "v1.0", "date": "2017-10-04", "replaces": "", "topic": "dpia", "related_articles": "Art.35, Art.36"}},
            {"text": "For consent to be valid under GDPR, it must be freely given, specific, informed, and unambiguous. Consent cannot be bundled as a condition of service unless the processing is necessary for that service. 'By using this service you agree...' does not meet the GDPR standard for consent. Consent requests must be separate from other terms and conditions. Pre-ticked boxes and inactivity do not constitute valid consent.",
             "id": "edpb_consent_v2_1", "meta": {"edpb_guideline_id": "Guidelines-05-2020", "title": "Guidelines on consent under Regulation 2016/679", "version": "v2.1", "date": "2024-05-15", "replaces": "v1.0 (2020-05-04)", "topic": "consent", "related_articles": "Art.7, Art.8"}},
            {"text": "Data protection by design and by default requires that controllers implement technical and organisational measures at the earliest stages of system design. Key principles include: data minimisation by default (only process what is necessary); transparency (users should understand what data is processed and why); purpose limitation (data collected for one purpose should not be repurposed without further consent or legal basis); storage limitation (automated deletion after the retention period expires).",
             "id": "edpb_bydesign_v1", "meta": {"edpb_guideline_id": "Guidelines-04-2023", "title": "Guidelines on data protection by design and by default", "version": "v1.0", "date": "2023-12-20", "replaces": "", "topic": "data_protection_by_design", "related_articles": "Art.25"}},
            {"text": "Following the Schrems II judgment (C-311/18), controllers relying on Standard Contractual Clauses (SCCs) for data transfers to third countries must conduct a Transfer Impact Assessment (TIA) to verify that the laws of the destination country provide essentially equivalent protection. Controllers must implement supplementary measures where necessary.",
             "id": "edpb_schrems2_supp", "meta": {"edpb_guideline_id": "Recommendations-01-2020", "title": "Recommendations on supplementary measures for transfers (Schrems II)", "version": "v2.0", "date": "2021-06-18", "replaces": "v1.0 (2020-11-10)", "topic": "cross_border_transfer", "related_articles": "Art.44, Art.46"}},
        ],
        COLLECTION_ENFORCEMENT_CASES: [
            {"text": "Meta Platforms Ireland Limited fined €1.2 billion by the Irish DPA for unlawful transfer of personal data from the EU to the US. The DPA found that Meta violated Art.46(1) GDPR by continuing to transfer data under SCCs without adequate supplementary measures following Schrems II. This is the largest GDPR fine to date.",
             "id": "case_meta_2023", "meta": {"case_id": "DPC-2023-01", "case_name": "Meta Platforms Ireland — Cross-border transfers", "court": "Irish DPA", "date": "2023-05-22", "fine_amount": "€1,200,000,000", "articles_violated": "Art.46(1)", "relevance": "cross-border transfer"}},
            {"text": "TikTok fined €345 million by the Irish DPA for violations related to children's data protection. The investigation found that TikTok processed children's personal data without adequate transparency, made children's accounts public by default, and failed to provide age-appropriate privacy information — violating Art.5(1)(a), Art.12(1), Art.13, Art.24, and Art.25 GDPR.",
             "id": "case_tiktok_2023", "meta": {"case_id": "DPC-2023-02", "case_name": "TikTok — Children's data", "court": "Irish DPA", "date": "2023-09-01", "fine_amount": "€345,000,000", "articles_violated": "Art.5(1)(a), Art.12(1), Art.13, Art.24, Art.25", "relevance": "transparency, children_data, design"}},
            {"text": "Amazon Europe Core fined €746 million by the Luxembourg DPA for processing personal data for targeted advertising without valid consent. The DPA found that Amazon's consent mechanism did not meet the requirements of Art.7 GDPR because consent was not freely given.",
             "id": "case_amazon_2021", "meta": {"case_id": "CNPD-2021-01", "case_name": "Amazon Europe Core — Targeted advertising", "court": "Luxembourg DPA", "date": "2021-07-16", "fine_amount": "€746,000,000", "articles_violated": "Art.7", "relevance": "consent, advertising"}},
        ],
        COLLECTION_PII_PATTERNS: [
            {"text": "Email address: a unique identifier used for electronic communication. Common column names: email, email_address, user_email, contact_email. Regex: r'.*(email|e_mail|e-mail).*' PII type: direct identifier. GDPR relevance: Art.4(1) definition of personal data.",
             "id": "pii_email", "meta": {"pii_type": "email_address", "category": "contact", "regex_pattern": r'.*(email|e_mail|e-mail).*', "sensitivity": "medium", "gdpr_article": "Art.4(1)"}},
            {"text": "Phone number: a numeric identifier for telecommunication. Common column names: phone, phone_number, mobile, tel, telephone, contact_number. Regex: r'.*(phone|mobile|tel|cell).*' PII type: direct identifier.",
             "id": "pii_phone", "meta": {"pii_type": "phone_number", "category": "contact", "regex_pattern": r'.*(phone|mobile|tel|cell).*', "sensitivity": "medium", "gdpr_article": "Art.4(1)"}},
            {"text": "Full name: the complete name of a natural person. Common column names: name, full_name, first_name, last_name, surname, given_name. Regex: r'.*(name|surname).*' PII type: direct identifier.",
             "id": "pii_name", "meta": {"pii_type": "full_name", "category": "identity", "regex_pattern": r'.*(name|surname).*', "sensitivity": "medium", "gdpr_article": "Art.4(1)"}},
            {"text": "Device IMEI: International Mobile Equipment Identity — a unique 15-digit identifier for mobile devices. Common column names: imei, device_imei, device_id, udid. Regex: r'.*(imei|device_id|udid|device_identifier).*' PII type: sensitive device identifier.",
             "id": "pii_imei", "meta": {"pii_type": "device_imei", "category": "device_identifier", "regex_pattern": r'.*(imei|device_id|udid).*', "sensitivity": "high", "gdpr_article": "Art.9, ePrivacy Art.5(3)"}},
            {"text": "GPS location: precise geolocation data capable of tracking a person's movements. Common column names: location, gps, lat, lng, longitude, latitude, geo. Regex: r'.*(location|gps|lat|lng|longitude|latitude|geo).*' PII type: sensitive location data.",
             "id": "pii_gps", "meta": {"pii_type": "gps_location", "category": "location", "regex_pattern": r'.*(location|gps|lat|lng|geo).*', "sensitivity": "high", "gdpr_article": "Art.9"}},
            {"text": "IP address: Internet Protocol address that can identify a device on a network. CJEU ruled (C-582/14) that dynamic IP addresses are personal data under GDPR. Common column names: ip, ip_address, ipaddr, client_ip, remote_addr. Regex: r'.*(ip|ipaddr|ip_address|remote_addr|client_ip).*'",
             "id": "pii_ip", "meta": {"pii_type": "ip_address", "category": "network", "regex_pattern": r'.*(ip|ipaddr|ip_address|remote_addr|client_ip).*', "sensitivity": "medium", "gdpr_article": "Art.4(1)"}},
        ],
        COLLECTION_RETENTION_GUIDELINES: [
            {"text": "Marketing and advertising data: Personal data used for marketing purposes should be retained only as long as necessary for the specific marketing campaign and no longer than the period for which consent was given. Recommended maximum: 365 days (1 year) from last interaction. Legal basis: consent expires if data subject does not engage.",
             "id": "ret_marketing", "meta": {"data_category": "marketing_data", "industry": "ecommerce", "max_retention_days": "365", "legal_basis": "consent", "guideline_source": "EDPB"}},
            {"text": "Purchase and transaction records: Financial transaction data must be retained for tax and accounting purposes. Under most EU member state tax laws, the minimum retention period is 7 years (2555 days). After tax purposes expire, data should be anonymised or deleted.",
             "id": "ret_financial", "meta": {"data_category": "financial_records", "industry": "ecommerce", "max_retention_days": "2555", "legal_basis": "tax_law", "guideline_source": "national_tax_authority"}},
            {"text": "User account data: Account profile information should be retained for the duration of the account plus a reasonable period after account closure for legal claims (typically 30-90 days). Recommended maximum for inactive accounts: 730 days (2 years). Legal basis: contract necessity + legitimate interest.",
             "id": "ret_account", "meta": {"data_category": "user_account", "industry": "ecommerce", "max_retention_days": "730", "legal_basis": "contract", "guideline_source": "industry_standard"}},
            {"text": "Session logs and analytics: Technical logs (IP addresses, user agent strings, page views) should be retained for 30-180 days depending on purpose. Security logs: 90-180 days. Analytics: 30-90 days with anonymisation after 30 days.",
             "id": "ret_sessions", "meta": {"data_category": "session_logs", "industry": "ecommerce", "max_retention_days": "180", "legal_basis": "legitimate_interest", "guideline_source": "EDPB"}},
        ],
    }


def _add_seed_collection(
    col,
    items: list[dict],
    embeddings: list[list[float]] = None,
) -> int:
    """向 Collection 添加种子数据的辅助函数。"""
    documents = [item["text"] for item in items]
    ids = [item["id"] for item in items]
    metadatas = [item["meta"] for item in items]

    add_kwargs = {
        "documents": documents,
        "ids": ids,
        "metadatas": metadatas,
    }
    if embeddings:
        add_kwargs["embeddings"] = embeddings

    col.add(**add_kwargs)
    return col.count()
