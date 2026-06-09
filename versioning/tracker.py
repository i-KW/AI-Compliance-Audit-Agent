"""
RegulationVersionTracker — 法规版本感知追踪器。

设计原则（面试重点）：
  这不是技术炫技。合规审计的核心前提 = 用对法规版本。
  医生用错药典版本会误诊，审计用错法规版本会误判合规。

三大功能：
  1. 输入文档时效性检查 — 文档超过 2 年 + 法规有更新 → 置顶警告
  2. RAG 元数据版本标记 — ChromaDB 每条 chunk 带版本信息
  3. 报告版本标注 — footer 列出审计使用的法规版本
  4. 旧结论自动标记 — 知识库更新后，历史审计结论标记 NEEDS_RECHECK

参考：
  - V2.2 架构文档 Section 5
"""

import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════

@dataclass
class RegulationVersion:
    """
    一条法规的版本信息。

    示例:
        RegulationVersion(
            regulation_id="GDPR-2016-679",
            name="General Data Protection Regulation",
            version="v1.0",
            effective_date="2018-05-25",
            last_amended="2018-05-25",
            source_url="https://eur-lex.europa.eu/eli/reg/2016/679",
            edpb_guidelines=[{...}],
        )
    """
    regulation_id: str                               # 法规 ID，如 "GDPR-2016-679"
    name: str                                        # 法规名称
    version: str                                     # 版本号，如 "v1.0"
    effective_date: str                              # 生效日期 "YYYY-MM-DD"
    last_amended: str                                # 最后修订日期 "YYYY-MM-DD"
    source_url: str                                  # 官方来源 URL
    edpb_guidelines: list[dict] = field(default_factory=list)  # 关联的 EDPB 指南
    in_effect: bool = True                           # 是否仍有效


@dataclass
class DocumentCurrencyResult:
    """文档时效性检查结果。"""
    is_current: bool                                 # 是否仍是最新
    warning: Optional[str] = None                    # 警告信息
    document_date: Optional[str] = None              # 文档日期
    relevant_updates_since: list[str] = field(default_factory=list)  # 文档日期后的法规更新


# ═══════════════════════════════════════════════════════════
# RegulationVersionTracker — 核心追踪器
# ═══════════════════════════════════════════════════════════

class RegulationVersionTracker:
    """
    法规版本追踪器。

    使用方式：
      tracker = RegulationVersionTracker()

      # 1. 检查输入文档是否过时
      result = tracker.check_document_currency("2023-01-15")
      if result.warning:
          print(f"⚠️ {result.warning}")

      # 2. 获取报告 footer
      footer = tracker.get_report_footer()

      # 3. 检查知识库更新
      updates = tracker.check_kb_updates("2025-01-01")
      if updates:
          print("以下法规已更新，历史审计结论需复核：")
    """

    # ═══ 文档时效性阈值 ═══
    DOCUMENT_MAX_AGE_MONTHS = 24  # 超过 2 年的隐私政策可能过时

    # ═══ 知识库法规登记表 ═══
    # 当你向 ChromaDB 添加新法规知识时，同步更新此字典。
    REGULATIONS: dict[str, RegulationVersion] = {
        "GDPR-2016-679": RegulationVersion(
            regulation_id="GDPR-2016-679",
            name="General Data Protection Regulation",
            version="v1.0",
            effective_date="2018-05-25",
            last_amended="2018-05-25",  # GDPR 正文自生效后未修订
            source_url="https://eur-lex.europa.eu/eli/reg/2016/679",
            edpb_guidelines=[
                {
                    "id": "Guidelines-05-2020",
                    "title": "Guidelines on consent under Regulation 2016/679",
                    "version": "v2.1",
                    "date": "2024-05-15",
                    "replaces": "v1.0 (2020-05-04)",
                },
                {
                    "id": "Guidelines-04-2023",
                    "title": "Guidelines on data protection by design and by default",
                    "version": "v1.0",
                    "date": "2023-12-20",
                    "replaces": "",
                },
            ],
        ),
        "Schrems-II-C-311-18": RegulationVersion(
            regulation_id="Schrems-II-C-311-18",
            name="CJEU Judgment — Data Protection Commissioner v Facebook Ireland (Schrems II)",
            version="v1.0",
            effective_date="2020-07-16",
            last_amended="2020-07-16",
            source_url="https://curia.europa.eu/juris/liste.jsf?num=C-311/18",
            edpb_guidelines=[],
        ),
        "EU-US-DPF-2023": RegulationVersion(
            regulation_id="EU-US-DPF-2023",
            name="EU-US Data Privacy Framework",
            version="v1.0",
            effective_date="2023-07-10",
            last_amended="2023-07-10",
            source_url="https://www.dataprivacyframework.gov/",
            edpb_guidelines=[],
        ),
    }

    # ═══ 公共方法 ═══

    def check_document_currency(
        self,
        document_date: Optional[str],
        reference_date: Optional[str] = None,
    ) -> DocumentCurrencyResult:
        """
        检查输入文档（隐私政策等）是否过时。

        检查逻辑：
          1. 文档日期无法解析 → 标记为需人工确认
          2. 文档超过 24 个月 → 标记为"可能过时"
          3. 文档日期后有法规更新 → 追加"以下法规已更新"提醒

        参数:
            document_date: 文档日期字符串 "YYYY-MM-DD"（从隐私政策中提取）
            reference_date: 参考日期（默认当前日期，测试时可传入固定日期）

        返回:
            DocumentCurrencyResult — 包含 is_current、warning、relevant_updates

        示例:
            >>> tracker = RegulationVersionTracker()
            >>> result = tracker.check_document_currency("2023-01-15", "2026-06-06")
            >>> result.is_current
            False
            >>> result.warning
            '文档已超过 24 个月（实际 41 月），且自文档日期以来有以下法规更新：'
        """
        if not document_date:
            return DocumentCurrencyResult(
                is_current=True,
                warning=None,
                document_date=None,
            )

        # 解析文档日期
        try:
            doc_date = datetime.strptime(document_date, "%Y-%m-%d")
        except ValueError:
            return DocumentCurrencyResult(
                is_current=False,
                warning="文档日期格式无法识别，请人工确认文档时效性。",
                document_date=document_date,
                relevant_updates_since=["日期解析失败"],
            )

        # 参考日期（默认今天）
        if reference_date:
            try:
                ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
            except ValueError:
                ref_date = datetime.now()
        else:
            ref_date = datetime.now()

        # 计算文档年龄（月）
        age_days = (ref_date - doc_date).days
        age_months = age_days / 30.0

        # 检查自文档日期以来的法规更新
        updates_since = []
        doc_date_str = doc_date.strftime("%Y-%m-%d")

        for reg_id, reg in self.REGULATIONS.items():
            # 检查法规本身的修订
            if reg.last_amended > doc_date_str:
                updates_since.append(
                    f"{reg.name} amended on {reg.last_amended}"
                )
            # 检查关联的 EDPB 指南
            for guideline in reg.edpb_guidelines:
                if guideline["date"] > doc_date_str:
                    replaces_info = (
                        f"（替代 {guideline['replaces']}）"
                        if guideline.get("replaces")
                        else ""
                    )
                    updates_since.append(
                        f"{guideline['title']} → {guideline['version']} "
                        f"({guideline['date']}) {replaces_info}"
                    )

        # 构建结果
        if age_months > self.DOCUMENT_MAX_AGE_MONTHS and updates_since:
            return DocumentCurrencyResult(
                is_current=False,
                warning=(
                    f"文档已超过 {self.DOCUMENT_MAX_AGE_MONTHS} 个月 "
                    f"（实际 {age_months:.0f} 月），且自文档日期以来有以下法规更新："
                ),
                document_date=document_date,
                relevant_updates_since=updates_since,
            )
        elif age_months > self.DOCUMENT_MAX_AGE_MONTHS:
            return DocumentCurrencyResult(
                is_current=False,
                warning=(
                    f"文档已超过 {self.DOCUMENT_MAX_AGE_MONTHS} 个月 "
                    f"（实际 {age_months:.0f} 月），建议确认仍是最新版本。"
                ),
                document_date=document_date,
                relevant_updates_since=[],
            )
        elif updates_since:
            return DocumentCurrencyResult(
                is_current=False,
                warning="文档虽在时效内，但自文档日期以来有以下法规更新：",
                document_date=document_date,
                relevant_updates_since=updates_since,
            )

        # 皆无问题
        return DocumentCurrencyResult(
            is_current=True,
            warning=None,
            document_date=document_date,
            relevant_updates_since=[],
        )

    def get_rag_build_date(self) -> str:
        """
        读取 RAG 知识库最后构建日期。

        日期由 rag/ingest.py 的 ingest_all_pdfs() 成功入库后写入
        rag/.rag_build_date 标记文件。

        返回:
            "YYYY-MM-DD" 格式字符串，或 "未知"（文件不存在时）
        """
        marker_path = os.path.join(
            os.path.dirname(__file__), "..", "rag", ".rag_build_date"
        )
        try:
            with open(marker_path) as f:
                date_str = f.read().strip()
                # 验证日期格式
                datetime.strptime(date_str, "%Y-%m-%d")
                return date_str
        except (FileNotFoundError, ValueError):
            return "未知"

    def get_report_footer(
        self,
        reference_date: Optional[str] = None,
        rag_build_date: Optional[str] = None,
    ) -> str:
        """
        生成审计报告的法规版本标注（footer）。

        只展示版本信息，不做"过时"判断。由 DPO 或合规团队自行评估时效性。

        参数:
            reference_date: 审计日期（默认当前日期）
            rag_build_date: RAG 知识库构建日期（默认自动读取标记文件）

        返回:
            多行字符串，列出审计使用的所有法规版本

        示例:
            >>> tracker = RegulationVersionTracker()
            >>> print(tracker.get_report_footer("2026-06-10", "2026-06-09"))
            本审计截至 2026-06-10 审计，基于 2026-06-09 构建的 RAG 资料和数据
            的以下法规版本：
              • GDPR 2016/679 v1.0 (生效: 2018-05-25, 最后修订: 2018-05-25)
                - Guidelines on consent v2.1 (2024-05-15)
                - Guidelines on data protection by design v1.0 (2023-12-20)
              • Schrems II CJEU C-311/18 v1.0 (生效: 2020-07-16)
              • EU-US DPF v1.0 (生效: 2023-07-10)
        """
        if reference_date:
            try:
                date_obj = datetime.strptime(reference_date, "%Y-%m-%d")
            except ValueError:
                date_obj = datetime.now()
        else:
            date_obj = datetime.now()

        date_str = date_obj.strftime("%Y-%m-%d")

        # RAG 构建日期：参数优先，否则自动读取标记文件
        if rag_build_date is None:
            rag_build_date = self.get_rag_build_date()

        lines = [
            f"本审计截至 {date_str} 审计，",
            f"基于 {rag_build_date} 构建的 RAG 资料和数据的以下法规版本：",
            "",
        ]

        for reg_id, reg in self.REGULATIONS.items():
            lines.append(
                f"  • {reg.name} {reg.version} "
                f"(生效: {reg.effective_date}, 最后修订: {reg.last_amended})"
            )
            for gl in reg.edpb_guidelines:
                lines.append(
                    f"    - {gl['title']} {gl['version']} ({gl['date']})"
                )
            if not reg.edpb_guidelines:
                lines.append(f"    (无关联 EDPB 指南)")

        return "\n".join(lines)

    def check_kb_updates(
        self,
        last_audit_date: str,
    ) -> list[dict]:
        """
        检查自上次审计以来知识库是否有法规更新。

        用途：标记旧审计结论为 NEEDS_RECHECK。

        参数:
            last_audit_date: 上次审计日期 "YYYY-MM-DD"

        返回:
            更新列表 [{"type": "guideline_update", "title": "...", ...}, ...]

        示例:
            >>> tracker = RegulationVersionTracker()
            >>> updates = tracker.check_kb_updates("2023-01-01")
            >>> len(updates) > 0  # 自 2023 年以来有 EDPB 指南更新
            True
        """
        try:
            last_date = datetime.strptime(last_audit_date, "%Y-%m-%d")
        except ValueError:
            return []

        last_date_str = last_date.strftime("%Y-%m-%d")
        updates = []

        for reg_id, reg in self.REGULATIONS.items():
            # 检查法规本身是否修订
            if reg.last_amended > last_date_str:
                updates.append({
                    "type": "regulation_amendment",
                    "regulation_id": reg_id,
                    "name": reg.name,
                    "amended_date": reg.last_amended,
                    "action_required": "法规已修订，历史审计结论可能过时，建议复核。",
                })

            # 检查关联的 EDPB 指南是否更新
            for guideline in reg.edpb_guidelines:
                if guideline["date"] > last_date_str:
                    updates.append({
                        "type": "guideline_update",
                        "regulation_id": reg_id,
                        "guideline_id": guideline["id"],
                        "title": guideline["title"],
                        "new_version": guideline["version"],
                        "date": guideline["date"],
                        "replaces": guideline.get("replaces", ""),
                        "action_required": "EDPB 指南已更新，历史审计结论可能受影响，建议复核。",
                    })

        return updates

    def get_version_metadata(self) -> dict:
        """
        获取所有法规的版本元数据摘要。

        用于存入 GDPRPrivacyAuditStateV2_2.regulation_versions。

        返回:
            {regulation_id: {name, version, effective_date}, ...}
        """
        return {
            reg_id: {
                "name": reg.name,
                "version": reg.version,
                "effective_date": reg.effective_date,
            }
            for reg_id, reg in self.REGULATIONS.items()
        }

    def add_regulation(self, regulation: RegulationVersion) -> None:
        """
        向追踪器注册一条新法规（运行时扩展）。

        参数:
            regulation: RegulationVersion 实例
        """
        self.REGULATIONS[regulation.regulation_id] = regulation

    def add_edpb_guideline(
        self,
        regulation_id: str,
        guideline: dict,
    ) -> bool:
        """
        向指定法规添加一条 EDPB 指南。

        参数:
            regulation_id: 法规 ID
            guideline: 指南信息字典

        返回:
            True 如果成功，False 如果法规不存在
        """
        reg = self.REGULATIONS.get(regulation_id)
        if not reg:
            return False
        reg.edpb_guidelines.append(guideline)
        return True
