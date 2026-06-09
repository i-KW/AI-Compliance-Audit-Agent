"""
GDPR Privacy Auditor — 本地 Web 界面。

启动方式:
    python web_server.py
    然后浏览器打开 http://127.0.0.1:5000

功能:
    1. 拖拽/选择上传隐私文档和 SQL 文件
    2. 一键启动 GDPR 合规审计
    3. 网页展示完整审计结果（风险等级、发现清单、冲突消解、DPIA、报告）

技术栈: Flask + 原生 HTML/CSS/JS（零外部前端依赖）
"""

import os
import sys
import json
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import run_audit
from state import RiskTier

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 最大 16MB 上传
app.config['TEMPLATES_AUTO_RELOAD'] = True  # 强制 Jinja2 每次从磁盘读取模板，不缓存


@app.after_request
def add_no_cache_headers(response):
    """禁止浏览器缓存 HTML/JSON，确保每次加载最新版本。"""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# 上传文件暂存目录
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 审计任务状态（内存中）
_audit_status = {
    "running": False,
    "progress": 0,
    "message": "",
    "result": None,
    "error": None,
}


# ═══════════════════════════════════════════════════════════
# 页面路由
# ═══════════════════════════════════════════════════════════

@app.route("/")
def index():
    """主页 — 审计界面。"""
    return render_template("index.html")


# ═══════════════════════════════════════════════════════════
# API 路由
# ═══════════════════════════════════════════════════════════

@app.route("/api/audit", methods=["POST"])
def api_audit():
    """
    启动 GDPR 审计任务。

    接受:
        - privacy_doc: 隐私文档文件（可选）
        - data_schema: 数据表结构文件（可选）
        - target_name: 审计目标名称
        - target_description: 审计目标描述
        - document_date: 文档日期

    返回:
        {"status": "started", "message": "..."}
    """
    global _audit_status

    if _audit_status["running"]:
        return jsonify({"status": "error", "message": "已有审计任务在运行，请等待完成。"}), 409

    # 读取上传文件
    privacy_docs = []
    data_schemas = []

    privacy_file = request.files.get("privacy_doc")
    if privacy_file and privacy_file.filename:
        content = _read_file(privacy_file)
        if content:
            privacy_docs.append({
                "name": privacy_file.filename,
                "content": content,
            })

    schema_file = request.files.get("data_schema")
    if schema_file and schema_file.filename:
        content = _read_file(schema_file)
        if content:
            data_schemas.append({
                "name": schema_file.filename,
                "content": content,
            })

    # 至少需要一个输入
    if not privacy_docs and not data_schemas:
        return jsonify({
            "status": "error",
            "message": "请至少上传一个隐私文档或数据表结构文件。",
        }), 400

    # 获取表单参数
    target_name = request.form.get("target_name", "").strip() or "未命名审计目标"
    target_description = request.form.get("target_description", "").strip()
    document_date = request.form.get("document_date", "").strip()

    # 重置状态
    _audit_status = {
        "running": True,
        "progress": 0,
        "message": "正在初始化审计...",
        "result": None,
        "error": None,
    }

    # 在后台线程运行审计
    thread = threading.Thread(
        target=_run_audit_task,
        args=(target_name, target_description, privacy_docs, data_schemas, document_date),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "status": "started",
        "message": f"审计已启动。输入: {len(privacy_docs)} 个文档, {len(data_schemas)} 个 schema",
        "input_summary": {
            "privacy_docs": [d["name"] for d in privacy_docs],
            "data_schemas": [d["name"] for d in data_schemas],
        },
    })


@app.route("/api/status")
def api_status():
    """
    查询审计进度。

    返回:
        {
            "running": bool,
            "progress": 0-100,
            "message": "当前步骤描述",
            "result": {...} | null  (审计完成时有值),
            "error": str | null
        }
    """
    return jsonify({
        "running": _audit_status["running"],
        "progress": _audit_status["progress"],
        "message": _audit_status["message"],
        "has_result": _audit_status["result"] is not None,
        "has_error": _audit_status["error"] is not None,
    })


@app.route("/api/result")
def api_result():
    """
    获取审计结果（审计完成后调用）。

    返回完整的审计结果 JSON。
    """
    if _audit_status["running"]:
        return jsonify({"status": "pending", "message": "审计仍在进行中..."})

    if _audit_status["error"]:
        return jsonify({"status": "error", "message": _audit_status["error"]})

    if _audit_status["result"] is None:
        return jsonify({"status": "none", "message": "尚未执行审计。"})

    return jsonify({
        "status": "completed",
        "result": _audit_status["result"],
    })


# ═══════════════════════════════════════════════════════════
# 后台审计任务
# ═══════════════════════════════════════════════════════════

def _run_audit_task(target_name, target_description, privacy_docs, data_schemas, document_date):
    """
    在后台线程中运行审计，更新全局状态。
    """
    global _audit_status

    try:
        _audit_status["progress"] = 10
        _audit_status["message"] = "正在收集证据（Agent 分析中）..."

        result = run_audit(
            target_name=target_name,
            target_description=target_description,
            privacy_documents=privacy_docs,
            data_schemas=data_schemas,
            document_date=document_date,
        )

        _audit_status["progress"] = 90
        _audit_status["message"] = "正在生成报告..."

        # 提取前端需要的字段（避免序列化不可 JSON 化的对象）
        serializable = _serialize_result(result)

        _audit_status["progress"] = 100
        _audit_status["message"] = "审计完成。"
        _audit_status["result"] = serializable
        _audit_status["running"] = False

    except Exception as e:
        import traceback
        _audit_status["error"] = f"{type(e).__name__}: {str(e)}"
        _audit_status["message"] = f"审计失败: {str(e)}"
        _audit_status["running"] = False
        traceback.print_exc()


def _serialize_result(result: dict) -> dict:
    """
    将审计结果转为 JSON-safe 字典。

    处理:
        - Enum 值 → 字符串
        - 内部标记字段 (以 _ 开头) → 不输出
        - 过长的文本 → 截断
    """
    output = {}

    # 核心字段
    output["audit_id"] = result.get("audit_id", "")
    output["target_name"] = result.get("target_name", "")
    output["phase"] = result.get("phase", "")
    output["risk_tier"] = result.get("risk_tier", "")
    output["confidence_score"] = result.get("confidence_score", 0.0)

    # 发现（去重 by finding_id）
    raw_findings = result.get("findings", [])
    seen = set()
    findings = []
    for f in raw_findings:
        fid = f.get("finding_id", "")
        if fid and fid not in seen:
            seen.add(fid)
            findings.append(f)
        elif not fid:
            findings.append(f)
    output["findings"] = [
        {
            "finding_id": f.get("finding_id", "?"),
            "source": f.get("source", ""),
            "state": f.get("state", ""),
            "category": f.get("category", ""),
            "severity": f.get("severity", ""),
            "title": f.get("title", ""),
            "description": f.get("description", "")[:500],
            "related_articles": f.get("related_articles", []),
            "verification_passed": f.get("_verification_passed", True),
            "verification_issues": f.get("_verification_issues", []),
        }
        for f in findings
    ]
    output["total_findings"] = len(findings)
    output["critical_findings_count"] = min(
        result.get("critical_findings_count", 0),
        len(findings)  # 安全兜底：严重发现数不可能超过总发现数
    )
    output["has_special_category_data"] = result.get("has_special_category_data", False)
    output["cross_border_risk_level"] = result.get("cross_border_risk_level", "LOW")

    # 按来源分组（用于前端分组展示）
    privacy_files = [d.get("name", "") for d in result.get("privacy_documents", [])]
    schema_files = [d.get("name", "") for d in result.get("data_schemas", [])]
    by_source = {}
    for f in output["findings"]:
        src = f.get("source", "unknown")
        if src not in by_source:
            by_source[src] = {"files": [], "findings": []}
        by_source[src]["findings"].append(f)
    if "privacy_doc_auditor" in by_source:
        by_source["privacy_doc_auditor"]["files"] = privacy_files
    if "data_schema_auditor" in by_source:
        by_source["data_schema_auditor"]["files"] = schema_files
    output["findings_by_source"] = by_source

    # 冲突
    conflicts = result.get("conflicts", [])
    output["conflicts"] = [
        {
            "conflict_id": c.get("conflict_id", ""),
            "conflict_type": c.get("conflict_type", ""),
            "resolved": c.get("resolved", False),
            "arbitration": {
                "method": c.get("arbitration_result", {}).get("method", ""),
                "winner": c.get("arbitration_result", {}).get("winner", ""),
                "explanation": c.get("arbitration_result", {}).get("explanation", "")[:300],
            } if c.get("arbitration_result") else None,
        }
        for c in conflicts
    ]
    output["total_conflicts"] = len(conflicts)

    # DPIA
    output["dpia_quality_score"] = result.get("dpia_quality_score", 0.0)
    output["dpia_dimensions_passed"] = result.get("dpia_dimensions_passed", "?/7")
    dpia_details = result.get("dpia_quality_details", {})
    output["dpia_details"] = {
        key: {
            "name": val.get("name", key),
            "score": val.get("score", 0.0),
            "weight": val.get("weight", 0.0),
        }
        for key, val in dpia_details.items()
    }

    # DPO
    dpo = result.get("dpo_decision", {})
    output["dpo_decision"] = {
        "action": dpo.get("action", ""),
        "timestamp": dpo.get("timestamp", ""),
    } if dpo else None

    # 报告
    report = result.get("report_text", "")
    output["report_text"] = report[:10000]  # 前端最多显示 10000 字
    output["report_truncated"] = len(report) > 10000

    # 版本/时效性
    output["documents_outdated"] = len(result.get("documents_outdated", [])) > 0
    output["kb_has_updates"] = result.get("kb_has_updates", False)
    output["regulation_versions"] = result.get("regulation_versions", {})

    # 引用验证（防幻觉）
    verification = result.get("_citation_verification", {})
    output["citation_verification"] = {
        "passed": verification.get("passed", 0),
        "failed": verification.get("failed", 0),
        "total": verification.get("total", 0),
        "has_issues": verification.get("has_issues", False),
        "issues": verification.get("issues", [])[:10],  # 最多显示 10 条
    }

    return output


def _read_file(file_storage) -> str | None:
    """
    安全读取上传文件内容。

    支持 UTF-8 和 GBK 编码。
    """
    try:
        raw = file_storage.read()
        # 尝试 UTF-8
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            # 尝试 GBK（Windows 中文常见）
            try:
                return raw.decode("gbk")
            except UnicodeDecodeError:
                return raw.decode("latin-1")
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════
# 启动
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import webbrowser
    import threading

    print("=" * 50)
    print("  GDPR Privacy Auditor — Web 界面")
    print("  浏览器即将打开: http://127.0.0.1:5000")
    print("  按 Ctrl+C 停止服务器")
    print("=" * 50)

    # 延迟 1.5 秒自动打开浏览器（等 Flask 启动完成）
    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:5000")

    threading.Thread(target=open_browser, daemon=True).start()

    app.run(host="127.0.0.1", port=5000, debug=False)
