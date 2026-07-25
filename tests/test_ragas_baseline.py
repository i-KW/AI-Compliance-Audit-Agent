"""
GDPR Privacy Auditor — RAGAS 基线评估脚本

用途：
  对当前 RAG 管道（ChromaDB 5-Collection + text-embedding-v3 混合搜索）
  跑 RAGAS 基线指标，输出评测报告。

RAGAS 指标说明：
  - context_precision: 检索到的文档中多少是真正相关的（越高越好）
  - context_recall:    所有相关文档中有多少被检索到了（越高越好）
  - faithfulness:      LLM 回答是否忠实于检索到的上下文（越高越好）
  - answer_relevancy:  LLM 回答与问题的相关程度（越高越好）

用法：
  python tests/test_ragas_baseline.py
  (或从项目根目录直接 python -m tests.test_ragas_baseline)

输出：
  - 控制台打印指标报告 + 每条 query 的明细
  - outputs/ragas_baseline_report.txt — 完整报告
  - outputs/ragas_baseline_report.json — JSON 格式（便于对比）
"""

import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# 确保项目根在 import path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Windows GBK 修复
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# RAG 管道
from rag.search import search_gdpr_knowledge
from rag.collections import get_collection_stats

# RAGAS
import ragas as ragas_lib
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)


# ═══════════════════════════════════════════════════════════
# 第 1 步: 测试集定义
# ═══════════════════════════════════════════════════════════

TEST_QUERIES = [
    {
        "question": "What are the requirements for obtaining valid consent under GDPR?",
        "ground_truth": (
            "Consent must be freely given, specific, informed and unambiguous. "
            "Controllers must demonstrate that consent was obtained. "
            "Consent requests must be clearly distinguishable from other matters. "
            "Bundled consent is prohibited. "
            "Pre-ticked boxes and inactivity do not constitute valid consent."
        ),
        "expected_articles": ["7", "8"],
        "expected_topics": ["consent"],
    },
    {
        "question": "When is a Data Protection Impact Assessment (DPIA) required?",
        "ground_truth": (
            "A DPIA is required where processing is likely to result in a high risk to "
            "the rights and freedoms of natural persons. The DPIA must contain a systematic "
            "description of processing operations and purposes, assessment of necessity and "
            "proportionality, risk assessment, and measures to address risks."
        ),
        "expected_articles": ["35"],
        "expected_topics": ["dpia"],
    },
    {
        "question": "What are the rules for international data transfers under GDPR?",
        "ground_truth": (
            "Transfers to third countries are permitted only if the controller complies "
            "with Chapter V conditions. Following Schrems II (C-311/18), controllers using "
            "SCCs must conduct a Transfer Impact Assessment to verify essentially equivalent "
            "protection in the destination country."
        ),
        "expected_articles": ["44", "46"],
        "expected_topics": ["cross_border_transfer"],
    },
    {
        "question": "What are the data protection principles under GDPR Article 5?",
        "ground_truth": (
            "Personal data shall be: (1) processed lawfully, fairly and transparently; "
            "(2) collected for specified, explicit and legitimate purposes; "
            "(3) adequate, relevant and limited to what is necessary (data minimisation); "
            "(4) accurate and kept up to date; "
            "(5) kept in identifiable form for no longer than necessary (storage limitation); "
            "(6) processed with appropriate security (integrity and confidentiality)."
        ),
        "expected_articles": ["5"],
        "expected_topics": [],
    },
    {
        "question": "What security measures are required for processing personal data under Article 32?",
        "ground_truth": (
            "Controllers and processors must implement appropriate technical and organisational measures "
            "including: pseudonymisation and encryption; ability to ensure ongoing confidentiality, "
            "integrity, availability and resilience; ability to restore access in a timely manner; "
            "and a process for regularly testing and evaluating the effectiveness of security measures."
        ),
        "expected_articles": ["32"],
        "expected_topics": [],
    },
    {
        "question": "What is considered special category data under GDPR?",
        "ground_truth": (
            "Special categories of personal data include data revealing racial or ethnic origin, "
            "political opinions, religious or philosophical beliefs, trade union membership, "
            "genetic data, biometric data for identification, health data, or data concerning "
            "a person's sex life or sexual orientation."
        ),
        "expected_articles": ["9"],
        "expected_topics": ["special_category"],
    },
    {
        "question": "What records of processing activities must a controller maintain?",
        "ground_truth": (
            "Each controller must maintain records of processing activities including: "
            "name and contact details of the controller; purposes of processing; "
            "categories of data subjects and personal data; categories of recipients; "
            "transfers to third countries; time limits for erasure; "
            "and a description of technical and organisational security measures."
        ),
        "expected_articles": ["30"],
        "expected_topics": [],
    },
    {
        "question": "What is data protection by design and by default?",
        "ground_truth": (
            "Controllers must implement measures to integrate data protection safeguards "
            "into processing activities at the earliest design stages. By default, only "
            "personal data necessary for each specific purpose should be processed. "
            "Key principles: data minimisation by default, transparency, purpose limitation, "
            "and storage limitation with automated deletion after retention periods."
        ),
        "expected_articles": ["25"],
        "expected_topics": ["data_protection_by_design"],
    },
]


# ═══════════════════════════════════════════════════════════
# 第 2 步: 运行检索
# ═══════════════════════════════════════════════════════════

def run_retrieval(query: str, n_results: int = 5) -> list[str]:
    results = search_gdpr_knowledge(
        query=query,
        n_results=n_results,
        include_metadata=True,
    )
    return [r["content"] for r in results]


# ═══════════════════════════════════════════════════════════
# 第 3 步: 用 LLM 生成回答
# ═══════════════════════════════════════════════════════════

def generate_answer(question: str, contexts: list[str]) -> str:
    try:
        from config import get_llm
        llm = get_llm(temperature=0.1)
        context_str = "\n\n".join(contexts)
        prompt = (
            "You are a GDPR compliance expert. Answer the following question based ONLY "
            "on the provided context. If the context does not contain sufficient information, "
            "say so clearly. Do NOT use external knowledge.\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"QUESTION: {question}\n\n"
            "ANSWER:"
        )
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"  [LLM answer skipped: {e}]")
        return f"[LLM unavailable] {contexts[0][:200] if contexts else 'No context'}"


# ═══════════════════════════════════════════════════════════
# 第 4 步: 收集评估数据
# ═══════════════════════════════════════════════════════════

def collect_eval_data(test_queries):
    questions, answers, contexts_list, ground_truths = [], [], [], []
    bar = "=" * 60
    print("\n" + bar)
    print("  RAGAS 基线评估 — 数据收集")
    print(bar)
    for i, tq in enumerate(test_queries):
        q = tq["question"]
        print(f"\n  [{i+1}/{len(test_queries)}] {q[:80]}...")
        ctxs = run_retrieval(q)
        contexts_list.append(ctxs)
        print(f"      检索到 {len(ctxs)} 个 chunks")
        ans = generate_answer(q, ctxs)
        answers.append(ans)
        questions.append(q)
        ground_truths.append(tq["ground_truth"])
        time.sleep(0.5)
    return questions, answers, contexts_list, ground_truths


# ═══════════════════════════════════════════════════════════
# 第 5 步: 检索质量辅助分析
# ═══════════════════════════════════════════════════════════

def analyze_retrieval_quality(test_queries, contexts_list):
    sep = "-" * 60
    print("\n" + sep)
    print("  检索质量辅助分析")
    print(sep)
    empty_count = 0
    article_hits = 0
    total_expected = 0
    for i, (tq, ctxs) in enumerate(zip(test_queries, contexts_list)):
        if len(ctxs) == 0:
            empty_count += 1
            print(f"  [i+1] 空检索: {tq['question'][:60]}...")
        all_text = " ".join(ctxs).lower()
        for art in tq.get("expected_articles", []):
            total_expected += 1
            if any(p in all_text for p in
                   [f"article {art}", f"art. {art}", f"art_{art}", f" art{art} "]):
                article_hits += 1
            else:
                print(f"  [i+1] 期望条款 Art.{art} 未命中: {tq['question'][:60]}...")
    n = len(test_queries)
    print(f"\n  空检索率:          {empty_count}/{n} ({100*empty_count/n:.0f}%)")
    print(f"  条款命中率:         {article_hits}/{total_expected} ({100*article_hits/total_expected:.0f}%)")
    avg = sum(len(c) for c in contexts_list) / n
    print(f"  平均检索 chunk 数: {avg:.1f}")
    return {
        "empty_rate": empty_count / n,
        "article_hit_rate": article_hits / max(total_expected, 1),
        "avg_chunks": avg,
    }


# ═══════════════════════════════════════════════════════════
# 第 6 步: 主流程
# ═══════════════════════════════════════════════════════════

def main():
    bar = "=" * 60
    print(bar)
    print("  GDPR Privacy Auditor — RAGAS 基线评估")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  RAGAS 版本: {ragas_lib.__version__}")
    print(f"  嵌入模型: {os.getenv('EMBEDDING_MODEL', 'text-embedding-v3')}")
    print(f"  LLM 模型: {os.getenv('LLM_MODEL', 'deepseek-chat')}")
    print(bar)

    # ChromaDB 统计
    print("\nChromaDB 统计:")
    stats = get_collection_stats()
    for name, info in stats.items():
        print(f"  {name}: {info['count']} chunks")
    total_chunks = sum(s["count"] for s in stats.values())
    print(f"  总计: {total_chunks} chunks")
    if total_chunks == 0:
        print("\n知识库为空，请先运行 seed_sample_data()")
        sys.exit(1)

    # 阶段 1: 数据收集
    print("\n" + bar)
    print("  第 1 阶段: 数据收集")
    print(bar)
    questions, answers, contexts_list, ground_truths = collect_eval_data(TEST_QUERIES)

    # 阶段 2: 检索质量分析
    print("\n" + bar)
    print("  第 2 阶段: 检索质量分析")
    print(bar)
    aux_metrics = analyze_retrieval_quality(TEST_QUERIES, contexts_list)

    # 阶段 3: RAGAS 指标
    print("\n" + bar)
    print("  第 3 阶段: RAGAS 指标计算")
    print(bar)

    scores = {}  # 兜底用的 scores dict

    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from ragas.llms.base import LangchainLLMWrapper

        # 配置 RAGAS LLM（用 LangchainLLMWrapper 避免 instructor 兼容问题）
        chat = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            api_key=os.getenv("OPENAI_API_KEY", "sk-d26d88981c4443e2a1c68e5b0b745946"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
            temperature=0.1,
            max_tokens=1024,
            timeout=120,
        )
        ragas_llm = LangchainLLMWrapper(langchain_llm=chat, bypass_n=True)

        # 配置 RAGAS 嵌入（answer_relevancy 需要）
        from langchain_openai import OpenAIEmbeddings
        ragas_emb = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
            api_key=os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY")),
            base_url=os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            dimensions=int(os.getenv("EMBEDDING_DIMENSION", "1024")),
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
        )

        eval_data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(eval_data)

        metrics_map = {
            "context_precision": context_precision,
            "context_recall": context_recall,
        }

        llm_available = not answers[0].startswith("[LLM unavailable")
        if llm_available:
            faithfulness.llm = ragas_llm
            answer_relevancy.llm = ragas_llm
            answer_relevancy.embeddings = ragas_emb
            metrics_map["faithfulness"] = faithfulness
            metrics_map["answer_relevancy"] = answer_relevancy
            print("  LLM 可用 -> 评估所有 4 个指标")
        else:
            print("  LLM 不可用 -> 仅评估检索指标")

        result = ragas_lib.evaluate(
            dataset=dataset,
            metrics=list(metrics_map.values()),
            llm=ragas_llm,
            embeddings=ragas_emb,
        )

        print(f"\n  {'='*50}")
        print(f"  RAGAS 基线报告")
        print(f"  {'='*50}")

        # 从 dataframe 提取分数（RAGAS 0.3.x 的 Result 对象不自动计算均值）
        df = result.to_pandas()
        for metric_name in metrics_map.keys():
            if metric_name in df.columns:
                vals = df[metric_name].dropna()
                if len(vals) > 0:
                    scores[metric_name] = float(vals.mean())
                    print(f"  {metric_name:25s}: {scores[metric_name]:.4f}")
                else:
                    print(f"  {metric_name:25s}: NaN (no valid scores)")
            else:
                print(f"  {metric_name:25s}: N/A (column missing)")

        print(f"  {'='*50}")

        # 逐条明细
        print(f"\n  {'='*50}")
        print(f"  逐条评估明细")
        print(f"  {'='*50}")
        for i in range(len(questions)):
            print(f"\n  [{i+1}] {questions[i][:80]}")
            for mn in metrics_map.keys():
                if mn in df.columns:
                    val = df.iloc[i][mn]
                    if val is not None and not (isinstance(val, float) and val != val):
                        print(f"      {mn}: {val:.4f}")
                    else:
                        print(f"      {mn}: NaN")
            print(f"      chunks: {len(contexts_list[i])}")

    except Exception as e:
        print(f"\n  RAGAS evaluate 出错: {e}")
        import traceback
        traceback.print_exc()

    # ═══ 构建报告 ═══
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ragas_version": ragas_lib.__version__,
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
        "llm_model": os.getenv("LLM_MODEL", "deepseek-chat"),
        "chromadb_stats": {k: v["count"] for k, v in stats.items()},
        "auxiliary": aux_metrics,
        "ragas_scores": scores,
    }

    os.makedirs("outputs", exist_ok=True)

    # TXT 报告
    txt_path = "outputs/ragas_baseline_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(bar + "\n")
        f.write("  GDPR Privacy Auditor — RAGAS 基线评估报告\n")
        f.write(f"  生成时间: {report['timestamp']}\n")
        f.write(f"  RAGAS 版本: {report['ragas_version']}\n")
        f.write(bar + "\n\n")

        f.write("--- ChromaDB 知识库统计 ---\n")
        for name, cnt in report["chromadb_stats"].items():
            f.write(f"  {name}: {cnt} chunks\n")

        f.write("\n--- 检索质量辅助指标 ---\n")
        f.write(f"  空检索率:      {aux_metrics['empty_rate']*100:.0f}%\n")
        f.write(f"  条款命中率:    {aux_metrics['article_hit_rate']*100:.0f}%\n")
        f.write(f"  平均 chunk 数: {aux_metrics['avg_chunks']:.1f}\n")

        f.write("\n--- RAGAS 指标 ---\n")
        for mn, sc in scores.items():
            f.write(f"  {mn}: {sc:.4f}\n")

        f.write("\n--- 测试查询明细 ---\n")
        for i in range(len(questions)):
            f.write(f"\n[{i+1}] Q: {questions[i]}\n")
            f.write(f"    A: {answers[i][:200]}...\n")
            f.write(f"    GT: {ground_truths[i][:150]}...\n")
            f.write(f"    Contexts: {len(contexts_list[i])} chunks\n")

    print(f"\nTXT 报告: {os.path.abspath(txt_path)}")

    # JSON 报告
    json_path = "outputs/ragas_baseline_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"JSON 报告: {os.path.abspath(json_path)}")

    print(f"\n{bar}")
    print("  评估完成")
    print(bar)


if __name__ == "__main__":
    main()
