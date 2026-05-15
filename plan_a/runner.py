"""
runner.py — Plan A 两种 RAG 模式的批量 query-answer runner。

将 Standalone 和 Parlant 两种 RAG 模式的计算/设置逻辑集中在此，
供 ``examples/`` 脚本直接调用；examples 仅负责参数解析与结果输出。

接口一览
--------
* :func:`load_test_queries`    — 从 JSON 数据集抽样测试问题
* :func:`generate_answer`      — 用 LLM 根据检索片段生成回答（公共辅助函数）
* :func:`run_standalone`       — Standalone 模式：Pipeline 检索 + LLM 回答（无需 Parlant）
* :func:`run_parlant`          — Parlant 模式：RAG Tool 检索 + LLM 回答（**无需 Server，后台执行，不弹 UI**）
* :func:`run_parlant_server`   — Parlant Server 模式：完整 Server + Agent + GL + Tool（含 NLP 桥接）
"""
from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any, Optional

from plan_a.core.types import DocChunk, RetrievalRequest
from plan_a.rag.pipeline import RAGPipeline
from utils.logging import get_logger

if TYPE_CHECKING:
    from model_client.api.llm import LLMClient

logger = get_logger("plan_a.runner")


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_test_queries(path: str | Path, limit: int = 3, seed: int = 0) -> list[str]:
    """
    从 JSON 数据集文件中抽样测试问题。

    Args:
        path:  JSON 文件路径（格式：``list of {"question": str, ...}``）。
        limit: 返回的问题数量上限。
        seed:  随机抽样的固定种子，保证结果可复现。

    Returns:
        问题文本列表（长度 ``<= limit``）。

    Raises:
        ValueError: 若文件中没有可用问题。
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    questions = [
        item["question"].strip()
        for item in samples
        if isinstance(item, dict) and isinstance(item.get("question"), str) and item["question"].strip()
    ]
    if not questions:
        raise ValueError(f"未从数据集读取到可用问题：{path}")
    if len(questions) <= limit:
        return questions
    return Random(seed).sample(questions, limit)


# ---------------------------------------------------------------------------
# 公共辅助：LLM 回答生成
# ---------------------------------------------------------------------------

async def generate_answer(llm: "LLMClient", question: str, chunks: list[DocChunk]) -> str:
    """
    根据检索到的文档片段用 LLM 生成回答。

    Args:
        llm:      :class:`~model_client.api.llm.LLMClient` 实例（运行时必须可用）。
        question: 用户问题。
        chunks:   RAGPipeline 返回的文档片段列表。

    Returns:
        LLM 生成的回答文本；无相关内容时返回固定提示语。
    """
    if not chunks:
        return "知识库中未找到与该问题相关的内容，无法回答。"
    context_text = "\n\n".join(
        f"[来源: {c.metadata.get('title') or c.source}]\n{c.content}"
        for c in chunks
    )
    prompt = (
        "请根据以下知识库内容简洁回答用户问题。若内容不足以回答，请如实说明。\n\n"
        f"用户问题：{question}\n\n"
        f"知识库内容：\n{context_text}\n\n回答："
    )
    try:
        return await llm.chat_async([{"role": "user", "content": prompt}])
    except Exception as exc:
        logger.error("[Runner] LLM 调用失败: %s", exc)
        return f"LLM 调用失败，无法生成回答：{exc}"


# ---------------------------------------------------------------------------
# 内部辅助：通过 Parlant RAG Tool 执行一组查询（供两种 Parlant 模式复用）
# ---------------------------------------------------------------------------

async def _run_tool_queries(
    search_knowledge,
    queries: list[str],
    llm: Optional["LLMClient"] = None,
) -> list[dict]:
    """
    内部辅助：通过 Parlant RAG Tool 执行一组查询。

    直接以 ``(None, query)`` 调用 ``@p.tool`` 函数（``None`` 表示无 Parlant 会话上下文），
    等同于 Parlant Agent 内部触发 Tool 调用时的检索路径，但无需启动 Server。

    Args:
        search_knowledge: :func:`~plan_a.tools.rag.make_rag_tool` 返回的 Parlant Tool 函数。
        queries:          查询文本列表。
        llm:              可选 LLMClient；提供时基于检索结果生成回答，省略时只返回检索结果。

    Returns:
        每条查询对应一个 dict：``{"request": {...}, "response": {...}}``，
        提供 llm 时还包含 ``"answer"`` 键。
    """
    results: list[dict] = []
    for query in queries:
        # None = ToolContext（直接调用时无 Parlant 会话上下文）
        tool_result = await search_knowledge(None, query)
        tool_data: dict[str, Any] = getattr(tool_result, "data", {}) if tool_result is not None else {}
        tool_resp: dict[str, Any] = tool_data.get("response") or {}
        chunk_count: int = tool_resp.get("chunk_count", 0)

        logger.info("[Runner] Tool 检索完成  query=%r  chunk_count=%d", query, chunk_count)

        row: dict[str, Any] = {
            "request": tool_data.get("request") or {"query": query},
            "response": tool_resp,
        }

        if llm is not None:
            formatted: str = tool_resp.get("formatted") or ""
            if formatted:
                answer_prompt = (
                    "请根据以下知识库内容简洁回答用户问题。若内容不足以回答，请如实说明。\n\n"
                    f"用户问题：{query}\n\n"
                    f"知识库内容：\n{formatted}\n\n回答："
                )
                try:
                    answer = await llm.chat_async([{"role": "user", "content": answer_prompt}])
                    row["answer"] = answer.strip()
                except Exception as exc:
                    logger.error("[Runner] LLM 调用失败: %s", exc)
                    row["answer"] = f"LLM 调用失败：{exc}"
            else:
                row["answer"] = "知识库中未找到与该问题相关的内容，无法回答。"

        results.append(row)
    return results


# ---------------------------------------------------------------------------
# Standalone 模式
# ---------------------------------------------------------------------------

async def run_standalone(
    pipeline: RAGPipeline,
    queries: list[str],
    llm: "LLMClient",
) -> list[dict]:
    """
    Standalone 模式 Q&A：RAGPipeline 直接检索 → LLMClient 生成回答。

    无需安装 Parlant，适合脚本测试与批量评估。

    Args:
        pipeline: 已初始化的 :class:`~plan_a.rag.pipeline.RAGPipeline`。
        queries:  查询文本列表。
        llm:      :class:`~model_client.api.llm.LLMClient` 实例。

    Returns:
        每条查询对应一个 dict：``{"retrieval": {...}, "answer": str}``。
    """
    results: list[dict] = []
    for query in queries:
        logger.info("[Runner.Standalone] query=%r", query)
        response = await pipeline.retrieve_standard(RetrievalRequest(query=query))
        answer = await generate_answer(llm, query, response.chunks)
        results.append({
            "retrieval": response.to_dict(include_metadata=False),
            "answer": answer.strip(),
        })
        logger.info("[Runner.Standalone] 完成  chunks=%d", len(response.chunks))
    return results


# ---------------------------------------------------------------------------
# Parlant 模式（无 Server，后台执行，不弹 UI）
# ---------------------------------------------------------------------------

async def run_parlant(
    pipeline: RAGPipeline,
    queries: list[str],
    llm: Optional["LLMClient"] = None,
) -> list[dict]:
    """
    Parlant 模式 Q&A（**无需启动 Parlant Server，后台执行，不弹 UI**）。

    通过 :func:`~plan_a.tools.rag.make_rag_tool` 获取 ``@p.tool`` 检索函数，
    在后台程序化调用，等同于 Parlant Agent 内部触发 Tool 时的检索路径，
    但不启动任何 HTTP Server 或 Web 界面。

    Args:
        pipeline: 已初始化的 :class:`~plan_a.rag.pipeline.RAGPipeline`。
        queries:  查询文本列表。
        llm:      可选 :class:`~model_client.api.llm.LLMClient`；
                  提供时在检索后生成 LLM 回答，省略时只返回检索结果。

    Returns:
        每条查询对应一个 dict：
        ``{"request": {...}, "response": {...}}``，提供 llm 时还包含 ``"answer"``。

    Raises:
        ImportError: 若未安装 ``parlant``（pip install parlant）。
        RuntimeError: 若未安装 ``torch``（Parlant LiteLLM 适配器依赖）。
    """
    from plan_a.tools import make_rag_tool

    search_knowledge = make_rag_tool(pipeline)
    logger.info("[Runner.Parlant] 开始（无 Server，后台执行）  query_count=%d", len(queries))
    results = await _run_tool_queries(search_knowledge, queries, llm)
    logger.info("[Runner.Parlant] 完成  result_count=%d", len(results))
    return results


# ---------------------------------------------------------------------------
# Parlant Server 模式（完整集成：Server + Agent + GL + Tool）
# ---------------------------------------------------------------------------

async def run_parlant_server(
    pipeline: RAGPipeline,
    queries: list[str],
    agent_name: str = "plan-a-rag-agent",
    agent_description: str = "Plan A RAG Agent：知识库问答助手",
    llm: Optional["LLMClient"] = None,
) -> tuple[dict, list[dict]]:
    """
    Parlant Server 模式（完整集成）：启动 Parlant Server，创建 Agent，
    配置 Guidelines（SR + Tool），在后台程序化执行 query-answer，不弹出 UI。

    NLP 服务使用 ``NLPServices.litellm``，模型来自 ``.env`` 配置（``QWEN3_*``/``LLM_*``），
    由 :func:`~model_client._utils.bridge_parlant_litellm_env` 自动桥接。

    Args:
        pipeline:          已初始化的 :class:`~plan_a.rag.pipeline.RAGPipeline`。
        queries:           查询文本列表。
        agent_name:        Parlant Agent 名称。
        agent_description: Parlant Agent 描述。
        llm:               可选 LLMClient（用于生成回答）。

    Returns:
        ``(agent_info, results)``：
        - ``agent_info`` — 包含 Agent ID 和 Guideline ID 的元数据字典；
        - ``results``    — 每条查询对应一个结果字典（同 :func:`run_parlant`）。

    Raises:
        ImportError: 若未安装 ``parlant``（pip install parlant）。
        RuntimeError: 若未安装 ``torch``（Parlant LiteLLM 适配器依赖）。
    """
    try:
        import parlant.sdk as p
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) == "torch" or "No module named 'torch'" in str(exc):
            raise RuntimeError("Parlant 依赖 torch，请先安装：pip install torch") from exc
        raise RuntimeError("未安装 parlant，请先安装：pip install parlant") from exc

    from model_client._utils import bridge_parlant_litellm_env
    from plan_a.guidelines import setup_guidelines
    from plan_a.tools import make_rag_tool

    bridge_parlant_litellm_env()
    logger.info("[Runner.ParlantServer] 启动 Parlant Server（NLP: LiteLLM）")

    async with p.Server(nlp_service=p.NLPServices.litellm) as server:
        agent = await server.create_agent(name=agent_name, description=agent_description)
        scoped_gl, tool_gl = await setup_guidelines(agent, pipeline)
        search_knowledge = make_rag_tool(pipeline)
        logger.info("[Runner.ParlantServer] Agent 就绪  id=%s", agent.id)

        results = await _run_tool_queries(search_knowledge, queries, llm)

    agent_info: dict[str, Any] = {
        "id": agent.id,
        "name": agent_name,
        "guidelines": {
            "scoped_retriever": scoped_gl.id,
            "rag_tool": tool_gl.id,
        },
    }
    logger.info("[Runner.ParlantServer] 完成  result_count=%d", len(results))
    return agent_info, results
