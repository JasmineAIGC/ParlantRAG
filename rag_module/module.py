"""
RAGModule — 对外唯一注册入口。

宿主系统实例化此类，然后调用 ``register(agent)`` 即可将
RAG 工具和 Guidelines 注入到任意 Parlant Agent。
"""
from __future__ import annotations

import parlant.sdk as p

from rag_module.base import BaseRetriever, BaseEvaluator
from rag_module.retrieval import KeywordRetriever
from rag_module.tools import make_rag_tools


class RAGModule:
    """
    RAG 检索 + LLM 评估模块，零侵入接入 Parlant Agent。

    Usage（最简）::

        from rag_module import RAGModule, LLMEvaluator

        rag = RAGModule(
            documents=my_docs,
            evaluator=LLMEvaluator(api_key="sk-...", model="gpt-4o-mini"),
        )
        await rag.register(agent)

    Usage（完整）::

        from rag_module import RAGModule, LLMEvaluator
        from rag_module.retrieval import FlexRAGRetriever

        retriever = FlexRAGRetriever(
            persist_dir="./knowledge_base",
            embedding_base_url="http://127.0.0.1:8018/v1",
            embedding_api_key="sk-xxxx",
            embedding_model="Qwen3-Embedding-0.6B",
        )
        evaluator = LLMEvaluator(api_key="sk-...", model="gpt-4o-mini")
        rag = RAGModule(retriever=retriever, evaluator=evaluator)
        await rag.register(agent)
    """

    def __init__(
        self,
        evaluator: BaseEvaluator,
        retriever: BaseRetriever | None = None,
        documents: list[dict] | None = None,
    ) -> None:
        """
        Args:
            evaluator:  评估器实例（必填）。推荐使用 :class:`~rag_module.evaluation.LLMEvaluator`。
            retriever:  检索器实例。不传时使用内置 :class:`~rag_module.retrieval.KeywordRetriever`。
            documents:  当使用内置 KeywordRetriever 时，传入文档列表。
                        格式：``[{"id": "...", "content": "...", "source": "..."}]``
        """
        if retriever is None:
            if documents is None:
                raise ValueError(
                    "未传入 retriever 时，必须提供 documents 参数（用于内置 KeywordRetriever）。"
                )
            self._retriever: BaseRetriever = KeywordRetriever(documents)
        else:
            self._retriever = retriever

        self._evaluator = evaluator

    async def register(self, agent: p.Agent) -> None:
        """
        将 RAG 工具和 Guidelines 注入到指定的 Parlant Agent。

        注入内容：
          - 工具 ``rag_search``：执行向量检索
          - 工具 ``evaluate_retrieval``：LLM 综合评估检索结果
          - Guidelines：驱动 Agent 按「检索 → 评估 → 作答」流程执行

        Args:
            agent: 已创建的 Parlant Agent 实例。
        """
        rag_search, evaluate_retrieval = make_rag_tools(
            retriever=self._retriever,
            evaluator=self._evaluator,
        )

        # Guideline 1：触发检索
        await agent.create_guideline(
            condition="用户提出了需要查询知识库才能回答的问题",
            action=(
                "使用 rag_search 工具检索相关内容。"
                "从用户问题中提炼核心关键词作为 query。"
            ),
            tools=[rag_search],
        )

        # Guideline 2：检索后立即评估
        await agent.create_guideline(
            condition="rag_search 工具已返回检索结果",
            action=(
                "立即调用 evaluate_retrieval 工具评估本次检索结果。"
                "- user_question：用户的原始问题"
                "- retrieved_chunks_json：rag_search 返回的 retrieved_chunks 字段序列化为 JSON 字符串。"
                "在拿到评估结论之前，不要开始作答。"
            ),
            tools=[evaluate_retrieval],
        )

        # Guideline 3：评估合格 → 正常作答
        await agent.create_guideline(
            condition="evaluate_retrieval 返回的 verdict 为 qualified",
            action=(
                "仅基于 rag_search 返回的检索内容作答，不添加检索结果之外的信息。"
                "在回答末尾注明信息来源文件名。"
            ),
        )

        # Guideline 4：评估不足 → 告知用户
        await agent.create_guideline(
            condition="evaluate_retrieval 返回的 verdict 为 insufficient",
            action=(
                "告知用户检索内容与问题的相关性不足，无法给出可靠回答。"
                "引用 evaluate_retrieval 的 conclusion 说明原因。"
                "建议用户用更具体的关键词重新提问。"
            ),
        )

        # Guideline 5：无覆盖 → 诚实告知
        await agent.create_guideline(
            condition=(
                "evaluate_retrieval 返回的 verdict 为 no_coverage，"
                "或 rag_search 返回的 retrieval_status 为 no_results"
            ),
            action=(
                "诚实告知用户：该问题超出了当前知识库的覆盖范围。"
                "不得凭自身知识作答。建议用户联系人工支持。"
            ),
        )
