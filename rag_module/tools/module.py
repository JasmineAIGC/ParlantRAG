"""RAGModule — Parlant Agent 的零侵入注册入口。"""
import parlant.sdk as p

from rag_module.retrieval import KeywordRetriever, ParlantRetriever
from rag_module.tools.parlant_tools import make_rag_tools


class RAGModule:
    """
    RAG 检索 + LLM 评估模块，使用 Parlant 官方 SDK 工具接入 Agent。

    不传 ``retriever`` 时会用 ``documents`` 创建关键词检索器；生产环境建议传入
    ``ParlantRetriever`` 或任意实现 ``retrieve(query, top_k)`` 的检索器。
    """

    def __init__(
        self,
        evaluator: object,
        retriever: object | None = None,
        documents: list[dict] | None = None,
    ) -> None:
        if retriever is None:
            if documents is None:
                raise ValueError("未传入 retriever 时，必须提供 documents 参数。")
            retriever = ParlantRetriever(KeywordRetriever(documents))
        self._retriever = retriever
        self._evaluator = evaluator

    async def register(self, agent: p.Agent) -> None:
        """将 RAG 工具和 Guidelines 注入指定的 Parlant Agent。"""
        rag_search, evaluate_retrieval = make_rag_tools(self._retriever, self._evaluator)

        await agent.create_guideline(
            condition="用户提出了需要查询知识库才能回答的问题",
            action="使用 rag_search 工具检索相关内容，从用户问题中提炼核心关键词作为 query。",
            tools=[rag_search],
        )
        await agent.create_guideline(
            condition="rag_search 工具已返回检索结果",
            action=(
                "立即调用 evaluate_retrieval 工具评估本次检索结果。"
                "user_question 使用用户原始问题；retrieved_chunks_json 使用 rag_search "
                "返回的 retrieved_chunks 字段序列化 JSON。评估完成前不要开始作答。"
            ),
            tools=[evaluate_retrieval],
        )
        await agent.create_guideline(
            condition="evaluate_retrieval 返回的 verdict 为 qualified",
            action="仅基于 rag_search 返回的检索内容作答，不添加检索结果之外的信息，并注明来源。",
        )
        await agent.create_guideline(
            condition="evaluate_retrieval 返回的 verdict 为 insufficient",
            action="告知用户检索内容不足以可靠回答，引用评估结论说明原因，并建议更具体地提问。",
        )
        await agent.create_guideline(
            condition="evaluate_retrieval 返回的 verdict 为 no_coverage，或 rag_search 返回 no_results",
            action="诚实告知该问题超出当前知识库覆盖范围，不要凭自身知识作答。",
        )
