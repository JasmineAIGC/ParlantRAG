"""
RAGModule 主类，对外唯一入口。

宿主系统只需要实例化这个类，然后调用 register(agent)。
"""
from __future__ import annotations

import parlant.sdk as p

from .base import BaseRetriever, BaseEvaluator
from .retriever import KeywordRetriever
from .evaluator import RuleBasedEvaluator
from .tools import make_rag_tools


class RAGModule:
    """
    RAG + 检索结果评价模块。

    设计原则：
      - 对宿主系统零侵入，只通过 ``register()`` 方法注入
      - 检索器和评价器完全可替换
      - Guidelines 的行为逻辑封装在模块内部

    Basic usage::

        rag = RAGModule(documents=my_docs)
        await rag.register(agent)

    With FlexRAG knowledge base::

        from rag_module import RAGModule
        from rag_module.retriever import FlexRAGRetriever

        retriever = FlexRAGRetriever(
            persist_dir="./knowledge_base",
            embedding_base_url="http://127.0.0.1:8018/v1",
            embedding_api_key="sk-xxxx",
            embedding_model="Qwen3-Embedding-0.6B",
        )
        rag = RAGModule(retriever=retriever)
        await rag.register(agent)

    Custom retriever/evaluator::

        rag = RAGModule(
            retriever=MyRetriever(),
            evaluator=MyEvaluator(),
        )
        await rag.register(agent)
    """

    def __init__(
        self,
        retriever: BaseRetriever | None = None,
        evaluator: BaseEvaluator | None = None,
        documents: list[dict] | None = None,
    ) -> None:
        """
        Args:
            retriever:  检索器实例，不传则使用内置 KeywordRetriever。
            evaluator:  评价器实例，不传则使用内置 RuleBasedEvaluator。
            documents:  当使用内置 KeywordRetriever 时，传入文档列表。
                        格式：``[{"id": "...", "content": "...", "source": "..."}]``
        """
        if retriever is None:
            if documents is None:
                raise ValueError("未传入 retriever 时，必须提供 documents 参数")
            self._retriever = KeywordRetriever(documents)
        else:
            self._retriever = retriever

        self._evaluator = evaluator or RuleBasedEvaluator()

    async def register(self, agent: p.Agent) -> None:
        """
        将 RAG 工具和 Guidelines 注入到指定的 Parlant Agent。

        此方法是模块对外的唯一接口，调用后模块生效。
        可以在任意 Agent 上调用，不影响 Agent 的其他配置。

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
            action="""
            使用 rag_search 工具检索相关内容。
            从用户问题中提炼核心关键词作为 query。
            """,
            tools=[rag_search],
        )

        # Guideline 2：检索后评价
        await agent.create_guideline(
            condition="rag_search 工具已返回检索结果",
            action="""
            立即调用 evaluate_retrieval 工具评价本次检索结果。
            - user_question：用户的原始问题
            - retrieved_chunks_json：rag_search 返回的 retrieved_chunks 字段序列化为 JSON 字符串
            在拿到评价结论之前，不要开始作答。
            """,
            tools=[evaluate_retrieval],
        )

        # Guideline 3：评价合格 → 正常作答
        await agent.create_guideline(
            condition="evaluate_retrieval 返回的 verdict 为 qualified",
            action="""
            仅基于 rag_search 返回的检索内容作答，不添加检索结果之外的信息。
            在回答末尾注明信息来源文件名。
            """,
        )

        # Guideline 4：评价不足 → 告知用户
        await agent.create_guideline(
            condition="evaluate_retrieval 返回的 verdict 为 insufficient",
            action="""
            告知用户检索内容与问题的相关性不足，无法给出可靠回答。
            引用 evaluate_retrieval 的 conclusion 说明原因。
            建议用户用更具体的关键词重新提问。
            """,
        )

        # Guideline 5：无覆盖 → 诚实告知
        await agent.create_guideline(
            condition="""
            evaluate_retrieval 返回的 verdict 为 no_coverage，
            或 rag_search 返回的 retrieval_status 为 no_results
            """,
            action="""
            诚实告知用户：该问题超出了当前知识库的覆盖范围。
            不得凭自身知识作答。
            建议用户联系人工支持。
            """,
        )
