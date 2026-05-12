"""
抽象接口层与共享数据类。

所有的检索器和评价器都必须实现这里定义的接口。
数据类在检索子包和评估子包之间共享，是两者唯一的耦合点。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 共享数据类
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """一个检索到的文档片段"""
    chunk_id: str
    content: str
    source: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """检索子包的输出，评估子包的输入"""
    query: str
    chunks: list[Chunk]
    status: str  # "success" | "no_results"


@dataclass
class RetrievalSignals:
    """
    评估前预计算的量化信号。

    由 :class:`~rag_module.evaluation.LLMEvaluator` 内部计算，
    连同原始片段一起送入 LLM，辅助其做出全面判断。
    """
    avg_score: float          # 所有片段的平均向量相似度
    max_score: float          # 最高片段得分
    min_score: float          # 最低片段得分
    keyword_coverage: float   # 问题关键词被片段覆盖的比例 (0~1)
    source_diversity: int     # 不同来源文件数量
    chunk_count: int          # 片段总数
    missing_keywords: list[str]  # 未被覆盖的问题关键词


@dataclass
class EvaluationResult:
    """评估子包的输出"""
    verdict: str          # "qualified" | "insufficient" | "no_coverage"
    relevance: float      # 相关性得分 0~1
    coverage: float       # 覆盖度得分 0~1
    confidence: float     # 综合置信度 0~1
    conclusion: str       # 人类可读的结论
    action_hint: str      # 给 Agent 的后续行动提示
    signals: RetrievalSignals | None = None  # 辅助判断所用的量化信号


# ---------------------------------------------------------------------------
# 抽象接口
# ---------------------------------------------------------------------------

class BaseRetriever(ABC):
    """
    检索器接口。
    实现此接口即可接入任意向量数据库或检索系统。
    """

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        从知识库检索与 query 最相关的片段。

        Args:
            query:  检索查询词
            top_k:  返回片段数量上限

        Returns:
            RetrievalResult
        """
        ...


class BaseEvaluator(ABC):
    """
    评价器接口。
    实现此接口即可替换评价逻辑。
    """

    @abstractmethod
    async def evaluate(
        self,
        question: str,
        retrieval: RetrievalResult,
    ) -> EvaluationResult:
        """
        评价检索结果的质量。

        Args:
            question:  用户的原始问题
            retrieval: 检索器返回的结果（:class:`RetrievalResult`）

        Returns:
            EvaluationResult
        """
        ...
