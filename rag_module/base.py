"""
抽象接口层。
所有的检索器和评价器都必须实现这两个接口。
宿主系统只需要实现接口，不需要了解模块内部细节。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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
    """检索结果"""
    query: str
    chunks: list[Chunk]
    status: str  # "success" | "no_results"


@dataclass
class EvaluationResult:
    """评价结果"""
    verdict: str          # "qualified" | "insufficient" | "no_coverage"
    relevance: float      # 相关性得分 0~1
    coverage: float       # 覆盖度得分 0~1
    confidence: float     # 综合置信度 0~1
    conclusion: str       # 人类可读的结论
    action_hint: str      # 给 Agent 的后续行动提示
    details: dict = field(default_factory=dict)


class BaseRetriever(ABC):
    """
    检索器接口。
    实现此接口即可接入任意向量数据库或检索系统。
    """

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
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
    实现此接口即可替换评价逻辑（规则评价 / LLM 评价 / 混合评价）。
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
            retrieval: 检索器返回的结果

        Returns:
            EvaluationResult
        """
        ...
