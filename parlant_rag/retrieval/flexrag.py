"""
已移除的旧 FlexRAG 检索器。

请迁移到 ``rag_module.retrieval.ParlantRetriever``，并按需选择
``from_keyword``、``from_dense`` 或 ``from_dense_env`` 构建不依赖 FlexRAG 的检索器。
"""


class FlexRAGRetriever:  # pragma: no cover - compatibility error path
    """兼容占位类：实例化时提示迁移到 ParlantRetriever。"""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "FlexRAGRetriever 已移除。请使用 "
            "rag_module.retrieval.ParlantRetriever / DenseRetriever / KeywordRetriever。"
        )

    @classmethod
    def from_env(cls, *args, **kwargs) -> "FlexRAGRetriever":
        raise RuntimeError(
            "FlexRAGRetriever.from_env() 已移除。请使用 "
            "ParlantRetriever.from_dense_env(documents) 或 ParlantRetriever.from_keyword(documents)。"
        )
