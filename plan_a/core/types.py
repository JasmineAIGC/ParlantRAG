"""
types.py — Plan A 模块共享数据类型。

此文件不依赖任何外部模块，只使用标准库。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


_SCORE_PRECISION = 4
"""score 字段序列化时保留的小数位数。"""


@dataclass
class DocChunk:
    """
    检索到的文档片段。

    由 VectorStore 生成，经 RAGPipeline 返回，
    最终传入 Scoped Retriever / RAG Tool 序列化给 Parlant。
    """

    content: str
    """文档正文内容。"""

    source: str
    """文档来源标识（文件名、URL 等）。"""

    score: float
    """相似度分数（越高越相关）。"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """可选的附加元数据（来自文档库）。"""

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，兼容 Parlant Tool/Retriever 返回格式。"""
        return {
            "content": self.content,
            "source": self.source,
            "score": round(self.score, _SCORE_PRECISION),
        }

    def __repr__(self) -> str:
        return f"DocChunk(source={self.source!r}, score={self.score:.3f})"
