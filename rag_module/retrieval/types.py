"""
检索模块公共数据类型。

``Chunk`` 和 ``RetrievalResult`` 是检索子包与评估模块之间共享的数据结构。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """一个检索到的文档片段。"""
    chunk_id: str
    content: str
    source: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """检索器的输出，评估器的输入。"""
    query: str
    chunks: list[Chunk]
    status: str  # "success" | "no_results"
