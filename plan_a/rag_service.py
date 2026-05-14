"""
rag_service.py — 向后兼容存根。

此文件已重构，实现已拆分到独立模块：

  - plan_a.rag_pipeline  → RAGPipeline（原 RagService）
  - plan_a.types         → DocChunk

请直接使用新模块。
"""
from plan_a.rag_pipeline import RAGPipeline as RagService  # noqa: F401
from plan_a.types import DocChunk  # noqa: F401

__all__ = ["RagService", "DocChunk"]
