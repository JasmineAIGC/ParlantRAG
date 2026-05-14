"""
tools.py — 向后兼容存根。

此文件已重构，请使用 plan_a.rag_tool 模块。
"""
from plan_a.rag_tool import make_rag_tool  # noqa: F401

__all__ = ["make_rag_tool"]
