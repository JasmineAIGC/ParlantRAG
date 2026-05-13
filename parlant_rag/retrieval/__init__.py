"""向后兼容入口：请优先使用 ``rag_module.retrieval``。"""
from rag_module.retrieval import *  # noqa: F403
from rag_module.retrieval import __all__  # noqa: F401
