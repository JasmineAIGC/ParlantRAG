"""向后兼容入口：请优先使用 ``rag_module.evaluation``。"""
from rag_module.evaluation import *  # noqa: F403
from rag_module.evaluation import __all__  # noqa: F401
