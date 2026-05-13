"""
parlant_rag — 向后兼容入口。

新实现已迁移到 ``rag_module``，不再依赖 FlexRAG。建议新代码直接导入
``rag_module``；此包仅保留原导入路径的兼容性。
"""
from rag_module import *  # noqa: F403
from rag_module import __all__  # noqa: F401
