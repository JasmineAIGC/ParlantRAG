"""
工具类子包 — Parlant 框架集成层。

将检索器和评估器封装为 Parlant 工具，并提供零侵入注册能力：

  - make_rag_tools  — 生成 rag_search / evaluate_retrieval 两个 Parlant 工具
  - RAGModule       — 一行 register(agent) 注入任意 Parlant Agent

该子包是 parlant_rag 与 Parlant 框架之间的唯一接口。
需要安装 parlant 才能使用：pip install parlant
"""
try:
    from parlant_rag.tools.parlant_tools import make_rag_tools
    from parlant_rag.tools.module import RAGModule
    __all__ = ["make_rag_tools", "RAGModule"]
except ImportError:
    # parlant 未安装时，tools 包仍可导入，但内容为空
    __all__ = []
