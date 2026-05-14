"""
agent_builder.py — 已弃用。

Agent[Parlant Agent] 是系统的**外部入口**，不属于 RAG 功能模块。
请参考 examples/05_plan_a_agent.py 了解如何在外部脚本中创建 Agent
并通过 plan_a.guidelines.setup_guidelines() 挂载 RAG 功能。

RAG 模块公开 API 见 plan_a/__init__.py。
"""
import warnings

warnings.warn(
    "plan_a.agent_builder 已弃用。\n"
    "Agent 创建属于外部入口，不在 plan_a/ RAG 模块内。\n"
    "请参考 examples/05_plan_a_agent.py，并使用 plan_a.guidelines.setup_guidelines() 挂载 RAG 功能。",
    DeprecationWarning,
    stacklevel=2,
)
