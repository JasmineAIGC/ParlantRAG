"""
评估子包。

唯一实现：:class:`LLMEvaluator` — 结合多维量化信号，利用 LLM 进行综合评估。
"""
from rag_module.evaluation.llm_evaluator import LLMEvaluator

__all__ = ["LLMEvaluator"]
