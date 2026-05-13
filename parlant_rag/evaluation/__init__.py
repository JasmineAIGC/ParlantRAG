"""
评估子包。

提供多信号融合 + LLM 综合评估能力：

数据类：
  - EvaluationResult  — 评估输出（verdict / relevance / coverage / confidence）
  - RetrievalSignals  — 本地预计算量化信号（内部使用）

评估器：
  - LLMEvaluator      — 先本地计算量化信号，再由 LLM 综合评判

所有 LLM 配置从项目根目录 .env 文件自动加载（QWEN3_* 优先）。
"""
from parlant_rag.evaluation.evaluator import EvaluationResult, RetrievalSignals, LLMEvaluator

__all__ = [
    "EvaluationResult",
    "RetrievalSignals",
    "LLMEvaluator",
]
