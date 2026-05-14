"""
评估器模块 — 基于 Parlant SchematicGenerator 的检索结果评估。

核心特性：
- 使用 Parlant SchematicGenerator 进行结构化输出评估
- 评估检索结果的相关性和覆盖度
- 提供可操作的改进建议

评估维度：
1. relevance: 检索结果与查询的相关程度
2. coverage: 检索结果对查询的覆盖程度
3. confidence: 评估结果的置信度
4. verdict: 最终评估结论（qualified/insufficient/no_coverage）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from parlant.core.nlp.generation import SchematicGenerator

from parlant_rag.retriever import RetrievalResult, RetrievedChunk
from parlant_rag.adapters import ModelClientSchematicGenerator
from utils.logging import get_logger


logger = get_logger("parlant_rag.evaluator")


@dataclass
class EvaluationResult:
    """检索结果评估结果。"""
    verdict: str  # qualified, insufficient, no_coverage
    relevance: float  # 0-1
    coverage: float  # 0-1
    confidence: float  # 0-1
    conclusion: str
    action_hint: str


class Evaluator:
    """
    基于 Parlant 框架的检索结果评估器。
    
    使用 Parlant SchematicGenerator 进行结构化输出，确保与 Parlant 框架的兼容性。
    
    评估流程：
    1. 分析查询与检索结果的匹配程度
    2. 使用 LLM 评估相关性和覆盖度
    3. 生成结构化评估结果
    4. 提供改进建议
    """
    
    def __init__(self, generator: Optional[SchematicGenerator[dict]] = None):
        """
        初始化评估器。
        
        Args:
            generator: Parlant SchematicGenerator 实例，用于生成结构化评估结果
        """
        self._generator = generator or ModelClientSchematicGenerator(dict)
    
    async def evaluate(self, question: str, retrieval: RetrievalResult) -> EvaluationResult:
        """
        评估检索结果。
        
        Args:
            question: 用户问题
            retrieval: 检索结果
        
        Returns:
            EvaluationResult: 评估结果
        """
        logger.info(f"[Evaluator] 开始评估，查询: {question!r}")
        
        # 如果没有检索到结果
        if retrieval.status == "no_coverage" or not retrieval.chunks:
            logger.info("[Evaluator] 未检索到相关文档")
            return EvaluationResult(
                verdict="no_coverage",
                relevance=0.0,
                coverage=0.0,
                confidence=0.8,
                conclusion="未找到与问题相关的文档",
                action_hint="请尝试调整问题描述，或添加更多相关文档到知识库",
            )
        
        # 如果检索出错
        if retrieval.status == "error":
            logger.info("[Evaluator] 检索过程出错")
            return EvaluationResult(
                verdict="no_coverage",
                relevance=0.0,
                coverage=0.0,
                confidence=0.9,
                conclusion="检索过程发生错误",
                action_hint="请检查网络连接或重试",
            )
        
        # 构建评估提示
        context_text = "\n\n".join([
            f"文档 {i+1} (相似度: {chunk.score:.2f}):\n{chunk.content}"
            for i, chunk in enumerate(retrieval.chunks)
        ])
        
        prompt = f"""
请对以下检索结果进行评估：

用户问题：{question}

检索到的文档：
{context_text}

请按照以下格式输出评估结果（JSON格式）：
{{
    "relevance": 相关性评分（0-1，越高表示越相关）,
    "coverage": 覆盖度评分（0-1，越高表示覆盖越全面）,
    "confidence": 评估置信度（0-1）,
    "verdict": "qualified"（合格）或 "insufficient"（不足）,
    "conclusion": 评估结论描述,
    "action_hint": 如果评估不合格，提供改进建议
}}

评估标准：
- relevance >= 0.7 且 coverage >= 0.7: qualified
- 其他情况: insufficient
"""
        
        try:
            result = await self._generator.generate(prompt)
            
            # 解析评估结果
            relevance = result.get("relevance", 0.5)
            coverage = result.get("coverage", 0.5)
            confidence = result.get("confidence", 0.7)
            verdict = result.get("verdict", "insufficient")
            conclusion = result.get("conclusion", "评估完成")
            action_hint = result.get("action_hint", "")
            
            # 确保值在有效范围内
            relevance = max(0.0, min(1.0, float(relevance)))
            coverage = max(0.0, min(1.0, float(coverage)))
            confidence = max(0.0, min(1.0, float(confidence)))
            
            # 根据评分确定最终结论
            if relevance >= 0.7 and coverage >= 0.7:
                verdict = "qualified"
            elif relevance < 0.3 or coverage < 0.3:
                verdict = "insufficient"
            
            logger.info(f"[Evaluator] 评估完成，verdict: {verdict}, relevance: {relevance:.2f}, coverage: {coverage:.2f}")
            
            return EvaluationResult(
                verdict=verdict,
                relevance=relevance,
                coverage=coverage,
                confidence=confidence,
                conclusion=conclusion,
                action_hint=action_hint,
            )
        
        except Exception as e:
            logger.exception("[Evaluator] 评估失败，使用简单规则评估")
            
            # 回退到简单规则评估
            avg_score = sum(chunk.score for chunk in retrieval.chunks) / len(retrieval.chunks)
            
            if avg_score >= 0.7:
                verdict = "qualified"
                conclusion = "检索结果质量较高，可以用于回答"
            else:
                verdict = "insufficient"
                conclusion = "检索结果质量一般，建议调整检索策略"
            
            return EvaluationResult(
                verdict=verdict,
                relevance=avg_score,
                coverage=min(1.0, len(retrieval.chunks) / 3),
                confidence=0.6,
                conclusion=conclusion,
                action_hint="",
            )
