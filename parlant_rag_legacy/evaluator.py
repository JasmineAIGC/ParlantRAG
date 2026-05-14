"""
评估器 — 使用 model_client 的 LLMClient 进行检索结果评估。
"""
from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

from model_client.api.llm import LLMClient

from parlant_rag.retriever import RetrievalResult
from utils.logging import get_logger


logger = get_logger("parlant_rag.evaluator")


@dataclass
class EvaluationResult:
    """评估结果。"""
    verdict: str  # qualified, insufficient, no_coverage
    relevance: float  # 0-1
    coverage: float  # 0-1
    confidence: float  # 0-1
    conclusion: str
    action_hint: str


class Evaluator:
    """基于 model_client 的检索结果评估器。"""
    
    def __init__(self, llm_client: LLMClient):
        self._llm_client = llm_client
    
    async def evaluate(self, question: str, retrieval: RetrievalResult) -> EvaluationResult:
        """评估检索结果质量。"""
        logger.info("[Evaluator] 开始评估  question=%r  chunk_count=%d", question, len(retrieval.chunks))
        
        if not retrieval.chunks:
            logger.info("[Evaluator] 未检索到任何片段，返回 no_coverage")
            return EvaluationResult(
                verdict="no_coverage",
                relevance=0.0,
                coverage=0.0,
                confidence=0.0,
                conclusion="未检索到任何相关片段",
                action_hint="告知用户当前问题超出知识库覆盖范围",
            )
        
        # 构建评估提示
        context_text = "\n\n".join([
            f"[片段 {i+1}, 相关性: {c.score:.2f}]\n{c.content}"
            for i, c in enumerate(retrieval.chunks)
        ])
        
        prompt = f"""
请评估以下检索结果是否能回答用户问题：

用户问题：{question}

检索到的内容：
{context_text}

请输出 JSON 格式，包含以下字段：
- verdict: "qualified"（可回答）/ "insufficient"（信息不足）/ "no_coverage"（无覆盖）
- relevance: 检索内容与问题的相关程度（0-1）
- coverage: 检索内容覆盖问题要点的程度（0-1）
- confidence: 基于检索内容回答问题的信心程度（0-1）
- conclusion: 简要说明评估理由
- action_hint: 下一步应该怎么做

只需输出 JSON，不要包含其他内容。
"""
        
        # 使用 model_client 的 LLMClient 调用 LLM
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await self._llm_client.chat_async(messages)
            logger.info("[Evaluator] LLM 评估完成")
        except Exception as e:
            logger.exception("[Evaluator] LLM 调用失败")
            return EvaluationResult(
                verdict="insufficient",
                relevance=0.5,
                coverage=0.5,
                confidence=0.3,
                conclusion=f"LLM 评估失败: {str(e)[:100]}",
                action_hint="返回评估错误",
            )
        
        # 解析 JSON 响应
        try:
            import json
            parsed = json.loads(response)
            
            result = EvaluationResult(
                verdict=parsed.get("verdict", "insufficient"),
                relevance=float(parsed.get("relevance", 0.5)),
                coverage=float(parsed.get("coverage", 0.5)),
                confidence=float(parsed.get("confidence", 0.5)),
                conclusion=parsed.get("conclusion", "评估完成"),
                action_hint=parsed.get("action_hint", "根据评估结果作答"),
            )
            
            logger.info(
                "[Evaluator] 评估结果  verdict=%s  relevance=%.3f  coverage=%.3f  confidence=%.3f",
                result.verdict, result.relevance, result.coverage, result.confidence
            )
            
            return result
            
        except json.JSONDecodeError:
            logger.warning("[Evaluator] JSON 解析失败  response=%r", response[:200])
            return EvaluationResult(
                verdict="insufficient",
                relevance=0.5,
                coverage=0.5,
                confidence=0.5,
                conclusion=f"评估响应解析失败: {response[:100]}",
                action_hint="返回解析错误",
            )
