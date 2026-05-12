"""
LLMEvaluator — 多信号融合 + LLM 综合评估器。

评估流程
--------
1. **本地预计算量化信号**（无需 LLM）：
   - 向量相似度分布（max / min / avg）
   - 问题关键词覆盖率 & 未覆盖词列表
   - 来源多样性（不同文件数量）

2. **构造富文本 Prompt**，将量化信号与原始片段一并送入 LLM：
   LLM 可以同时参考：
   - 片段与问题的语义关联（自然语言理解）
   - 量化信号（数值佐证）
   - 来源可信度与多样性

3. **解析 LLM 返回的 JSON**，填充 :class:`~rag_module.base.EvaluationResult`。

LLM 调用参考 ``temp/llm/llm_client.py`` 的 ``AdkLLMClient`` 模式，
使用 ``LLMClient`` 通过 ``litellm.acompletion()`` 接入本地 vLLM / OpenAI
兼容端点，配置通过 ``LLMConfig`` 从 ``.env`` 文件加载。

依赖::

    pip install litellm python-dotenv
"""
from __future__ import annotations

import json
import re
from typing import Optional

from rag_module.base import (
    BaseEvaluator,
    EvaluationResult,
    RetrievalResult,
    RetrievalSignals,
)
from rag_module.llm_client import LLMClient, LLMConfig

# 用于从问题中提取关键词时过滤的停用词
_STOPWORDS = {
    "的", "了", "是", "在", "我", "有", "和", "与", "或", "吗", "呢", "吧",
    "一个", "这个", "那个", "什么", "怎么", "为什么", "如何", "请问",
    "what", "how", "why", "is", "the", "a", "an", "of", "in", "to", "do",
    "does", "did", "can", "could", "would", "should", "will",
}


def _compute_signals(question: str, retrieval: RetrievalResult) -> RetrievalSignals:
    """
    从检索结果中本地预计算量化信号，供 LLM Prompt 使用。

    这些信号是纯数值统计，不依赖 LLM，计算开销极低。
    """
    chunks = retrieval.chunks

    # 向量相似度分布
    scores = [c.score for c in chunks]
    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    max_score = round(max(scores), 4) if scores else 0.0
    min_score = round(min(scores), 4) if scores else 0.0

    # 问题关键词覆盖率
    question_words = {
        w for w in question.lower().split()
        if w and w not in _STOPWORDS and len(w) > 1
    }
    all_content = " ".join(c.content.lower() for c in chunks)
    content_words = set(all_content.split())
    covered = question_words & content_words
    keyword_coverage = round(len(covered) / len(question_words), 3) if question_words else 0.0
    missing_keywords = sorted(question_words - covered)

    # 来源多样性
    source_diversity = len({c.source for c in chunks})

    return RetrievalSignals(
        avg_score=avg_score,
        max_score=max_score,
        min_score=min_score,
        keyword_coverage=keyword_coverage,
        source_diversity=source_diversity,
        chunk_count=len(chunks),
        missing_keywords=missing_keywords,
    )


def _build_prompt(
    question: str,
    retrieval: RetrievalResult,
    signals: RetrievalSignals,
) -> str:
    """
    构造发送给 LLM 的评估 Prompt。

    Prompt 结构：
    - 系统角色说明
    - 量化信号摘要（帮助 LLM 定量锚定）
    - 检索片段全文（帮助 LLM 定性判断）
    - 输出格式要求（严格 JSON）
    """
    chunks_text = "\n\n".join(
        f"[片段 {i + 1}]  来源：{c.source}  相似度得分：{c.score:.4f}\n{c.content}"
        for i, c in enumerate(retrieval.chunks)
    )

    missing_kw_str = (
        "、".join(signals.missing_keywords[:8]) if signals.missing_keywords else "无"
    )

    prompt = f"""你是一名专业的 RAG 检索质量评估专家，需要综合多种信息对检索结果作出客观评判。

═══════════════════════════════════════════════
用户问题
═══════════════════════════════════════════════
{question}

═══════════════════════════════════════════════
量化信号（预先计算，供参考）
═══════════════════════════════════════════════
• 片段总数：{signals.chunk_count}
• 向量相似度 — 最高：{signals.max_score:.4f}｜平均：{signals.avg_score:.4f}｜最低：{signals.min_score:.4f}
• 关键词覆盖率：{signals.keyword_coverage:.0%}（未覆盖词：{missing_kw_str}）
• 来源多样性：{signals.source_diversity} 个不同文件

═══════════════════════════════════════════════
检索到的内容片段
═══════════════════════════════════════════════
{chunks_text}

═══════════════════════════════════════════════
评估要求
═══════════════════════════════════════════════
请从以下四个维度综合评判，并给出最终裁决：

1. 语义相关性（relevance）：片段内容与问题在语义上的匹配程度，0~1。
2. 信息覆盖度（coverage）：片段能覆盖问题所有方面的程度，0~1。
3. 综合置信度（confidence）：基于以上片段能否给出可靠答案，0~1。
4. 裁决（verdict）：
   - "qualified"    → 信息足够，可直接据此作答
   - "insufficient" → 有部分相关信息，但不完整
   - "no_coverage"  → 几乎没有覆盖，无法作答

注意：请结合量化信号与片段内容一起判断，不要仅凭单一维度下结论。

请严格返回如下 JSON，不要包含任何额外文字：
{{
  "relevance": <0到1的浮点数>,
  "coverage": <0到1的浮点数>,
  "confidence": <0到1的浮点数>,
  "verdict": "qualified" | "insufficient" | "no_coverage",
  "conclusion": "<一句话中文结论，说明判断依据>",
  "action_hint": "<给 AI 助手的后续操作建议>"
}}"""

    return prompt


class LLMEvaluator(BaseEvaluator):
    """
    多信号融合 + LLM 综合评估器。

    流程：
      1. 本地预计算量化信号（向量得分分布、关键词覆盖率、来源多样性）
      2. 将信号 + 原始片段一起送入 LLM，由 LLM 做综合判断
      3. 解析 LLM 返回的结构化 JSON，填充 EvaluationResult

    LLM 调用通过 :class:`~rag_module.llm_client.LLMClient` 完成，
    底层使用 ``litellm.acompletion()``，支持本地 vLLM 和任意 OpenAI 兼容端点。
    配置通过 :class:`~rag_module.llm_client.LLMConfig` 从环境变量或 ``.env`` 加载。

    依赖::

        pip install litellm python-dotenv

    Usage（从 .env 自动加载）::

        evaluator = LLMEvaluator()
        result = await evaluator.evaluate(question="...", retrieval=retrieval_result)

    Usage（手动指定配置）::

        from rag_module.llm_client import LLMConfig

        config = LLMConfig(
            api_key="sk-xxxx",
            model="Qwen2.5-7B-Instruct",
            base_url="http://10.x.x.x:18023/v1",
        )
        evaluator = LLMEvaluator(config=config)
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Args:
            config: LLM 配置；为 ``None`` 时调用 :meth:`LLMConfig.from_env` 从
                    环境变量（或 ``.env`` 文件）自动加载。

        Raises:
            ImportError: 若未安装 ``litellm``。
            ValueError:  ``config`` 为 ``None`` 且环境变量中找不到 API Key 时。
        """
        self._client = LLMClient(config=config)

    async def evaluate(
        self,
        question: str,
        retrieval: RetrievalResult,
    ) -> EvaluationResult:
        """
        评估检索结果质量。

        Args:
            question:  用户的原始问题。
            retrieval: 检索器返回的 :class:`~rag_module.base.RetrievalResult`。

        Returns:
            :class:`~rag_module.base.EvaluationResult`
        """
        # --- 无结果快速返回 ---
        if not retrieval.chunks:
            empty_signals = RetrievalSignals(
                avg_score=0.0, max_score=0.0, min_score=0.0,
                keyword_coverage=0.0, source_diversity=0,
                chunk_count=0, missing_keywords=[],
            )
            return EvaluationResult(
                verdict="no_coverage",
                relevance=0.0,
                coverage=0.0,
                confidence=0.0,
                conclusion="知识库中未检索到任何相关内容。",
                action_hint="告知用户该问题超出当前知识库的覆盖范围。",
                signals=empty_signals,
            )

        # --- 预计算量化信号 ---
        signals = _compute_signals(question, retrieval)

        # --- 构造 Prompt 并调用 LLM ---
        prompt = _build_prompt(question, retrieval, signals)

        raw = await self._client.chat_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        # --- 解析 JSON（容错处理） ---
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to extract JSON from a Markdown code block (multi-line safe)
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
            if match:
                try:
                    result = json.loads(match.group(1))
                except json.JSONDecodeError:
                    result = {}
            else:
                # Last resort: find first {...} block
                try:
                    match2 = re.search(r"\{[\s\S]*\}", raw)
                    result = json.loads(match2.group(0)) if match2 else {}
                except json.JSONDecodeError:
                    result = {}

        return EvaluationResult(
            verdict=result.get("verdict", "no_coverage"),
            relevance=float(result.get("relevance", 0.0)),
            coverage=float(result.get("coverage", 0.0)),
            confidence=float(result.get("confidence", 0.0)),
            conclusion=result.get("conclusion", "LLM 未返回结论。"),
            action_hint=result.get("action_hint", ""),
            signals=signals,
        )
