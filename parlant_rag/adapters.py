"""
Parlant 适配器层 — 将 model_client 实现桥接到 Parlant 接口。

核心适配器：
- ModelClientEmbedder: 实现 Parlant Embedder 接口，使用 model_client 的 OpenAILikeEmbedding
- ModelClientStreamingTextGenerator: 实现 Parlant StreamingTextGenerator 接口
- ModelClientSchematicGenerator: 实现 Parlant SchematicGenerator 接口
- ModelClientNLPService: 实现 Parlant NLPService 接口，整合所有适配器

设计模式：适配器模式（Adapter Pattern）
将 model_client 的具体实现适配到 Parlant 的抽象接口，实现架构解耦。
"""
from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Optional, Sequence, TypeVar, cast
from collections.abc import Mapping

from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    SchematicGenerator,
    StreamingTextGenerator,
    StreamingTextGenerationResult,
)
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.service import (
    NLPService,
    SchematicGeneratorHints,
    StreamingTextGeneratorHints,
    EmbedderHints,
)
from parlant.core.nlp.moderation import ModerationService

from model_client.embedding import OpenAILikeEmbedding
from model_client.llm_client import LLMClient


class _DummyTokenizer(EstimatingTokenizer):
    """用于 model_client 的虚拟分词器。"""
    
    def estimate_tokens(self, text: str) -> int:
        return int(len(text) / 4)  # 简单估算


class ModelClientEmbedder(Embedder):
    """
    Parlant Embedder 接口的 model_client 实现。
    
    使用 OpenAILikeEmbedding 进行文本向量化，提供 Parlant 标准接口。
    """
    
    def __init__(self, embedder: Optional[OpenAILikeEmbedding] = None):
        self._embedder = embedder or OpenAILikeEmbedding.from_env()
        self._tokenizer = _DummyTokenizer()
    
    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        """将文本列表转换为向量。"""
        embeddings = await self._embedder._aget_text_embeddings(texts)
        return EmbeddingResult(vectors=embeddings)
    
    @property
    def id(self) -> str:
        return f"model_client_{self._embedder.model_name}"
    
    @property
    def max_tokens(self) -> int:
        return self._embedder.max_seq_len or 8192
    
    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return self._tokenizer
    
    @property
    def dimensions(self) -> int:
        return self._embedder.dimensions


class ModelClientStreamingTextGenerator(StreamingTextGenerator):
    """
    Parlant StreamingTextGenerator 接口的 model_client 实现。
    
    使用 LLMClient 进行流式文本生成，提供 Parlant 标准接口。
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm_client = llm_client or LLMClient.from_env()
    
    @property
    def id(self) -> str:
        return f"model_client_streaming_{self._llm_client.model}"
    
    async def generate(self, prompt: str) -> StreamingTextGenerationResult:
        """生成流式文本。"""
        async def stream():
            async for chunk in self._llm_client.stream_chat_completion(
                model=self._llm_client.model,
                messages=[{"role": "user", "content": prompt}],
            ):
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        return StreamingTextGenerationResult(stream=stream())


T = TypeVar("T")


class ModelClientSchematicGenerator(SchematicGenerator[T], ABC):
    """
    Parlant SchematicGenerator 接口的 model_client 实现。
    
    使用 LLMClient 进行结构化输出生成，提供 Parlant 标准接口。
    """
    
    def __init__(self, output_type: type[T], llm_client: Optional[LLMClient] = None):
        self._output_type = output_type
        self._llm_client = llm_client or LLMClient.from_env()
    
    @property
    def id(self) -> str:
        return f"model_client_schematic_{self._llm_client.model}"
    
    async def generate(self, prompt: str) -> T:
        """生成结构化输出。"""
        # 如果输出类型是 dict，使用 JSON 模式
        if self._output_type == dict:
            system_prompt = "请以 JSON 格式输出结果。"
            response = await self._llm_client.chat_completion(
                model=self._llm_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            import json
            return cast(T, json.loads(response.choices[0].message.content or "{}"))
        
        # 默认返回字符串
        response = await self._llm_client.chat_completion(
            model=self._llm_client.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return cast(T, response.choices[0].message.content)


class ModelClientModerationService(ModerationService):
    """
    Parlant ModerationService 接口的 model_client 实现。
    """
    
    async def classify(self, text: str) -> list[str]:
        """分类文本内容。"""
        return []  # 暂时返回空列表，可根据需要实现
    
    async def requires_moderation(self, text: str) -> bool:
        """判断是否需要审核。"""
        return False


class ModelClientNLPService(NLPService):
    """
    Parlant NLPService 接口的 model_client 实现。
    
    整合所有 model_client 适配器，提供统一的 NLP 服务入口。
    """
    
    def __init__(self):
        self._embedder: Optional[ModelClientEmbedder] = None
        self._streaming_generator: Optional[ModelClientStreamingTextGenerator] = None
        self._schematic_generators: Dict[type, ModelClientSchematicGenerator] = {}
        self._moderation_service: Optional[ModelClientModerationService] = None
    
    @property
    def supports_streaming(self) -> bool:
        return True
    
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> SchematicGenerator[T]:
        """获取结构化输出生成器。"""
        if t not in self._schematic_generators:
            self._schematic_generators[t] = ModelClientSchematicGenerator(t)
        return self._schematic_generators[t]
    
    async def get_streaming_text_generator(
        self, hints: StreamingTextGeneratorHints = {}
    ) -> StreamingTextGenerator:
        """获取流式文本生成器。"""
        if self._streaming_generator is None:
            self._streaming_generator = ModelClientStreamingTextGenerator()
        return self._streaming_generator
    
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        """获取文本嵌入器。"""
        if self._embedder is None:
            self._embedder = ModelClientEmbedder()
        return self._embedder
    
    async def get_moderation_service(self) -> ModerationService:
        """获取内容审核服务。"""
        if self._moderation_service is None:
            self._moderation_service = ModelClientModerationService()
        return self._moderation_service
