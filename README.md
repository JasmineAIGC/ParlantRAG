# ParlantRAG

一个轻量级 RAG 模块，无缝接入 [Parlant](https://github.com/emcie-co/parlant) Agent 框架。  
知识库由 [FlexRAG](https://github.com/Xavier-123/FlexRAG) 的 `scripts/build_knowledge_base.py` 构建。

---

## 核心功能

| 功能 | 说明 |
|---|---|
| **检索（Retrieve）** | 从向量知识库召回与用户问题最相关的文档片段 |
| **评价（Evaluate）** | 对检索结果进行相关性、覆盖度、置信度的多维评估 |
| **零侵入注册** | 通过 `RAGModule.register(agent)` 一行注入任意 Parlant Agent |
| **可插拔组件** | 检索器（Retriever）和评价器（Evaluator）均可自由替换 |

---

## 目录结构

```
rag_module/
├── __init__.py          对外暴露的公共接口
├── base.py              抽象接口定义（BaseRetriever / BaseEvaluator / 数据类）
├── retriever.py         内置检索器实现
│                          - KeywordRetriever   关键词检索，无需额外依赖
│                          - DenseRetriever     sentence-transformers + FAISS
│                          - FlexRAGRetriever   加载 FlexRAG 构建的知识库
├── evaluator.py         内置评价器实现
│                          - RuleBasedEvaluator 规则评价，无需 LLM
│                          - LLMEvaluator       OpenAI 兼容 LLM 评价
├── tools.py             Parlant 工具定义（rag_search / evaluate_retrieval）
├── module.py            RAGModule 主类（注册入口）
│
└── examples/
    ├── basic_usage.py   最简使用示例（KeywordRetriever）
    └── custom_usage.py  FlexRAGRetriever / LLMEvaluator / 自定义实现示例
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 构建知识库（使用 FlexRAG）

```bash
# 克隆 FlexRAG
git clone https://github.com/Xavier-123/FlexRAG
cd FlexRAG

# 构建知识库
python scripts/build_knowledge_base.py \
    --input-dir ./my_docs \
    --output-dir ./my_knowledge_base \
    --embedding-base-url http://127.0.0.1:8018/v1 \
    --embedding-model Qwen3-Embedding-0.6B
```

### 3. 接入 Parlant Agent

```python
import asyncio
import parlant.sdk as p
from rag_module import RAGModule
from rag_module.retriever import FlexRAGRetriever

async def main():
    retriever = FlexRAGRetriever(
        persist_dir="./my_knowledge_base",
        embedding_base_url="http://127.0.0.1:8018/v1",
        embedding_api_key="sk-xxxx",
        embedding_model="Qwen3-Embedding-0.6B",
    )

    async with p.Client() as client:
        agent = await client.agents.create(name="知识助手")

        rag = RAGModule(retriever=retriever)
        await rag.register(agent)

asyncio.run(main())
```

---

## 内置检索器

### KeywordRetriever（无需额外依赖）

```python
from rag_module import RAGModule

docs = [
    {"id": "1", "content": "Parlant 是一个开源 AI Agent 框架。", "source": "intro.md"},
    {"id": "2", "content": "Guideline 由 condition 和 action 组成。", "source": "guide.md"},
]
rag = RAGModule(documents=docs)
```

### FlexRAGRetriever（推荐，与 FlexRAG 配套）

```python
from rag_module.retriever import FlexRAGRetriever

retriever = FlexRAGRetriever(
    persist_dir="./my_knowledge_base",  # FlexRAG 输出目录
    embedding_base_url="http://127.0.0.1:8018/v1",
    embedding_api_key="sk-xxxx",
    embedding_model="Qwen3-Embedding-0.6B",
    top_k=5,
    bm25_weight=0.3,   # 0.0 = 纯向量, 0.3 = 30% BM25 混合（需要 bm25_index/）
)
```

### DenseRetriever（本地 sentence-transformers）

```python
from rag_module.retriever import DenseRetriever

retriever = DenseRetriever(
    documents=docs,
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
)
await retriever.build_index()
```

---

## 内置评价器

### RuleBasedEvaluator（无需 LLM，速度快）

```python
from rag_module.evaluator import RuleBasedEvaluator

evaluator = RuleBasedEvaluator(
    qualified_threshold=0.45,     # 置信度 >= 0.45 → qualified
    insufficient_threshold=0.20,  # 置信度 >= 0.20 → insufficient
)
```

### LLMEvaluator（语义级评价，精度更高）

```python
from rag_module.evaluator import LLMEvaluator

evaluator = LLMEvaluator(
    api_key="sk-...",
    model="gpt-4o-mini",
    base_url="https://your-openai-compatible-endpoint/v1",  # 可选
)
```

---

## 自定义检索器 / 评价器

继承 `BaseRetriever` 或 `BaseEvaluator` 并实现对应方法即可：

```python
from rag_module import BaseRetriever, RetrievalResult, Chunk

class MyRetriever(BaseRetriever):
    async def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        # 调用你自己的检索逻辑（Chroma / Pinecone / Weaviate 等）
        chunks = [Chunk(chunk_id="1", content="...", source="file.md", score=0.9)]
        return RetrievalResult(query=query, chunks=chunks, status="success")

rag = RAGModule(retriever=MyRetriever())
```

---

## 直接调用（无需 Parlant 服务）

```python
import asyncio
from rag_module import KeywordRetriever, RuleBasedEvaluator

async def test():
    retriever = KeywordRetriever([
        {"id": "1", "content": "Parlant 是一个开源框架。", "source": "doc.md"},
    ])
    evaluator = RuleBasedEvaluator()

    result = await retriever.retrieve("Parlant 框架")
    eval_result = await evaluator.evaluate("Parlant 框架是什么？", result)

    print(eval_result.verdict)    # "qualified" / "insufficient" / "no_coverage"
    print(eval_result.conclusion)

asyncio.run(test())
```

更多示例请参考 [`rag_module/examples/`](rag_module/examples/)。

---

## 架构设计

```
宿主系统 (Parlant Agent)
    │
    └── RAGModule.register(agent)
            │
            ├── rag_search 工具         ← BaseRetriever.retrieve()
            │       └── 返回 retrieved_chunks
            │
            └── evaluate_retrieval 工具 ← BaseEvaluator.evaluate()
                    └── 返回 verdict / scores / conclusion

Guidelines（自动注入）：
  1. 用户提问 → 调用 rag_search
  2. 检索完成 → 调用 evaluate_retrieval
  3. verdict=qualified   → 基于检索内容作答
  4. verdict=insufficient → 告知相关性不足
  5. verdict=no_coverage  → 告知超出知识库范围
```
