# ParlantRAG

一个轻量级 RAG 模块，无缝接入 [Parlant](https://github.com/emcie-co/parlant) Agent 框架。  
知识库由 [FlexRAG](https://github.com/Xavier-123/FlexRAG) 的 `scripts/build_knowledge_base.py` 构建。

---

## 核心功能

| 功能 | 说明 |
|---|---|
| **检索（Retrieve）** | 从向量知识库召回与用户问题最相关的文档片段 |
| **评估（Evaluate）** | 预计算多维量化信号，结合 LLM 对检索结果做综合评判 |
| **零侵入注册** | 通过 `RAGModule.register(agent)` 一行注入任意 Parlant Agent |
| **可插拔组件** | 检索器（Retriever）和评估器（Evaluator）均可自由替换 |

---

## 目录结构

```
rag_module/
├── __init__.py              对外暴露的公共接口
├── base.py                  共享数据类 + 抽象接口（检索/评估两包的唯一耦合点）
├── tools.py                 Parlant 工具定义（rag_search / evaluate_retrieval）
├── module.py                RAGModule 主类（注册入口）
│
├── retrieval/               ── 检索子包 ──────────────────────────────
│   ├── __init__.py
│   ├── keyword.py           KeywordRetriever（无依赖，关键词 Jaccard 匹配）
│   ├── dense.py             DenseRetriever（sentence-transformers + FAISS）
│   └── flexrag.py           FlexRAGRetriever（加载 FlexRAG 构建的知识库）
│
└── evaluation/              ── 评估子包 ──────────────────────────────
    ├── __init__.py
    └── llm_evaluator.py     LLMEvaluator（多信号预计算 + LLM 综合评判）

examples/                    顶层示例
├── basic_usage.py           最简 Parlant 集成（KeywordRetriever + LLMEvaluator）
├── flexrag_usage.py         FlexRAG 知识库 + 完整 Parlant 集成
└── standalone_test.py       独立测试（不需要启动 Parlant 服务）
```

---

## 评估器设计（`LLMEvaluator`）

评估不使用纯规则，而是**先本地预计算量化信号，再由 LLM 综合判断**：

```
步骤 1：本地预计算（无 LLM，极低开销）
  ├── 向量相似度分布（max / avg / min）
  ├── 问题关键词覆盖率 & 未覆盖词列表
  └── 来源多样性（不同文件数量）

步骤 2：构造富文本 Prompt
  ├── 量化信号摘要（锚定数值）
  └── 检索片段全文（定性判断）

步骤 3：LLM 综合评判 → 返回结构化 JSON
  ├── relevance / coverage / confidence（0~1）
  ├── verdict（qualified / insufficient / no_coverage）
  ├── conclusion（一句话结论）
  └── action_hint（给 Agent 的建议）
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 构建知识库（使用 FlexRAG）

```bash
git clone https://github.com/Xavier-123/FlexRAG && cd FlexRAG
python scripts/build_knowledge_base.py \
    --input-dir ./my_docs \
    --output-dir ./my_knowledge_base \
    --embedding-base-url http://127.0.0.1:8018/v1 \
    --embedding-model Qwen3-Embedding-0.6B
```

### 3. 接入 Parlant Agent

```python
from rag_module import RAGModule, LLMEvaluator
from rag_module.retrieval import FlexRAGRetriever

retriever = FlexRAGRetriever(
    persist_dir="./my_knowledge_base",
    embedding_base_url="http://127.0.0.1:8018/v1",
    embedding_api_key="sk-xxxx",
    embedding_model="Qwen3-Embedding-0.6B",
    top_k=5,
    bm25_weight=0.3,   # 70% 向量 + 30% BM25 混合（可选）
)
# LLM 配置从 .env 文件自动加载（参考 .env.example）
evaluator = LLMEvaluator()

rag = RAGModule(retriever=retriever, evaluator=evaluator)
await rag.register(agent)
```

### 4. 运行独立测试（不需要 Parlant 服务）

```bash
# 复制并填写 .env.example → .env
cp .env.example .env
PYTHONPATH=. python examples/standalone_test.py
```

---

## 检索器

| 类 | 文件 | 依赖 | 适用场景 |
|---|---|---|---|
| `KeywordRetriever` | `retrieval/keyword.py` | 无 | 快速原型、小规模文档 |
| `DenseRetriever` | `retrieval/dense.py` | `sentence-transformers faiss-cpu` | 本地向量检索 |
| `FlexRAGRetriever` | `retrieval/flexrag.py` | `llama-index faiss-cpu httpx` | 与 FlexRAG 配套 |

## 评估器

| 类 | 文件 | 依赖 | 特点 |
|---|---|---|---|
| `LLMEvaluator` | `evaluation/llm_evaluator.py` | `openai` | 多信号 + LLM 综合评判 |

---

## 自定义检索器

继承 `BaseRetriever` 并实现 `retrieve()` 方法：

```python
from rag_module.base import BaseRetriever, RetrievalResult, Chunk

class MyRetriever(BaseRetriever):
    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        chunks = [Chunk(chunk_id="1", content="...", source="file.md", score=0.9)]
        return RetrievalResult(query=query, chunks=chunks, status="success")

rag = RAGModule(retriever=MyRetriever(), evaluator=evaluator)
```

---

## Guideline 流程

注入 Agent 后，驱动如下执行流程：

```
用户提问
  │
  ▼ rag_search（向量检索）
  │
  ▼ evaluate_retrieval（多信号 LLM 评估）
  │
  ├── verdict=qualified   → 基于检索内容作答，注明来源
  ├── verdict=insufficient → 告知相关性不足，建议换词重问
  └── verdict=no_coverage  → 告知超出知识库范围，不凭自身知识作答
```

