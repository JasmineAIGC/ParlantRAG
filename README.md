# ParlantRAG

一个轻量级 RAG 模块，已从 FlexRAG 迁移为基于 [Parlant](https://github.com/emcie-co/parlant) 官方 SDK 的实现。新代码主入口为 `rag_module`；旧的 `parlant_rag` 路径保留为兼容转发层。

---

## 核心变化

| 目标 | 实现 |
|---|---|
| 创建新检索器 | `rag_module.retrieval.ParlantRetriever`，可包装关键词 / 向量检索器，并可导出 Parlant `@p.tool` 工具 |
| 更新评估器 | `rag_module.evaluation.LLMEvaluator` 保持既有 `model_client.api.llm.LLMClient` 调用方式，并提供 Parlant SDK 工具接口 |
| 新建模块路径 | 推荐使用 `rag_module`；`parlant_rag` 仅做向后兼容 |
| 简化依赖 | 移除 FlexRAG / llama-index 硬依赖；Embedding API 与本地模型调用方式保持不变 |

---

## 目录结构

```text
rag_module/                  新主包——Parlant SDK 原生 RAG 实现
├── __init__.py              对外公共接口
├── retrieval/
│   ├── types.py             Chunk、RetrievalResult
│   ├── keyword.py           KeywordRetriever（无外部检索依赖）
│   ├── dense.py             DenseRetriever（FAISS + 既有 embedding 客户端）
│   └── parlant.py           ParlantRetriever（检索器封装 + Parlant tool）
├── evaluation/
│   └── evaluator.py         LLMEvaluator（多信号 + 既有 LLMClient）
└── tools/
    ├── parlant_tools.py     make_rag_tools
    └── module.py            RAGModule 注册入口

parlant_rag/                 旧导入路径兼容层（转发到 rag_module）
model_client/                既有模型调用基础层（LLM + Embedding 客户端）
examples/                    独立运行示例
```

---

## 安装依赖

```bash
pip install -r requirements.txt
```

可选：如果使用本地 sentence-transformers embedding：

```bash
pip install sentence-transformers
```

---

## 配置说明（.env）

LLM 与 Embedding API 调用保持原有方式，仍由 `model_client` 读取环境变量：

```ini
# LLM 服务（用于 LLMEvaluator）
QWEN3_API_KEY=sk-xxxx
QWEN3_BASE_URL=http://10.x.x.x:18023/v1
QWEN3_MODEL=Qwen2.5-7B-Instruct

# Embedding 服务（用于 DenseRetriever.from_env / ParlantRetriever.from_dense_env）
EMBEDDING_BASE_URL=http://10.x.x.x:18024/v1
EMBEDDING_API_KEY=sk-xxxx
EMBEDDING_MODEL=Qwen3-Embedding-8B

# 本地 Embedding 模式（可选）
EMBEDDING_MODEL_PATH=/path/to/models/BAAI/bge-large-zh-v1.5
```

---

## 快速开始

### 1. 关键词检索（无向量依赖）

```python
from rag_module.retrieval import ParlantRetriever

retriever = ParlantRetriever.from_keyword([
    {"id": "doc-1", "content": "Parlant uses tools and guidelines.", "source": "demo.md"},
])
result = await retriever.retrieve("Parlant tools", top_k=5)
```

### 2. 向量检索（保持现有 Embedding API 调用）

```python
from rag_module.retrieval import ParlantRetriever

retriever = ParlantRetriever.from_dense_env(documents)
await retriever.build_index()
result = await retriever.retrieve("查询词", top_k=5)
```

### 3. 检索 + LLM 评估

```python
from rag_module.evaluation import LLMEvaluator

retrieval = await retriever.retrieve("查询词", top_k=5)
evaluation = await LLMEvaluator().evaluate("查询词", retrieval)
```

### 4. 接入 Parlant Agent

```python
from rag_module import LLMEvaluator, RAGModule
from rag_module.retrieval import ParlantRetriever

retriever = ParlantRetriever.from_keyword(documents)
rag = RAGModule(retriever=retriever, evaluator=LLMEvaluator())
await rag.register(agent)
```

`ParlantRetriever.as_tool()` 和 `LLMEvaluator.as_tool()` 也可单独生成 Parlant SDK `@p.tool` 工具。

---

## 运行示例

```bash
python examples/00_check_env.py
python examples/02_retrieval.py
python examples/03_retrieval_eval.py
```

`examples/03_retrieval_eval.py` 会实际调用 LLM API，需要 `.env` 中的 LLM 配置可用。

---

## 数据类

```python
@dataclass
class Chunk:
    chunk_id: str
    content: str
    source: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)

@dataclass
class RetrievalResult:
    query: str
    chunks: list[Chunk]
    status: str  # "success" | "no_results"
```
