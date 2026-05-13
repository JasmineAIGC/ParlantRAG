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
parlant_rag/                 主包——所有核心逻辑
├── __init__.py              对外公共接口（快速导入入口）
│
├── retrieval/               ── 检索模块 ──────────────────────────────
│   ├── __init__.py
│   ├── types.py             共享数据类（Chunk、RetrievalResult）
│   ├── keyword.py           KeywordRetriever（无依赖，关键词 Jaccard 匹配）
│   ├── dense.py             DenseRetriever（sentence-transformers + FAISS）
│   └── flexrag.py           FlexRAGRetriever（加载 FlexRAG 构建的知识库）
│
├── evaluation/              ── 评估模块 ──────────────────────────────
│   ├── __init__.py
│   └── evaluator.py         LLMEvaluator（多信号预计算 + LLM 综合评判）
│
└── tools/                   ── 工具类（Parlant 集成层） ───────────────
    ├── __init__.py
    ├── parlant_tools.py     Parlant 工具定义（rag_search / evaluate_retrieval）
    └── module.py            RAGModule 主类（注册入口）

model_client/                模型调用基础层（LLM + Embedding 客户端）
├── api/                     通过 OpenAI 兼容 API 调用远程模型
│   ├── llm.py               LLMClient、LLMConfig
│   └── embedding.py         OpenAILikeEmbedding
└── local/                   本地模型（无需 API 服务）
    └── embedding.py         SentenceTransformerEmbedding

examples/                    独立运行示例（无需启动 Parlant 服务）
├── 00_check_env.py          验证 .env 配置（LLM / Embedding API 连通性）
├── 01_load_kb.py            验证 FAISS 知识库加载
├── 02_retrieval.py          检索流程演示
└── 03_retrieval_eval.py     检索 + LLM 评估完整流程
```

---

## 配置说明（.env）

所有 API 和路径配置统一来自项目根目录的 `.env` 文件。  
复制模板并填写实际值：

```bash
cp .env.example .env
```

关键配置项：

```ini
# LLM 服务（用于 LLMEvaluator）
QWEN3_API_KEY=sk-xxxx
QWEN3_BASE_URL=http://10.x.x.x:18023/v1
QWEN3_MODEL=Qwen2.5-7B-Instruct

# Embedding 服务（用于 FlexRAGRetriever，须与构建知识库时一致）
EMBEDDING_BASE_URL=http://10.x.x.x:18024/v1
EMBEDDING_API_KEY=sk-xxxx
EMBEDDING_MODEL=Qwen3-Embedding-8B

# 知识库路径（FlexRAG 构建的 FAISS 索引目录）
KNOWLEDGE_PERSIST_DIR=./knowledge_base/knowledge_persist_dir/hotpotqa
```

> **本地 Embedding 模式**（无 API 服务时）：注释掉 `EMBEDDING_BASE_URL`，改为设置：
> ```ini
> EMBEDDING_MODEL_PATH=/path/to/models/BAAI/bge-large-zh-v1.5
> ```

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
    --embedding-base-url http://10.x.x.x:18024/v1 \
    --embedding-model Qwen3-Embedding-8B
```

### 3. 验证环境配置

```bash
python examples/00_check_env.py
```

### 4. 独立运行检索 + 评估

```bash
# 仅检索
python examples/02_retrieval.py

# 检索 + LLM 评估
python examples/03_retrieval_eval.py
```

### 5. 接入 Parlant Agent

```python
from parlant_rag import RAGModule, LLMEvaluator
from parlant_rag.retrieval import FlexRAGRetriever

# 所有配置从根目录 .env 自动加载
retriever = FlexRAGRetriever.from_env()
evaluator = LLMEvaluator()

rag = RAGModule(retriever=retriever, evaluator=evaluator)
await rag.register(agent)
```

---

## 模块详解

### 检索模块（`parlant_rag.retrieval`）

| 类 | 文件 | 依赖 | 适用场景 |
|---|---|---|---|
| `KeywordRetriever` | `retrieval/keyword.py` | 无 | 快速原型、小规模文档 |
| `DenseRetriever` | `retrieval/dense.py` | `sentence-transformers faiss-cpu` | 本地向量检索 |
| `FlexRAGRetriever` | `retrieval/flexrag.py` | `llama-index faiss-cpu` | 与 FlexRAG 知识库配套 |

### 评估模块（`parlant_rag.evaluation`）

| 类 | 文件 | 依赖 | 特点 |
|---|---|---|---|
| `LLMEvaluator` | `evaluation/evaluator.py` | `litellm openai` | 多信号 + LLM 综合评判 |

评估流程：

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

### 工具类（`parlant_rag.tools`）

需安装 `parlant`：

```bash
pip install "parlant-rag[parlant]"
```

注入后 Agent 执行流程：

```
用户提问
  │
  ▼ rag_search（向量检索）
  │
  ▼ evaluate_retrieval（多信号 LLM 评估）
  │
  ├── verdict=qualified    → 基于检索内容作答，注明来源
  ├── verdict=insufficient → 告知相关性不足，建议换词重问
  └── verdict=no_coverage  → 告知超出知识库范围，不凭自身知识作答
```

---

## 数据类

```python
@dataclass
class Chunk:
    chunk_id: str
    content:  str
    source:   str
    score:    float = 0.0
    metadata: dict  = field(default_factory=dict)

@dataclass
class RetrievalResult:
    query:  str
    chunks: list[Chunk]
    status: str          # "success" | "no_results"

@dataclass
class EvaluationResult:
    verdict:    str      # "qualified" | "insufficient" | "no_coverage"
    relevance:  float    # 0~1
    coverage:   float    # 0~1
    confidence: float    # 0~1
    conclusion: str
    action_hint: str
    signals:    Optional[RetrievalSignals] = None
```
