# rag_query_bundle — Parlant RAG 单条查询快速入门

本目录是**自包含**的独立运行包，无需安装整个仓库。目录结构：

```
rag_query_bundle/
├── run_rag_query.py   # 查询入口脚本
├── .env.example       # 环境变量配置模板
├── README.md          # 本文档
├── plan_a/            # RAG 核心（Embedding、VectorStore、Pipeline、Parlant Tools）
├── model_client/      # LLM 客户端封装（--answer 时使用）
└── utils/             # 日志工具
```

---

## 运行时依赖

### Python 包

```bash
pip install parlant litellm python-dotenv openai httpx \
            llama-index llama-index-vector-stores-faiss faiss-cpu
```

> Parlant 3.x 的 LiteLLM 适配器在导入时依赖 `torch`，如遇 `No module named 'torch'` 报错，执行：
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

## 配置（`.env`）

在 `rag_query_bundle/` 目录内新建 `.env` 文件（复制模板）：

```bash
cd rag_query_bundle
cp .env.example .env
# 然后编辑 .env，填写实际值
```

必填项：

| 变量 | 说明 |
|---|---|
| `QWEN3_API_KEY` | LLM API 密钥（Parlant LiteLLM 服务必填） |
| `QWEN3_BASE_URL` | LLM 服务地址（OpenAI 兼容端点） |
| `QWEN3_MODEL` | LLM 模型名称 |
| `EMBEDDING_BASE_URL` | Embedding 服务地址 |
| `EMBEDDING_API_KEY` | Embedding API 密钥 |
| `EMBEDDING_MODEL` | Embedding 模型名称 |
| `KNOWLEDGE_PERSIST_DIR` | FAISS 索引目录路径（含 `faiss_index.bin` 和 `docstore.json`） |

> `QWEN3_*` / `LLM_*` 等 LLM 配置在 Parlant Server 初始化时必填；`--answer` 时同样用于生成回答。

---

## 前置条件：构建知识库索引

运行前须先构建 FAISS 向量索引。从仓库根目录执行：

```bash
python examples/01_load_knowledge_base.py
```

---

## 运行方式

在 `rag_query_bundle/` 目录下执行：

```bash
cd rag_query_bundle

# 默认查询，文本输出（仅检索，不调 LLM 生成回答）
python run_rag_query.py

# 自定义查询
python run_rag_query.py --query "退货政策是什么？"

# 指定检索片段数
python run_rag_query.py --query "..." --top-k 5

# JSON 格式输出
python run_rag_query.py --query "..." --output-format json

# 保存到文件
python run_rag_query.py --query "..." --output-file result.json

# 检索后额外调 LLM 生成回答（后处理阶段）
python run_rag_query.py --query "..." --answer --output-file result.json
```

### CLI 参数一览

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--query` | str | 内置示例问题 | 检索查询文本 |
| `--top-k` | int | None（使用 pipeline 默认值） | 最多返回的文档片段数 |
| `--include-metadata` | flag | False | 在 `chunks` 中附加每个片段的原始元数据 |
| `--answer` | flag | False | 后处理阶段调用 LLM 生成回答（默认关闭） |
| `--output-format` | text \| json | text | 终端输出格式 |
| `--output-file` | PATH | None | 保存 JSON 结果的文件路径；指定时自动使用 JSON 格式 |

---

## 输出格式

### 文本模式（`--output-format text`，默认）

```
⏱  初始化完成  took=29000ms
⏱  RAG 检索完成  took=85ms  chunks=3
⏱  后处理完成   took=0ms  |  总耗时=29085ms
✅ 查询完成  chunks=3
```

> **初始化时间说明**：初始化阶段含 Parlant Server 启动、LiteLLM 服务初始化、知识库实体 Embedding 缓存，这三步为**一次性操作**，服务存续期间不会重复，与后续每次检索无关。

加 `--answer` 后后处理阶段会额外打印回答：

```
   回答：根据知识库内容，……（最多显示 300 字符）
```

### JSON 模式（`--output-format json` 或 `--output-file`）

```json
{
  "query": "退货政策是什么？",
  "retrieval": {
    "top_k": 5,
    "chunk_count": 3,
    "took_ms": 42,
    "chunks": [
      {
        "content": "本店支持 30 天无理由退货……",
        "source": "policy_faq.txt",
        "score": 0.9123
      }
    ]
  },
  "answer": "",
  "timing": {
    "init_ms": 29000,
    "rag_ms": 85,
    "postprocess_ms": 0,
    "total_ms": 29085
  }
}
```

> 加 `--answer` 时 `postprocess_ms` 含 LLM 调用耗时，`answer` 字段有内容。

### 字段说明

#### 顶层字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `query` | string | 用户输入的查询文本（即 `--query` 参数值） |
| `retrieval` | object | 向量检索结果 |
| `answer` | string | 后处理阶段 LLM 生成的回答（`--answer` 时有内容，否则为空字符串） |
| `timing` | object | 各阶段耗时统计（毫秒） |

#### `retrieval` 对象

| 字段 | 类型 | 说明 |
|---|---|---|
| `top_k` | int | 本次检索使用的片段上限（来自 `--top-k` 或 pipeline 配置） |
| `chunk_count` | int | 实际命中并返回的片段数（`<= top_k`） |
| `took_ms` | int | RAG Tool 内部向量检索耗时（毫秒，不含 LLM 调用） |
| `chunks` | array | 检索到的文档片段列表，按相似度降序排列 |

#### `chunks[i]` 对象

| 字段 | 类型 | 说明 |
|---|---|---|
| `content` | string | 文档片段正文 |
| `source` | string | 片段来源标识（文件名、文档 ID 等） |
| `score` | float | 向量相似度分数（范围通常为 0～1，越高越相关） |
| `metadata` | object | 原始文档元数据，**仅在 `--include-metadata` 时出现**（字段因知识库而异） |

#### `timing` 对象

| 字段 | 类型 | 说明 |
|---|---|---|
| `init_ms` | int | 初始化耗时（毫秒）：Parlant Server 启动 + LiteLLM 服务初始化 + 知识库实体 Embedding 缓存（**一次性**） |
| `rag_ms` | int | RAG 检索耗时（毫秒）：Agent RAG Tool 向量检索；**不含 LLM 调用** |
| `postprocess_ms` | int | 后处理耗时（毫秒）：无 `--answer` 时 < 1ms；有 `--answer` 时含 LLM 调用时间 |
| `total_ms` | int | 总耗时（毫秒），等于 init + rag + postprocess 之和（含极小的调度开销） |

---

## RAG 流程简述

```
用户查询
  │
  ▼ 阶段 1：初始化（一次性，~20-30s）
Parlant Server 启动 → LiteLLM 服务初始化 → 知识库实体 Embedding 缓存
  │
  ▼ 阶段 2：RAG 检索（<100ms）
Agent RAG Tool → Query Embedding → FAISS 向量搜索 → List[DocChunk]
  │
  ▼ 阶段 3：后处理（可选，--answer）
LLMClient.chat_async() → 回答文本
组装 JSON 输出
```

完整架构说明见 `docs/planA.md`。
