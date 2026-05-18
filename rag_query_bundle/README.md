# rag_query_bundle — Parlant RAG 单条查询快速入门

本目录是 `examples/06_rag_query.py` 的最小化运行包，包含：

| 文件 | 说明 |
|---|---|
| `run_rag_query.py` | 查询入口脚本 |
| `.env.example` | 环境变量配置模板 |
| `README.md` | 本文档 |

---

## 运行时依赖

### Python 包（从仓库根目录安装）

```bash
pip install -r requirements.txt
# 若需使用 Parlant Server 模式，还需：
pip install parlant
```

> Parlant 3.x 的 LiteLLM 适配器在导入时依赖 `torch`，如遇 `No module named 'torch'` 报错，执行：
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 代码包（仓库内）

`run_rag_query.py` 会自动将仓库根目录加入 `sys.path`，以下包无需单独安装，直接使用仓库源码：

- `plan_a/` — RAG 核心（Embedding、VectorStore、Pipeline、Parlant Tools）
- `model_client/` — LLM 客户端封装
- `utils/` — 日志工具

---

## 配置（`.env`）

复制模板并填写实际值：

```bash
cp rag_query_bundle/.env.example .env   # 在仓库根目录操作
```

必填项：

| 变量 | 说明 |
|---|---|
| `QWEN3_API_KEY` | LLM API 密钥 |
| `QWEN3_BASE_URL` | LLM 服务地址（OpenAI 兼容端点） |
| `QWEN3_MODEL` | LLM 模型名称 |
| `EMBEDDING_BASE_URL` | Embedding 服务地址 |
| `EMBEDDING_API_KEY` | Embedding API 密钥 |
| `EMBEDDING_MODEL` | Embedding 模型名称 |
| `KNOWLEDGE_PERSIST_DIR` | FAISS 索引目录路径（含 `faiss_index.bin` 和 `docstore.json`） |

---

## 前置条件：构建知识库索引

运行前须先构建 FAISS 向量索引：

```bash
python examples/01_load_knowledge_base.py
```

---

## 运行方式

在仓库根目录执行：

```bash
# 默认查询，文本输出
python rag_query_bundle/run_rag_query.py

# 自定义查询
python rag_query_bundle/run_rag_query.py --query "退货政策是什么？"

# 指定检索片段数
python rag_query_bundle/run_rag_query.py --query "..." --top-k 5

# JSON 格式输出
python rag_query_bundle/run_rag_query.py --query "..." --output-format json

# 保存到文件
python rag_query_bundle/run_rag_query.py --query "..." --output-file result.json
```

### CLI 参数一览

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--query` | str | 内置示例问题 | 检索查询文本 |
| `--top-k` | int | None（使用 pipeline 默认值） | 最多返回的文档片段数 |
| `--include-metadata` | flag | False | 在 `chunks` 中附加每个片段的原始元数据 |
| `--output-format` | text \| json | text | 终端输出格式 |
| `--output-file` | PATH | None | 保存 JSON 结果的文件路径；指定时自动使用 JSON 格式 |

---

## 输出格式

### 文本模式（`--output-format text`，默认）

```
✅ 查询完成  chunks=3
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
  "answer": "根据知识库内容，本店支持 30 天无理由退货……"
}
```

### 字段说明

#### 顶层字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `query` | string | 用户输入的查询文本（即 `--query` 参数值） |
| `retrieval` | object | 向量检索结果 |
| `answer` | string | LLM 根据检索内容生成的回答；若知识库未命中则为固定提示语 |

#### `retrieval` 对象

| 字段 | 类型 | 说明 |
|---|---|---|
| `top_k` | int | 本次检索使用的片段上限（来自 `--top-k` 或 pipeline 配置） |
| `chunk_count` | int | 实际命中并返回的片段数（`<= top_k`） |
| `took_ms` | int | 向量检索耗时（毫秒，不含 Embedding 时间） |
| `chunks` | array | 检索到的文档片段列表，按相似度降序排列 |

#### `chunks[i]` 对象

| 字段 | 类型 | 说明 |
|---|---|---|
| `content` | string | 文档片段正文 |
| `source` | string | 片段来源标识（文件名、文档 ID 等） |
| `score` | float | 向量相似度分数（范围通常为 0～1，越高越相关） |
| `metadata` | object | 原始文档元数据，**仅在 `--include-metadata` 时出现**（字段因知识库而异） |

---

## RAG 流程简述

```
用户查询
  │
  ▼ Parlant Server（后台启动，不弹 UI）
Agent + Guidelines（Scoped Retriever + RAG Tool）
  │
  ▼ RAGPipeline.retrieve_standard()
Embedding → FAISS 向量搜索 → List[DocChunk]
  │
  ▼ LLMClient.chat_async()
回答文本
```

完整架构说明见 `docs/planA.md`。
