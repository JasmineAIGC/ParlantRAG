# ParlantRAG

一个基于 **Plan A** 架构的轻量 RAG 项目，核心模块为 `plan_a/`，并提供可独立运行的 `examples/`。

## 当前代码结构

```text
plan_a/                      Plan A RAG 主模块（独立于 Agent）
model_client/                LLM / Embedding 客户端
examples/                    环境检查、检索、评估、Agent 集成示例
knowledge_base/              示例知识库与索引
utils/                       日志等通用工具
```

## Plan A 架构说明

详见：`docs/planA.md`

- `plan_a.rag.RAGPipeline`：RAG 检索管道（Embedding + VectorStore）
- `plan_a.retrievers`：Scoped Retriever（Guideline 触发）
- `plan_a.tools`：RAG Tool（Agent 主动调用）
- `plan_a.guidelines.setup_guidelines()`：将 RAG 能力挂载到 Parlant Agent

## 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 配置环境变量

```bash
cp .env.example .env
```

至少需要配置：

- `KNOWLEDGE_PERSIST_DIR`（或 `KNOWLEDGE_BASE_PATH`）
- `EMBEDDING_MODEL_PATH` **或** `EMBEDDING_BASE_URL` + `EMBEDDING_MODEL`
- 进行评估/问答时配置 `QWEN3_*`（或 `LLM_*`）

### 3) 运行示例

```bash
# 环境连通性检查
python examples/00_check_env.py

# 知识库加载检查
python examples/01_load_knowledge_base.py

# 检索测试
python examples/02_test_retrieval.py

# 检索 + LLM 评估
python examples/03_test_retrieval_eval.py

# 完整演示：独立 RAG 问答 + Parlant Agent 集成
python examples/05_plan_a_agent.py
```

## knowledge_base 说明

`plan_a.vdb.VectorStore` 兼容以下索引/文档格式：

- `faiss_index.bin` 或 `faiss_index`
- `document_store.json` 或 `docstore.json`
- 支持从 `knowledge_base/<dataset>.json` 回填文档内容（当持久化 docstore 为空时）

## 注意事项

- `plan_a/` 是独立模块，不依赖旧版实现模块。
- `third_party/` 为只读第三方代码，不应在业务修复中修改。
