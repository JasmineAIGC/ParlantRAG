# ParlantRAG

一个基于 **Plan A** 架构的轻量 RAG 项目，核心模块为 `plan_a/`，并提供可独立运行的 `examples/`。

## 当前代码结构

```text
plan_a/                      Plan A RAG 主模块（独立于 Agent）
  core/                      共享配置与类型（PlanAConfig, RAGMode, DocChunk …）
  rag/                       Embedding + RAGPipeline
  vdb/                       VectorStore（FAISS）
  retrievers/                Scoped Retriever（Parlant Guideline 自动触发）
  tools/                     RAG Tool（Parlant @p.tool 手动调用）
  guidelines/                setup_guidelines()：将 RAG 挂载到 Parlant Agent
model_client/                LLM / Embedding 客户端
examples/                    环境检查、检索、评估、Agent 集成示例
knowledge_base/              示例知识库与索引
utils/                       日志等通用工具
```

## Plan A 架构说明

详见：`docs/planA.md`

- `plan_a.rag.RAGPipeline`：RAG 检索管道（Embedding + VectorStore）
- `plan_a.retrievers`：Scoped Retriever（Guideline 触发，自动检索）
- `plan_a.tools`：RAG Tool（Agent 主动调用）
- `plan_a.guidelines.setup_guidelines()`：将 RAG 能力挂载到 Parlant Agent

## RAG 实现方式（RAGMode）

`plan_a` 支持两种 RAG 模式，通过 `RAGMode` 枚举选择：

| 模式 | 说明 | 入口 |
|------|------|------|
| `standalone` | 不依赖 Parlant，`RAGPipeline` 检索 + LLM 回答 | `05_plan_a_agent.py --mode standalone` |
| `parlant`    | Parlant RAG Tool 检索 + LLM 回答（**后台执行，无需 Server / UI**） | `05_plan_a_agent.py --mode parlant` |
| Parlant Server | 完整 Server + Agent + GL + Tool（生产部署 / 集成测试） | `06_rag_query.py` |

## 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
pip install parlant torch  # 运行 Parlant 模式时需要
```

### 2) 配置环境变量

```bash
cp .env.example .env
```

至少需要配置：

- `KNOWLEDGE_PERSIST_DIR`（或 `KNOWLEDGE_BASE_PATH`）
- `EMBEDDING_MODEL_PATH` **或** `EMBEDDING_BASE_URL` + `EMBEDDING_MODEL`
- 进行评估/问答时配置 `QWEN3_*`（或 `LLM_*`）
- Parlant Server 模式（`06_rag_query.py`）会自动将 `QWEN3_*` / `LLM_*` 桥接为
  Parlant 所需的 `LITELLM_PROVIDER_*`，并以 `NLPServices.litellm` 启动 Server
- 使用 Parlant LiteLLM 后端（Server 模式）时请确保已安装 `torch`（`pip install torch`）

### 3) 运行示例

```bash
# 环境连通性检查
python examples/00_check_env.py

# 知识库加载检查
python examples/01_load_knowledge_base.py

# 检索测试
python examples/02_test_retrieval.py --limit 3 --seed 42

# 检索 + LLM 评估
python examples/03_test_retrieval_eval.py --limit 2 --seed 42

# 独立 RAG 问答（无需 Parlant）
python examples/05_plan_a_agent.py --mode standalone --limit 3

# Parlant 模式（后台执行，无需 Server / UI）
python examples/05_plan_a_agent.py --mode parlant

# 两种方式都运行（默认）
python examples/05_plan_a_agent.py --limit 3 --seed 0

# Parlant Server 模式（完整集成：Server + Agent + GL + Tool，单条查询）
python examples/06_rag_query.py --query "你的问题" --output-format json

# 所有 examples 均支持结构化输出
python examples/02_test_retrieval.py --output-format json
```

## knowledge_base 说明

`plan_a.vdb.VectorStore` 兼容以下索引/文档格式：

- `faiss_index.bin` 或 `faiss_index`
- `document_store.json` 或 `docstore.json`
- 支持从 `knowledge_base/<dataset>.json` 回填文档内容（当持久化 docstore 为空时）

## 注意事项

- `plan_a/` 是独立模块，不依赖旧版实现模块。
- `third_party/` 为只读第三方代码，不应在业务修复中修改。
- `plan_a/runner.py` 封装了两种 RAG 模式的所有计算/设置逻辑；examples 脚本只做初始化和调用。
- Parlant 模式（`--mode parlant`）**无需启动 Server**，后台直接调用 `@p.tool` 接口，不弹出任何 UI。
- Parlant Server 模式（`06_rag_query.py`）启动 Server 时使用 `NLPServices.litellm`，模型来自 `.env` 配置。
