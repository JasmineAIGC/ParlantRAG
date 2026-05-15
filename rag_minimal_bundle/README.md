# RAG 最小可分发包（基于 `run_rag_query.py`）

这个目录可单独提供给其他人，用于快速运行 RAG 检索并输出 JSON 结果。

## 目录内容

- `run_rag_query.py`：RAG 查询入口脚本
- `plan_a/`：RAG 主逻辑（检索管道、向量库、类型定义等）
- `model_client/`：Embedding/LLM 客户端封装
- `utils/`：日志工具
- `requirements.txt`：最小依赖
- `.env.example`：环境变量模板

## 快速使用

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 配置环境变量

```bash
cp .env.example .env
# 然后编辑 .env，至少填写：
# - KNOWLEDGE_PERSIST_DIR（或 KNOWLEDGE_BASE_PATH）
# - Embedding 配置（本地或 API）
```

3. 执行查询（推荐先跳过 Parlant 校验）

```bash
python run_rag_query.py \
  --query "What is known for one of the stars of The Newcomers?" \
  --top-k 5 \
  --output-format json \
  --output-file results/rag_output.json \
  --skip-parlant
```

> 如果你安装了 `parlant`（及其依赖，例如 `torch`），可去掉 `--skip-parlant`。

---

## `rag_output.json` 字段说明

顶层结构：

- `script`：脚本名（固定为 `run_rag_query.py`）
- `request`：本次请求参数
- `retrieval`：标准检索结果
- `parlant_tool`：Parlant Tool 验证结果（可选/可跳过）

### 1) `request`

- `query` (`string`)：查询文本
- `top_k` (`number | null`)：期望返回的片段上限；为 `null` 时使用配置默认值
- `include_metadata` (`boolean`)：是否在输出中包含片段元数据

### 2) `retrieval`

- `query` (`string`)：实际执行检索的查询
- `top_k` (`number`)：本次检索使用的 top_k
- `chunk_count` (`number`)：实际返回片段数
- `took_ms` (`number`)：检索耗时（毫秒）
- `chunks` (`array`)：命中文档片段列表

`chunks[i]` 字段：

- `content` (`string`)：片段正文
- `source` (`string`)：来源标识（文档名/URL 等）
- `score` (`number`)：相似度分数（越大越相关）
- `metadata` (`object`, 可选)：仅当 `include_metadata=true` 才返回

### 3) `parlant_tool`

常见状态：

- `{"status": "skipped"}`：使用了 `--skip-parlant`
- `{"status": "ok", "tool_name": "...", "tool_result": {...}}`：Parlant 验证成功
- `{"status": "missing_torch", "error": "..."}`：缺少 `torch`
- `{"status": "missing_parlant", "error": "..."}`：未安装 `parlant`

其中 `tool_result` 的结构与 `retrieval` 基本一致（用于校验 Tool 调用结果）。

---

## 运行前检查清单

- 索引目录存在：`faiss_index.bin`（或 `faiss_index`）
- 文档存储存在：`document_store.json`（或 `docstore.json`）
- `.env` 中 Embedding 配置可用
