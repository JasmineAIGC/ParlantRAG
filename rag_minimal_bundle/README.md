# RAG 最小可分发包（Parlant-first）

这个目录可单独提供给其他人，用于快速运行 **基于 Parlant** 的 RAG 查询并输出 JSON 结果。

## 目录内容

- `run_rag_query.py`：Parlant-first RAG 查询入口脚本（GL -> SR / Tool -> RAGPipeline）
- `plan_a/`：RAG 主逻辑（检索管道、向量库、类型定义等）
- `model_client/`：Embedding/LLM 客户端封装
- `utils/`：日志工具
- `requirements.txt`：最小依赖
- `.env.example`：环境变量模板

## 快速使用

1. 安装依赖

```bash
pip install -r requirements.txt
pip install parlant torch
```

2. 配置环境变量

```bash
cp .env.example .env
```

3. 执行查询

```bash
python run_rag_query.py \
  --query "What is known for one of the stars of The Newcomers?" \
  --top-k 5 \
  --output-format json \
  --output-file results/rag_output.json
```

## `rag_output.json` 字段说明

顶层结构：

- `script`：脚本名（固定为 `run_rag_query.py`）
- `request`：请求参数
- `parlant`：Parlant Agent 与 Guideline 挂载信息
- `rag_tool_execution`：通过 Parlant RAG Tool 执行的结果

其中 `parlant.guidelines` 包含：

- `scoped_retriever_guideline_id`
- `rag_tool_guideline_id`
