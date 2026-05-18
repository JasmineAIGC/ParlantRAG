# RAG 过程分析：核心检索机制与 Parlant 的作用

> 分析对象：`rag_query_bundle/run_rag_query.py`  
> 涉及模块：`plan_a/rag/pipeline.py`、`plan_a/vdb/vector_store.py`、`plan_a/retrievers/scoped.py`、`plan_a/tools/rag.py`、`plan_a/guidelines/setup.py`、`plan_a/runner.py`

---

## 一、RAG 核心过程（与 Parlant 无关的纯检索）

RAG（Retrieval-Augmented Generation）的核心是"先检索、再生成"。检索部分在本仓库中完全由 `RAGPipeline.retrieve_standard()` 实现，分两步：

```
用户查询 (str)
  │
  ▼ Embedder.embed_query()        # 调用 Embedding 模型，将文字映射为高维向量
query_vector (List[float])
  │
  ▼ VectorStore.search()          # FAISS 近似最近邻搜索，返回向量空间中最近的 K 个文档片段
List[DocChunk]  （按相似度降序）
```

**步骤说明：**

1. **文本向量化（Embedding）**  
   使用 `Embedder`（优先本地 SentenceTransformer，否则 OpenAI-like API）把用户的自然语言查询转成一个浮点数向量。所有文档在建库时也用同一模型转成了向量，存入 FAISS 索引。

2. **向量检索（FAISS ANN Search）**  
   `VectorStore.search(query_vector, top_k)` 在预建的 FAISS 索引（`faiss_index.bin`）里做近似最近邻搜索，找出与查询向量余弦相似度最高的 K 个文档片段，耗时通常只有几毫秒。

3. **返回 DocChunk 列表**  
   每个片段包含 `content`（正文）、`source`（来源文件）、`score`（相似度得分）。

**生成（Generation）是可选的后处理：**  
只有在加 `--answer` 参数时，才会把检索到的 chunks 拼成 prompt 调用 LLM（`LLMClient.chat_async()`）生成自然语言回答。

---

## 二、`run_rag_query.py` 实际执行路径

```
main()
  ├─ [阶段1-纯RAG] RAGPipeline.initialize()
  │               └─ VectorStore.initialize()   # 加载 FAISS 索引到内存
  │
  ├─ [阶段1-Parlant] ParlantServerContext.__aenter__()
  │                  ├─ p.Server(nlp_service=litellm)   # 启动 Parlant 内核
  │                  ├─ server.create_agent()            # 创建 Agent
  │                  └─ setup_guidelines()               # 注册 2 条 Guideline
  │                      ├─ GL1: Scoped Retriever（自动检索，注入上下文）
  │                      └─ GL2: RAG Tool（Agent 主动调用）
  │
  ├─ [阶段2] ctx.run_queries([query], llm=None)
  │           └─ search_knowledge(context, query)        # @p.tool 装饰的工具函数
  │               └─ RAGPipeline.retrieve_standard()     # ← 真正的 FAISS 检索（≈几毫秒）
  │
  └─ [阶段3-可选] LLMClient.chat_async()                 # --answer 开启时生成回答
```

**结论：** 真正的检索耗时只占整个流程的极小一部分（几毫秒），Parlant 的初始化（Server + LiteLLM + 实体嵌入缓存）才是主要启动开销。Parlant 在这条链路里充当"容器和调度层"。

---

## 三、为什么要使用 Parlant？

Parlant 带来的不是"更好的检索算法"，而是**三个工程层面的价值**：

### 3.1 Guideline 驱动的智能触发（最核心）

不使用 Parlant 时，需要自己写逻辑决定"这个问题要不要检索"，往往是硬编码的 if-else 或者一次额外的 LLM 分类调用。

Parlant 的 Guideline 把这件事**声明化**：

```python
await agent.create_guideline(
    condition="用户提问涉及知识库内容，例如政策、产品信息…",
    action="自动检索知识库，将相关内容作为上下文…",
)
await scoped_guideline.attach_retriever(retriever_fn)
```

条件匹配时，Parlant 引擎**自动**执行检索并将结果注入 Agent 推理上下文（Scoped Retriever），无需 Agent 显式调用任何工具。

### 3.2 双路检索架构（主动 + 被动）

本仓库配置了两条 Guideline，覆盖不同的检索触发场景：

| 方式 | 实现 | 触发逻辑 |
|------|------|----------|
| **Scoped Retriever（主力）** | `guideline.attach_retriever()` | Guideline 条件匹配时 Parlant 自动触发，结果直接注入上下文 |
| **RAG Tool（辅助）** | `@p.tool` + `guideline.tools=[...]` | Agent 在多步推理时主动调用，更精准可控 |

若不用 Parlant，只能选其一，且需要自己实现调度逻辑。

### 3.3 服务化与会话管理

`ParlantServerContext` 的设计是"初始化一次，多轮复用"：  
Server + Agent + Guidelines 在 `__aenter__` 时创建一次，后续每次 `run_queries()` 直接复用，不会重复初始化。

部署为 HTTP 服务时，这意味着多用户并发提问都由同一 Parlant Agent 处理，具备**会话状态保持、多轮对话上下文、工具调用审计**等能力。

---

## 四、与"不用 Parlant 的纯检索"对比

| 维度 | 不用 Parlant（纯检索） | 用 Parlant（本仓库方案）|
|------|----------------------|------------------------|
| **启动开销** | 几乎为零（只加载 FAISS 索引） | 需要几秒（Server + LiteLLM + 实体嵌入缓存） |
| **检索触发** | 每次调用都触发，无条件判断 | Guideline 按语义条件按需触发 |
| **检索结果用途** | 直接返回给调用方 | 自动注入 Agent 推理上下文，LLM 生成时"看得见"检索内容 |
| **多步推理** | 需自行实现 | RAG Tool 允许 Agent 在推理链中按需调用检索（ReAct 模式） |
| **对话状态** | 无状态，每次独立 | Agent 维护会话上下文，支持多轮追问 |
| **可扩展性** | 需自己加中间件 | 可直接叠加 Parlant 的其他工具、Guidelines、评估模块 |
| **适合场景** | 批量离线检索、评测、轻量脚本 | 实时对话服务、需要 Agent 推理的问答系统 |

---

## 五、总结

`run_rag_query.py` 当前的主要作用是**验证/演示"Parlant + FAISS 检索"集成链路是否工作正常**。对于单次离线查询，完全可以绕过 Parlant 直接调 `RAGPipeline.retrieve_standard()`，启动开销更低。

Parlant 真正发挥价值的场景是：**需要 Agent 在对话中按需检索、多步推理、维护会话上下文**。此时 Guideline + Scoped Retriever + RAG Tool 的组合，比"自己写 if-else 调检索"更健壮、更可维护，并且天然支持从脚本升级为生产级对话服务。
