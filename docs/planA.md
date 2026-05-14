**基于 Parlant 框架的「极简版」RAG 实现方案**

### 极简架构图回顾

```mermaid
flowchart TD
    User((用户)) <--> Agent[Parlant Agent]
    Agent --> GL[Guidelines]
    GL --> SR[Scoped Retriever]
    SR <--> RAG[RAG 服务]
    GL --> Tool[RAG Tool]
    Tool <--> RAG
    RAG <--> VDB[(向量数据库)]
```

---

### **模块解释一览表**

| 模块 | 名称 | 简要说明 | 示例 |
|------|------|----------|------|
| **User** | 用户 | 发起提问并接收回答的真实用户 | 用户在聊天框输入：“30天能退货吗？” |
| **Agent** | Parlant Agent | **整个系统的“大脑”**，负责接收消息、协调逻辑、生成最终回复 | Parlant 中通过 `create_agent()` 创建的智能客服助手 |
| **GL** | Guidelines | **规则/准则**，用自然语言定义“在什么情况下该做什么” | `condition="当用户询问退货政策、会员权益时"`<br>`action="去检索知识库后回答"` |
| **SR** | Scoped Retriever | **推荐主力**的 RAG 实现方式。绑定在 Guideline 上，**自动**检索相关知识 | 用户问退货政策时，自动调用 RAG 检索相关文档并注入上下文 |
| **Tool** | RAG Tool | **辅助方式**。需要 Agent 主动调用的检索工具，控制更精准 | `@p.tool` 定义的 `search_knowledge()` 函数，用户明确说“帮我查最新政策文档”时才调用 |
| **RAG** | RAG 服务 / Pipeline | 你自己实现的**检索管道**，负责 Embedding、向量搜索、排序等操作 | 使用 LlamaIndex 或 LangChain 编写的检索流程，接收问题后返回最相关文档 |
| **VDB** | 向量数据库 | 实际存储公司知识文档的地方 | Milvus / PGVector / Chroma 中存储的数百篇公司政策、产品手册、FAQ 文档 |

---

### 模块之间的关系简述

- **用户** 与 **Agent** 双向交互（问答）
- **Agent** 依赖 **Guidelines** 来判断当前对话该怎么处理
- **Guidelines** 可以同时触发两种知识获取方式：
  - **Scoped Retriever**（主力，自动、并行、高效）
  - **RAG Tool**（辅助，手动控制更强）
- 两种方式最终都调用 **RAG 服务** 去向量数据库中查找最相关的内容
- **RAG 服务** 处理完后把知识返回给 Parlant，由 Agent 生成最终回复

---

### 总结（一句话理解）

- **Parlant Agent** 是“大脑”
- **Guidelines** 是“决策规则”
- **Scoped Retriever** 是“自动查资料的主要方式”
- **RAG 服务 + 向量数据库** 是“真正的知识仓库”

