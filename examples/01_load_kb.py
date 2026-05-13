"""
01_load_kb.py — 测试加载 FlexRAG 构建的知识库

验证 FAISS 索引文件能否被正常加载，并打印基本的索引信息。
不发起任何检索查询，仅确认知识库文件完整可用。

运行方式（从项目根目录 或 examples/ 目录均可）：
    python examples/01_load_kb.py
    # 或
    cd examples && python 01_load_kb.py

前置条件：
    1. 用 FlexRAG scripts/build_knowledge_base.py 构建好知识库
    2. 复制并填写 .env.example → .env，至少配置以下变量：
         KNOWLEDGE_PERSIST_DIR=./knowledge_base/knowledge_persist_dir/hotpotqa
         EMBEDDING_MODEL_PATH=/path/to/models/BAAI/bge-large-zh-v1.5
         EMBEDDING_MODEL=bge-large-zh-v1.5
    3. pip install python-dotenv llama-index llama-index-vector-stores-faiss faiss-cpu sentence-transformers
"""
from __future__ import annotations

import logging
import os
import sys

# 确保从任意目录运行时都能找到 parlant_rag（无需设置 PYTHONPATH）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志：将 INFO 及以上级别输出到控制台，包含完整异常 traceback。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

from dotenv import load_dotenv

load_dotenv()

from parlant_rag.retrieval import FlexRAGRetriever  # noqa: E402


def main() -> None:
    persist_dir = os.getenv("KNOWLEDGE_PERSIST_DIR", "./knowledge_base")

    print(f"{'='*60}")
    print(f"📂 知识库路径：{persist_dir}")
    print(f"{'='*60}")

    # 检查索引文件是否存在
    faiss_path = os.path.join(persist_dir, "faiss_index.bin")
    if not os.path.exists(faiss_path):
        print(f"\n❌ 未找到 FAISS 索引文件：{faiss_path}")
        print("   请先执行 FlexRAG scripts/build_knowledge_base.py 构建知识库。")
        return

    print(f"\n✅ FAISS 索引文件已找到：{faiss_path}")

    # 加载知识库
    print("\n⏳ 正在加载知识库...")
    retriever = FlexRAGRetriever.from_env(persist_dir=persist_dir)

    # 打印索引基本信息
    print(f"\n✅ 知识库加载成功")
    print(f"   文档片段总数：{retriever.doc_count}")
    print(f"   Embedding 模型：{retriever.embedding_model}")
    if retriever.embedding_mode == "local":
        print(f"   Embedding 模式：本地（{retriever.embedding_model_path}）")
    else:
        print(f"   Embedding 模式：API（{retriever.embedding_base_url}）")


if __name__ == "__main__":
    main()
