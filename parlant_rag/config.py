"""
RAG 配置管理 — 从 .env 文件加载配置。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RAGConfig:
    """RAG 模块配置。"""
    
    # 知识库配置
    knowledge_base_path: str = field(default="./knowledge_base")
    
    # 检索配置
    top_k: int = field(default=5)
    min_relevance_score: float = field(default=0.7)
    
    @classmethod
    def from_env(cls) -> RAGConfig:
        """
        从环境变量加载配置。
        
        知识库路径使用 KNOWLEDGE_PERSIST_DIR，与 .env 文件保持一致。
        """
        return cls(
            # 知识库路径
            knowledge_base_path=os.getenv("KNOWLEDGE_PERSIST_DIR", "./knowledge_base"),
            
            # 检索配置
            top_k=int(os.getenv("QWEN3_TOP_K", os.getenv("RAG_TOP_K", "5"))),
            min_relevance_score=float(os.getenv("RAG_MIN_SCORE", "0.7")),
        )
