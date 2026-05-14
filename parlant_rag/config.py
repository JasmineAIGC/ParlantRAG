"""
RAG 配置模块 — 基于 Parlant 框架的配置管理。

使用 Parlant 的配置模式，支持从环境变量和配置文件加载。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import os


@dataclass
class RAGConfig:
    """
    RAG 系统配置类。
    
    配置项：
    - knowledge_base_path: 知识库目录路径
    - vector_db_collection: 向量数据库集合名称
    - top_k: 检索返回的文档数量
    - max_context_length: 最大上下文长度（token）
    - retrieval_threshold: 检索相似度阈值
    - evaluation_enabled: 是否启用检索评估
    - log_level: 日志级别
    """
    
    knowledge_base_path: str = field(
        default_factory=lambda: os.environ.get("KNOWLEDGE_BASE_PATH", "./data/knowledge_base")
    )
    vector_db_collection: str = field(
        default_factory=lambda: os.environ.get("VECTOR_DB_COLLECTION", "parlant_rag")
    )
    top_k: int = field(
        default_factory=lambda: int(os.environ.get("TOP_K", "5"))
    )
    max_context_length: int = field(
        default_factory=lambda: int(os.environ.get("MAX_CONTEXT_LENGTH", "8192"))
    )
    retrieval_threshold: float = field(
        default_factory=lambda: float(os.environ.get("RETRIEVAL_THRESHOLD", "0.7"))
    )
    evaluation_enabled: bool = field(
        default_factory=lambda: os.environ.get("EVALUATION_ENABLED", "true").lower() == "true"
    )
    log_level: str = field(
        default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO")
    )
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """从环境变量加载配置。"""
        return cls()
    
    @classmethod
    def from_file(cls, config_path: str) -> "RAGConfig":
        """从配置文件加载配置。"""
        import yaml
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def validate(self) -> None:
        """验证配置的有效性。"""
        if self.top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if self.max_context_length <= 0:
            raise ValueError("max_context_length 必须大于 0")
        if not (0 <= self.retrieval_threshold <= 1):
            raise ValueError("retrieval_threshold 必须在 0 到 1 之间")
