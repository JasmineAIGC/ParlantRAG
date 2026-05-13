"""
公共工具函数。

提供：
  - :func:`first_env` — 按优先级读取环境变量
  - :func:`make_openai_clients` — 创建 OpenAI 同步/异步客户端对

实际实现位于 :mod:`model_client._utils`，此处重新导出以保持向后兼容性。
"""
from model_client._utils import first_env, make_openai_clients

__all__ = ["first_env", "make_openai_clients"]
