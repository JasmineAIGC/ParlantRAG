"""
OpenAI 兼容 API 公共工具。

统一提供：

* **环境变量顺序读取** — :func:`first_env`
* **OpenAI 同步 / 异步客户端工厂** — :func:`make_openai_clients`

``LLMClient`` 和 ``OpenAILikeEmbedding`` 均通过此模块连接 OpenAI 兼容 API，
各自只需少量专用代码（model 选择、chat / embedding 端点调用）。

依赖::

    pip install openai>=1.0.0 python-dotenv
"""
from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI as _OpenAI, AsyncOpenAI as _AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    _OpenAI = None      # type: ignore[assignment]
    _AsyncOpenAI = None # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 环境变量工具
# ---------------------------------------------------------------------------

def first_env(keys: Sequence[str], default: Optional[str] = None) -> Optional[str]:
    """按顺序读取环境变量，返回首个非空值。

    Args:
        keys:    按优先级排列的环境变量名列表。
        default: 所有变量均未设置时的回退值（默认 ``None``）。

    Returns:
        首个非空环境变量的值，或 ``default``。
    """
    for key in keys:
        if value := os.getenv(key):
            return value
    return default


# ---------------------------------------------------------------------------
# OpenAI 客户端工厂
# ---------------------------------------------------------------------------

def make_openai_clients(
    base_url: str,
    api_key: Optional[str],
    access_key: Optional[str] = None,
) -> Tuple[Any, Any]:
    """返回 ``(sync_client, async_client)``，均已配置连接参数。

    两个客户端共享相同的 ``base_url`` / ``api_key`` / 可选 ``access-key`` 请求头，
    可直接用于 Embedding 或 Chat Completion 调用：

    .. code-block:: python

        sync_c, async_c = make_openai_clients(base_url, api_key, access_key)

        # Embedding 调用（同步）
        resp = sync_c.embeddings.create(model=model, input=texts)

        # Chat 调用（异步）
        resp = await async_c.chat.completions.create(model=model, messages=msgs)

    Args:
        base_url:   OpenAI 兼容 API 的 base URL（如 ``http://host:port/v1``）。
        api_key:    API Key；留空时使用 ``"EMPTY"``（本地部署惯例）。
        access_key: 可选的 ``access-key`` 请求头（部分私有部署网关需要）。

    Returns:
        ``(openai.OpenAI, openai.AsyncOpenAI)`` 元组。

    Raises:
        ImportError: 若未安装 ``openai`` SDK。
    """
    if not _OPENAI_AVAILABLE:
        raise ImportError(
            "make_openai_clients 需要安装 openai SDK：\n"
            "pip install openai>=1.0.0"
        )
    headers = {"access-key": access_key} if access_key else {}
    kw: dict = dict(
        api_key=api_key or "EMPTY",
        base_url=base_url,
        default_headers=headers or None,
    )
    return _OpenAI(**kw), _AsyncOpenAI(**kw)
