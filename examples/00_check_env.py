"""
00_check_env.py — 验证 .env 配置的所有 API 和本地模型是否真正可用

根据 .env 文件中的配置，依次检测：
  1. LLM API（QWEN3_BASE_URL / LLM_BASE_URL）— 发送一次 chat completion 请求
  2. Embedding API（EMBEDDING_BASE_URL）— 请求一次 embeddings 端点
  3. 本地 Embedding 模型（EMBEDDING_MODEL_PATH）— 加载并执行一次推理

所有已配置项均通过时打印 "✅ 全部通过"；任何已配置项失败则以退出码 1 结束，
方便在 CI 或启动脚本中使用。

运行方式（从项目根目录 或 examples/ 目录均可）：
    python examples/00_check_env.py
    # 或
    cd examples && python 00_check_env.py

前置条件：
    1. 复制并填写 .env.example → .env
    2. pip install python-dotenv openai
       （本地 Embedding 模式还需 sentence-transformers llama-index）
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

# 确保从任意目录运行时都能找到 rag_module（无需设置 PYTHONPATH）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from model_client._utils import first_env  # noqa: E402

# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

_PASS = "✅"
_FAIL = "❌"
_SKIP = "⏭ "


def _header(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def _result(label: str, ok: bool, detail: str = "") -> None:
    mark = _PASS if ok else _FAIL
    suffix = f"  →  {detail}" if detail else ""
    print(f"  {mark}  {label}{suffix}")


# ---------------------------------------------------------------------------
# 检测项 1：LLM API（openai 兼容 chat completions）
# ---------------------------------------------------------------------------

async def _check_llm() -> tuple[bool, str]:
    """发送 Hello 给 LLM，返回 (是否成功, 描述信息)。"""
    llm_url = first_env([
        "QWEN3_BASE_URL", "LLM_BASE_URL", "CONTEXTMIND_BASE_URL",
        "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL",
    ])
    llm_key = first_env([
        "QWEN3_API_KEY", "LLM_API_KEY", "CONTEXTMIND_API_KEY",
        "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
    ])
    llm_model = first_env([
        "QWEN3_MODEL", "LLM_MODEL", "CONTEXTMIND_MODEL",
        "OPENAI_MODEL", "DEEPSEEK_MODEL",
    ], default="")

    if not llm_url:
        return False, "未配置 QWEN3_BASE_URL / LLM_BASE_URL，跳过"

    label = f"LLM API  model={llm_model or '(未配置)'}  url={llm_url}"

    try:
        from openai import AsyncOpenAI  # type: ignore[import]
        client = AsyncOpenAI(api_key=llm_key or "EMPTY", base_url=llm_url)
        t0 = time.monotonic()
        resp = await client.chat.completions.create(
            model=llm_model or "default",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=16,
        )
        elapsed = time.monotonic() - t0
        reply = (resp.choices[0].message.content or "").strip()[:60]
        return True, f"{label}  [{elapsed:.1f}s]  回复：{reply!r}"
    except Exception as exc:
        _msg = str(exc).lower()
        if any(k in _msg for k in ("connection", "connect", "refused", "unreachable", "timeout")):
            return False, f"{label}  无法连接（端口未开放 / 服务未启动）"
        return False, f"{label}  错误：{exc}"


# ---------------------------------------------------------------------------
# 检测项 2：Embedding API（openai SDK）
# ---------------------------------------------------------------------------

async def _check_embedding_api() -> tuple[bool, str]:
    """发送一次 embeddings 请求，返回 (是否成功, 描述信息)。"""
    emb_url = first_env(["EMBEDDING_BASE_URL"])
    emb_key = first_env(["EMBEDDING_API_KEY", "QWEN3_API_KEY", "LLM_API_KEY"])
    emb_model = first_env(["EMBEDDING_MODEL"], default="")

    if not emb_url:
        return False, "未配置 EMBEDDING_BASE_URL，跳过"

    label = f"Embedding API  model={emb_model or '(未配置)'}  url={emb_url}"

    try:
        from openai import AsyncOpenAI  # type: ignore[import]
        client = AsyncOpenAI(api_key=emb_key or "EMPTY", base_url=emb_url)
        t0 = time.monotonic()
        resp = await client.embeddings.create(
            model=emb_model or "default",
            input=["测试向量"],
        )
        elapsed = time.monotonic() - t0
        dim = len(resp.data[0].embedding)
        return True, f"{label}  [{elapsed:.1f}s]  dim={dim}"
    except Exception as exc:
        return False, f"{label}  错误：{exc}"


# ---------------------------------------------------------------------------
# 检测项 3：本地 Embedding 模型（sentence-transformers）
# ---------------------------------------------------------------------------

async def _check_local_embedding() -> tuple[bool, str]:
    """加载本地模型并推理，返回 (是否成功, 描述信息)。"""
    model_path = os.getenv("EMBEDDING_MODEL_PATH", "")
    if not model_path:
        return False, "未配置 EMBEDDING_MODEL_PATH，跳过"

    label = f"本地 Embedding  path={model_path}"

    if not os.path.exists(model_path):
        return False, f"{label}  路径不存在"

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
        t0 = time.monotonic()
        model = await asyncio.to_thread(SentenceTransformer, model_path)
        vec = await asyncio.to_thread(
            model.encode, ["测试向量"], normalize_embeddings=True, show_progress_bar=False
        )
        elapsed = time.monotonic() - t0
        dim = len(vec[0])
        return True, f"{label}  [{elapsed:.1f}s]  dim={dim}"
    except Exception as exc:
        return False, f"{label}  错误：{exc}"


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

async def main() -> int:
    print("=" * 60)
    print("  ParlantRAG — .env 配置可用性检测")
    print("=" * 60)

    checks = [
        ("LLM API", _check_llm()),
        ("Embedding API", _check_embedding_api()),
        ("本地 Embedding 模型", _check_local_embedding()),
    ]

    results: list[tuple[str, bool, str]] = []
    for name, coro in checks:
        _header(f"检测：{name}")
        ok, detail = await coro
        skipped = "跳过" in detail
        if skipped:
            print(f"  {_SKIP}  {detail}")
            # 未配置不算失败
            results.append((name, True, detail))
        else:
            _result(name, ok, detail)
            results.append((name, ok, detail))

    # 汇总
    print(f"\n{'='*60}")
    print("  检测汇总")
    print(f"{'='*60}")
    failed = []
    for name, ok, detail in results:
        skipped = "跳过" in detail
        if skipped:
            print(f"  {_SKIP}  {name:<20}  （未配置，已跳过）")
        else:
            mark = _PASS if ok else _FAIL
            status = "通过" if ok else "失败"
            print(f"  {mark}  {name:<20}  {status}")
            if not ok:
                failed.append(name)

    print()
    if failed:
        print(f"❌ 以下项目检测失败：{', '.join(failed)}")
        print("   请检查 .env 配置及对应服务是否已启动。")
        return 1
    else:
        configured = [n for n, _, d in results if "跳过" not in d]
        if configured:
            print(f"✅ 全部已配置项目检测通过（{len(configured)} 项）。")
        else:
            print("⚠️  .env 中未检测到任何已配置项（LLM / Embedding），请检查配置。")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
