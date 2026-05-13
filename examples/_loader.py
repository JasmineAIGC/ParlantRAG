"""示例数据加载工具。"""
from __future__ import annotations

import json
from pathlib import Path


def load_hotpotqa_documents(
    path: str | Path = "knowledge_base/hotpotqa.json",
    max_examples: int = 40,
) -> list[dict]:
    """将 HotpotQA 示例数据转换为 ``id/content/source`` 文档片段列表。"""
    data_path = Path(path)
    rows = json.loads(data_path.read_text(encoding="utf-8"))[:max_examples]
    docs: list[dict] = []
    for row in rows:
        row_id = row.get("_id", "item")
        for title, sentences in row.get("context", []):
            content = " ".join(sentences).strip()
            if content:
                docs.append({"id": f"{row_id}:{title}", "content": content, "source": title})
    return docs
