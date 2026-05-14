from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from plan_a.core import PlanAConfig
from plan_a.vdb import VectorStore


class PlanAConfigTests(unittest.TestCase):
    def test_from_env_defaults_to_local_hotpotqa_persist_dir(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = PlanAConfig.from_env()

        self.assertEqual(
            config.knowledge_base_path,
            "./knowledge_base/knowledge_persist_dir/hotpotqa",
        )
        self.assertEqual(config.top_k, 5)


class VectorStorePathResolutionTests(unittest.TestCase):
    def test_accepts_direct_faiss_index_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            index_path = base / "faiss_index.bin"
            index_path.write_bytes(b"index")

            store = VectorStore()
            persist_dir, resolved_index = store._resolve_persist_dir_and_index(str(index_path))

            self.assertEqual(persist_dir, base)
            self.assertEqual(resolved_index, index_path)

    def test_accepts_knowledge_persist_dir_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            dataset_dir = base / "hotpotqa"
            dataset_dir.mkdir()
            index_path = dataset_dir / "faiss_index.bin"
            index_path.write_bytes(b"index")

            store = VectorStore()
            persist_dir, resolved_index = store._resolve_persist_dir_and_index(str(base))

            self.assertEqual(persist_dir, dataset_dir)
            self.assertEqual(resolved_index, index_path)


if __name__ == "__main__":
    unittest.main()
