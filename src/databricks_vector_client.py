"""
Mosaic AI Vector Search client — wraps a Direct Access Index with BYO 768-dim
sentence-transformer embeddings. Hybrid retrieval is delegated to the native
query_type="hybrid" parameter (keyword + semantic fused server-side in 2026
Mosaic AI) instead of the custom fusion in hybrid_search.py.

Design contract
---------------
- Fails fast: missing DATABRICKS_HOST/TOKEN or endpoint/index name raises
  RuntimeError unless ALLOW_LOCAL_FALLBACK=true is set. No silent degradation.
- Opt-in fallback: when ALLOW_LOCAL_FALLBACK=true, similarity_search() may
  route to the local vector_search.search_by_text path with a WARNING log.
  upsert() in fallback mode is a no-op and logs WARNING — the Databricks
  index is the source of truth, and writing to local SQLite during a
  fallback would diverge state silently.
- Returns a neutral result shape ({content_id, score, metadata, ...}) so the
  caller can migrate between this client and vector_search.search_by_text
  without type changes.

Required environment
--------------------
- DATABRICKS_HOST
- DATABRICKS_TOKEN
- DATABRICKS_VS_ENDPOINT    serverless VS endpoint name, e.g. "rag-vs-endpoint"
- DATABRICKS_VS_INDEX       full UC path, e.g. "main.rag.ai_content_index"

Fallback-only environment
-------------------------
- ALLOW_LOCAL_FALLBACK=true
"""
from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional, Sequence

from governance_metrics import record_fallback

logger = logging.getLogger("databricks_vector_client")

_REQUIRED_ENV = (
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_VS_ENDPOINT",
    "DATABRICKS_VS_INDEX",
)
_FALLBACK_FLAG = "ALLOW_LOCAL_FALLBACK"

DEFAULT_TOP_K = 10


def _fallback_allowed() -> bool:
    return os.getenv(_FALLBACK_FLAG, "").strip().lower() == "true"


def _missing_databricks_env() -> list[str]:
    return [v for v in _REQUIRED_ENV if not os.getenv(v)]


class MosaicAIVectorSearchClient:
    """
    One instance == one Direct Access Index in one serverless VS endpoint.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        index: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
    ):
        self.endpoint = endpoint or os.getenv("DATABRICKS_VS_ENDPOINT")
        self.index_name = index or os.getenv("DATABRICKS_VS_INDEX")
        self.columns = list(columns or ("content_id", "content_text", "source_type", "metadata_json"))
        self._index = None
        self._fallback_active = False

        missing = _missing_databricks_env()
        if not missing:
            self._index = self._build_index_handle()
            return

        if _fallback_allowed():
            logger.warning(
                "MosaicAIVectorSearchClient: Databricks env vars missing (%s); "
                "ALLOW_LOCAL_FALLBACK=true so routing reads to local SQLite "
                "vector_search. upsert() is a no-op in fallback mode.",
                ",".join(missing),
            )
            self._fallback_active = True
            record_fallback("vs")
            return

        raise RuntimeError(
            f"MosaicAIVectorSearchClient: required env vars missing: {missing}. "
            f"Set them to use Mosaic AI Vector Search, or set {_FALLBACK_FLAG}=true "
            f"to permit local fallback (dev only)."
        )

    def _build_index_handle(self):
        try:
            from databricks.vector_search.client import VectorSearchClient  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "databricks-vectorsearch is not installed. "
                "Run `pip install databricks-vectorsearch>=0.50`."
            ) from e
        client = VectorSearchClient()
        return client.get_index(endpoint_name=self.endpoint, index_name=self.index_name)

    def upsert(self, records: Iterable[dict]) -> None:
        """
        Upsert embeddings into the Direct Access Index.

        Each record must contain the primary key column ("content_id") plus
        the embedding vector under key "embedding" (list[float], 768-dim) and
        any filterable metadata columns declared when the index was created.
        """
        records = list(records)
        if not records:
            raise ValueError("upsert: records iterable was empty")
        self._validate_records(records)

        if self._fallback_active:
            logger.warning(
                "MosaicAIVectorSearchClient.upsert: no-op in fallback mode "
                "(%d records ignored). Set workspace env vars to enable writes.",
                len(records),
            )
            return

        if self._index is None:
            raise RuntimeError("upsert: index handle not initialized")
        self._index.upsert(records)

    @staticmethod
    def _validate_records(records: list[dict]) -> None:
        for i, r in enumerate(records):
            if not isinstance(r, dict):
                raise ValueError(f"record[{i}] must be a dict, got {type(r).__name__}")
            if "content_id" not in r:
                raise ValueError(f"record[{i}] missing required key 'content_id'")
            if "embedding" not in r:
                raise ValueError(f"record[{i}] missing required key 'embedding'")
            emb = r["embedding"]
            if not isinstance(emb, (list, tuple)) or not emb:
                raise ValueError(f"record[{i}] embedding must be non-empty list of floats")

    def similarity_search(
        self,
        query_text: str,
        query_vector: Sequence[float],
        k: int = DEFAULT_TOP_K,
        hybrid: bool = True,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Top-k similarity search. When hybrid=True, uses native
        query_type="hybrid" for server-side keyword + semantic fusion.

        Returns a list of {content_id, score, ...other requested columns}.
        """
        if not query_text:
            raise ValueError("query_text must be non-empty")
        if not query_vector:
            raise ValueError("query_vector must be non-empty")
        if k <= 0:
            raise ValueError("k must be positive")

        if self._fallback_active:
            return self._fallback_similarity_search(query_text, k)

        if self._index is None:
            raise RuntimeError("similarity_search: index handle not initialized")

        kwargs: dict[str, Any] = {
            "query_vector": list(query_vector),
            "num_results": k,
            "columns": self.columns,
        }
        if hybrid:
            kwargs["query_text"] = query_text
            kwargs["query_type"] = "hybrid"
        if filters:
            kwargs["filters_json"] = filters

        raw = self._index.similarity_search(**kwargs)
        return self._normalize_results(raw)

    @staticmethod
    def _normalize_results(raw: Any) -> list[dict]:
        """Flatten the Databricks VS response envelope into a list of dicts."""
        result = raw.get("result", raw) if isinstance(raw, dict) else raw
        data_array = result.get("data_array") if isinstance(result, dict) else None
        columns = result.get("manifest", {}).get("columns") if isinstance(result, dict) else None
        if not data_array or not columns:
            return []
        col_names = [c["name"] for c in columns]
        return [dict(zip(col_names, row)) for row in data_array]

    @staticmethod
    def _fallback_similarity_search(query_text: str, k: int) -> list[dict]:
        try:
            from vector_search import search_by_text  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Fallback path requires the local vector_search module. "
                "Ensure src/ is on sys.path."
            ) from e
        return search_by_text(query_text, top_k=k)

    @property
    def is_fallback(self) -> bool:
        """True iff this client is routing through the local vector_search fallback."""
        return self._fallback_active
