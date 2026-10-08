"""Embedded Qdrant vector store adapter for Strata.

Implements the VectorStore protocol backed by local embedded Qdrant,
managing dense vector storage, differential index updates, and approximate
nearest neighbor search without external network or service dependencies.
"""

import logging
import sqlite3
import uuid
from collections.abc import Mapping
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.local.persistence import CollectionPersistence

# Proactively initialize CHECK_SAME_THREAD with a properly closed SQLite connection.
# In Python 3.14+, qdrant-client's default unclosed probe connection triggers a ResourceWarning.
if CollectionPersistence.CHECK_SAME_THREAD is None:
    _conn = sqlite3.connect(":memory:")
    try:
        _row = _conn.execute(
            "select * from pragma_compile_options where compile_options like ?",
            ("THREADSAFE=%",),
        ).fetchone()
        _threadsafe = str(_row[0]) if _row is not None else ""
        CollectionPersistence.CHECK_SAME_THREAD = _threadsafe != "THREADSAFE=1"
    finally:
        _conn.close()

from strata.core.config import Settings, get_settings
from strata.core.models import Chunk, RetrievedChunk, RetrievedContext
from strata.core.protocols import VectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStore):
    """Embedded Qdrant vector index adapter satisfying the VectorStore protocol.

    Provides dense vector storage, differential point invalidation, and cosine
    similarity search over granular search chunks.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: QdrantClient | None = None,
        location: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        """Initialize embedded Qdrant vector store.

        Args:
            settings: Optional Strata Settings instance.
            client: Optional pre-configured QdrantClient (e.g. for testing).
            location: Optional client location string (e.g. ':memory:').
            collection_name: Optional override for the vector collection name.
        """
        self.settings: Settings = settings or get_settings()
        self.collection_name: str = collection_name or self.settings.qdrant_collection_name

        if client is not None:
            self.client: QdrantClient = client
        elif location is not None:
            self.client = QdrantClient(location=location)
        else:
            self.settings.qdrant_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(self.settings.qdrant_path))

        self._ensure_collection()

    def _resolve_distance_metric(self) -> models.Distance:
        """Resolve distance metric from settings to Qdrant Distance enum."""
        metric_str = self.settings.distance_metric.strip().upper()
        try:
            return models.Distance[metric_str]
        except KeyError:
            logger.warning(
                "Unknown distance metric '%s', falling back to COSINE",
                self.settings.distance_metric,
            )
            return models.Distance.COSINE

    def _ensure_collection(self) -> None:
        """Verify collection exists; if not, create it with configured vector parameters."""
        if not self.client.collection_exists(self.collection_name):
            distance = self._resolve_distance_metric()
            logger.info(
                "Creating Qdrant collection '%s' (dim=%d, distance=%s)",
                self.collection_name,
                self.settings.embedding_dimension,
                distance.name,
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.settings.embedding_dimension,
                    distance=distance,
                ),
            )

    @staticmethod
    def _to_point_id(chunk_id: str) -> str:
        """Normalize a string chunk ID into a valid Qdrant UUID string."""
        try:
            return str(uuid.UUID(chunk_id))
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]] | None = None,
        *,
        vectors: list[list[float]] | None = None,
    ) -> None:
        """Upsert chunk vectors and light metadata payloads into vector store.

        Args:
            chunks: List of Chunk domain models to index.
            embeddings: Corresponding dense embedding float vectors.
            vectors: Alias for embeddings.

        Raises:
            ValueError: If neither embeddings nor vectors are supplied, or
                if counts/dimensions do not match settings.
        """
        effective_vectors = vectors if vectors is not None else embeddings
        if effective_vectors is None:
            raise ValueError("Either 'embeddings' or 'vectors' must be provided.")

        if len(chunks) != len(effective_vectors):
            raise ValueError(
                f"Mismatch between number of chunks ({len(chunks)}) "
                f"and vectors ({len(effective_vectors)})"
            )

        if not chunks:
            return

        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, effective_vectors, strict=True):
            if len(vector) != self.settings.embedding_dimension:
                raise ValueError(
                    f"Vector dimension {len(vector)} does not match expected "
                    f"dimension {self.settings.embedding_dimension} for chunk {chunk.id}"
                )

            point_id = self._to_point_id(chunk.id)
            payload: dict[str, Any] = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "heading_path": chunk.heading_path,
                "tags": list(chunk.tags),
                "chunk_hash": chunk.chunk_hash,
                "index": chunk.index,
                "token_count": chunk.token_count,
                "parent_section_id": chunk.parent_section_id,
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.debug(
            "Upserted %d points to collection '%s'",
            len(points),
            self.collection_name,
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        payload_filter: Mapping[str, object] | None = None,
        tags: list[str] | None = None,
    ) -> list[RetrievedContext]:
        """Execute approximate nearest neighbor search with optional filters.

        Args:
            query_vector: Dense embedding vector to query.
            limit: Maximum candidate points to return.
            payload_filter: Optional arbitrary key-value metadata filter.
            tags: Optional tags list to filter candidates.

        Returns:
            Ranked list of RetrievedContext (RetrievedChunk) models sorted by score descending.
        """
        if not query_vector or limit <= 0:
            return []

        if len(query_vector) != self.settings.embedding_dimension:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} does not match expected "
                f"dimension {self.settings.embedding_dimension}"
            )

        conditions: list[models.Condition] = []

        if tags:
            conditions.append(
                models.FieldCondition(
                    key="tags",
                    match=models.MatchAny(any=tags),
                )
            )

        if payload_filter:
            for key, val in payload_filter.items():
                if key == "tags":
                    if isinstance(val, list):
                        conditions.append(
                            models.FieldCondition(
                                key="tags",
                                match=models.MatchAny(any=[str(x) for x in val]),
                            )
                        )
                    else:
                        conditions.append(
                            models.FieldCondition(
                                key="tags",
                                match=models.MatchValue(value=str(val)),
                            )
                        )
                elif isinstance(val, (list, tuple)):
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=list(val)),
                        )
                    )
                else:
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=val),  # type: ignore[arg-type]
                        )
                    )

        qdrant_filter: models.Filter | None = models.Filter(must=conditions) if conditions else None

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=qdrant_filter,
        )

        retrieved: list[RetrievedContext] = []
        for rank, scored_point in enumerate(response.points, start=1):
            payload = scored_point.payload or {}
            chunk_id = str(payload.get("chunk_id", scored_point.id))
            doc_id = str(payload.get("document_id", ""))
            heading_path = str(payload.get("heading_path", ""))
            chunk_hash = str(payload.get("chunk_hash", ""))
            chunk_tags = [str(t) for t in payload.get("tags", [])]
            chunk_index = int(payload.get("index", 0))
            token_count = int(payload.get("token_count", 0))
            parent_section_id = payload.get("parent_section_id")
            if parent_section_id is not None:
                parent_section_id = str(parent_section_id)

            chunk = Chunk(
                id=chunk_id,
                document_id=doc_id,
                chunk_hash=chunk_hash,
                text="",
                heading_path=heading_path,
                index=chunk_index,
                parent_section_id=parent_section_id,
                token_count=token_count,
                tags=chunk_tags,
            )
            score = float(scored_point.score)
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    dense_score=score,
                    dense_rank=rank,
                )
            )

        return retrieved

    def search_tuples(
        self,
        query_vector: list[float],
        limit: int = 10,
        tags: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Convenience method returning raw (chunk_id, score) tuples.

        Args:
            query_vector: Dense embedding vector to query.
            limit: Maximum candidate points to return.
            tags: Optional tags list to filter candidates.

        Returns:
            Ranked list of (chunk_id, score) tuples ordered by descending relevance.
        """
        results = self.search(query_vector=query_vector, limit=limit, tags=tags)
        return [(hit.chunk.id, hit.score) for hit in results]

    def delete_by_ids(self, chunk_ids: list[str]) -> None:
        """Delete points matching the provided chunk IDs from the collection."""
        if not chunk_ids:
            return

        point_ids: list[models.ExtendedPointId] = [self._to_point_id(cid) for cid in chunk_ids]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=point_ids),
        )
        logger.debug(
            "Deleted %d points from collection '%s'",
            len(point_ids),
            self.collection_name,
        )

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunk points belonging to a document ID."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
        logger.debug(
            "Deleted points for document_id '%s' from collection '%s'",
            document_id,
            self.collection_name,
        )

    def delete_by_chunk_hashes(self, chunk_hashes: list[str]) -> None:
        """Delete specific chunk points matching chunk hashes."""
        if not chunk_hashes:
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="chunk_hash",
                            match=models.MatchAny(any=chunk_hashes),
                        )
                    ]
                )
            ),
        )
        logger.debug(
            "Deleted %d chunk hashes from collection '%s'",
            len(chunk_hashes),
            self.collection_name,
        )

    def get_all_point_ids(self) -> set[str]:
        """Scroll through all points in the collection and return the set of string chunk IDs."""
        point_ids: set[str] = set()
        offset: models.ExtendedPointId | None = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                if record.payload and "chunk_id" in record.payload:
                    point_ids.add(str(record.payload["chunk_id"]))
                else:
                    point_ids.add(str(record.id))
            if offset is None:
                break
        return point_ids

    def count(self) -> int:
        """Return total number of vectors in collection."""
        res = self.client.count(collection_name=self.collection_name)
        return res.count

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying Qdrant client connection and release resources."""
        if hasattr(self, "client") and hasattr(self.client, "close"):
            self.client.close()

    def __enter__(self) -> "QdrantVectorStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()
