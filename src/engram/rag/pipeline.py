import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4

from engram.context import ContextSource
from engram.tools import Tool, ToolResult


@dataclass(frozen=True, slots=True)
class Document:
    content: str
    id: str = field(default_factory=lambda: uuid4().hex)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    content: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk: Chunk
    score: float


class SearchIndex(Protocol):
    def add(self, chunks: Iterable[Chunk]) -> None: ...

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]: ...


class EmbeddingModel(Protocol):
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class TextChunker:
    """Split text on paragraph or whitespace boundaries with bounded overlap."""

    def __init__(self, max_chars: int = 1_600, overlap_chars: int = 160) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than zero.")
        if not 0 <= overlap_chars < max_chars:
            raise ValueError("overlap_chars must be between zero and max_chars.")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def split(self, document: Document) -> list[Chunk]:
        text = document.content.strip()
        if not text:
            return []
        chunks: list[Chunk] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.max_chars)
            if end < len(text):
                boundary = max(text.rfind("\n\n", start, end), text.rfind(" ", start, end))
                if boundary > start + self.max_chars // 2:
                    end = boundary
            content = text[start:end].strip()
            if content:
                chunks.append(
                    Chunk(
                        id=f"{document.id}:{len(chunks)}",
                        document_id=document.id,
                        content=content,
                        index=len(chunks),
                        metadata=dict(document.metadata),
                    )
                )
            if end >= len(text):
                break
            start = max(start + 1, end - self.overlap_chars)
        return chunks


class LexicalIndex:
    """A deterministic dependency-free retriever for small knowledge bases."""

    _tokens = re.compile(r"[\w]+|[^\x00-\x7F]", re.UNICODE)

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._terms: dict[str, Counter[str]] = {}

    def add(self, chunks: Iterable[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            self._terms[chunk.id] = Counter(self._tokenize(chunk.content))

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        if limit <= 0:
            return []
        query_terms = Counter(self._tokenize(query))
        if not query_terms:
            return []
        document_frequency = Counter(term for terms in self._terms.values() for term in set(terms))
        count = max(1, len(self._terms))

        def score(chunk_id: str) -> float:
            terms = self._terms[chunk_id]
            total = 0.0
            for term, query_count in query_terms.items():
                if term not in terms:
                    continue
                inverse_frequency = math.log(1 + count / (1 + document_frequency[term]))
                total += min(query_count, terms[term]) * inverse_frequency
            return total / math.sqrt(max(1, sum(terms.values())))

        ranked = sorted(self._chunks, key=score, reverse=True)
        return [
            SearchHit(self._chunks[chunk_id], score(chunk_id))
            for chunk_id in ranked[:limit]
            if score(chunk_id) > 0
        ]

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return [match.group(0).lower() for match in cls._tokens.finditer(text)]


class VectorIndex:
    """An in-process cosine index backed by any user-provided embedding model."""

    def __init__(self, embeddings: EmbeddingModel) -> None:
        self.embeddings = embeddings
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, tuple[float, ...]] = {}

    def add(self, chunks: Iterable[Chunk]) -> None:
        values = list(chunks)
        vectors = self.embeddings.embed([chunk.content for chunk in values])
        if len(vectors) != len(values):
            raise ValueError("Embedding model returned the wrong number of vectors.")
        for chunk, vector in zip(values, vectors, strict=True):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = tuple(float(item) for item in vector)

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        if limit <= 0 or not self._chunks:
            return []
        vectors = self.embeddings.embed([query])
        if len(vectors) != 1:
            raise ValueError("Embedding model must return one vector for one query.")
        query_vector = tuple(float(item) for item in vectors[0])
        ranked = sorted(
            self._chunks,
            key=lambda chunk_id: _cosine(query_vector, self._vectors[chunk_id]),
            reverse=True,
        )
        return [
            SearchHit(self._chunks[chunk_id], _cosine(query_vector, self._vectors[chunk_id]))
            for chunk_id in ranked[:limit]
        ]


class KnowledgeBase:
    """A compact ingestion-retrieval pipeline that can feed agents or tools."""

    def __init__(
        self,
        index: SearchIndex | None = None,
        chunker: TextChunker | None = None,
    ) -> None:
        self.index = index or LexicalIndex()
        self.chunker = chunker or TextChunker()
        self.documents: dict[str, Document] = {}

    def add(self, documents: Iterable[Document]) -> int:
        chunks: list[Chunk] = []
        for document in documents:
            self.documents[document.id] = document
            chunks.extend(self.chunker.split(document))
        self.index.add(chunks)
        return len(chunks)

    def add_texts(self, texts: Iterable[str]) -> int:
        return self.add(Document(content=text) for text in texts)

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        return self.index.search(query, limit=limit)

    def context_source(self, query: str, *, limit: int = 5) -> ContextSource:
        hits = self.search(query, limit=limit)
        content = "\n\n".join(
            f"Source {position} (score={hit.score:.3f}):\n{hit.chunk.content}"
            for position, hit in enumerate(hits, start=1)
        )
        return ContextSource(name="knowledge", content=content, priority=10)

    def as_tool(self, *, name: str = "retrieve_knowledge") -> Tool:
        knowledge = self

        def retrieve(query: str, limit: int = 5) -> ToolResult:
            """Retrieve relevant passages from the local knowledge base."""
            hits = knowledge.search(query, limit=limit)
            text = "\n\n".join(hit.chunk.content for hit in hits) or "No passages matched."
            return ToolResult.success(
                text,
                data=[
                    {
                        "chunk_id": hit.chunk.id,
                        "document_id": hit.chunk.document_id,
                        "score": hit.score,
                    }
                    for hit in hits
                ],
            )

        return Tool.from_callable(retrieve, name=name)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have equal dimensions.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
