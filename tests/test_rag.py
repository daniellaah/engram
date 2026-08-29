from collections.abc import Sequence

from engram.rag import Document, KnowledgeBase, LexicalIndex, TextChunker, VectorIndex


class TinyEmbeddings:
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[float(text.lower().count("python")), float(len(text))] for text in texts]


def test_knowledge_base_chunks_and_retrieves() -> None:
    knowledge = KnowledgeBase(
        index=LexicalIndex(),
        chunker=TextChunker(max_chars=40, overlap_chars=5),
    )
    count = knowledge.add(
        [
            Document(content="Python is a programming language. It emphasizes readability."),
            Document(content="Coffee is brewed from roasted beans."),
        ]
    )

    hits = knowledge.search("Python language", limit=2)

    assert count >= 2
    assert hits
    assert "Python" in hits[0].chunk.content


def test_vector_index_uses_pluggable_embeddings() -> None:
    index = VectorIndex(TinyEmbeddings())
    knowledge = KnowledgeBase(index=index)
    knowledge.add_texts(["Python tooling", "Coffee notes"])

    hits = knowledge.search("Python", limit=1)

    assert hits[0].chunk.content == "Python tooling"


def test_retrieval_tool_returns_sources() -> None:
    knowledge = KnowledgeBase()
    knowledge.add_texts(["Engram keeps agent context compact."])

    result = knowledge.as_tool().invoke({"query": "agent context", "limit": 3})

    assert "context compact" in result.content
    assert result.data[0]["document_id"]
