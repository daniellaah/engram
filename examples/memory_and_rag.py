from engram import Document, KnowledgeBase, LLMClient, MemoryManager, SimpleAgent


def main() -> None:
    memory = MemoryManager()
    memory.remember("The user prefers concise technical explanations.", importance=0.9)

    knowledge = KnowledgeBase()
    knowledge.add(
        [
            Document(content="Engram executes local tools through native function calls."),
            Document(content="Engram keeps retrieved context separate from conversation history."),
        ]
    )

    with LLMClient() as llm:
        agent = SimpleAgent(llm, tools=[memory.as_tool(), knowledge.as_tool()])
        print(agent.run("Explain how the framework handles tools and context."))


if __name__ == "__main__":
    main()
