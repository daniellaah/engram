from engram import LLMClient, ReActAgent, ToolRegistry, web_search


def main() -> None:
    tools = ToolRegistry()
    tools.register(
        "search",
        "Search the web for current information. Input must be a search query.",
        web_search,
    )

    with LLMClient() as llm:
        agent = ReActAgent(llm, tools)
        answer = agent.run("What is the current weather in San Francisco?")
    print(answer)


if __name__ == "__main__":
    main()
