from engram import LLMClient, ReActAgent, Tool


def get_weather(city: str) -> str:
    """Return sample weather data for a city."""
    return f"The weather in {city} is sunny and 20 degrees Celsius."


def main() -> None:
    with LLMClient() as llm:
        agent = ReActAgent(llm, [Tool.from_callable(get_weather)])
        print(agent.run("Should I take an umbrella in Seattle?"))


if __name__ == "__main__":
    main()
