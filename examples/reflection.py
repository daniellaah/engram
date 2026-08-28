from engram import LLMClient, ReflectionAgent


def main() -> None:
    task = "Write a Python function that returns all prime numbers up to an integer limit."
    with LLMClient() as llm:
        agent = ReflectionAgent(llm, max_iterations=2)
        solution = agent.run(task)
    print(solution)


if __name__ == "__main__":
    main()
