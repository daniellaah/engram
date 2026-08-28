from engram import LLMClient, PlanAndSolveAgent


def main() -> None:
    task = (
        "A store has 120 notebooks. It sells 35% of them and then receives "
        "18 more. How many notebooks does it have now?"
    )
    with LLMClient() as llm:
        agent = PlanAndSolveAgent(llm)
        answer = agent.run(task)
    print(answer)


if __name__ == "__main__":
    main()
