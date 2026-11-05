
from langchain_core.messages import HumanMessage
from src.rag.graph_builder import builder


def run_test(name, question, user_id="test_user"):
    print(f"\n{'=' * 20} {name} {'=' * 20}")

    try:
        result = builder.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "user_id": user_id,
            }
        )

        print("Route:", result.get("route"))
        print("Retrieval attempts:", result.get("retrieval_attempts"))
        print("Context:", (result.get("context") or "")[:500])

        messages = result.get("messages", [])
        if messages:
            print("\nFinal response:")
            print(messages[-1].content[:1500])

    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    run_test(
        "GENERAL ROUTING",
        "What is the difference between supervised and unsupervised learning?",
    )

    run_test(
        "DOCUMENT ROUTING",
        "Explain the contents of my uploaded document.",
    )

    run_test(
        "WEB ROUTING",
        "Search the web for the latest developments in AI.",
    )
