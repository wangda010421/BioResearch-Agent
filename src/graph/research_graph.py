from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from llm.deepseek import get_llm


class ResearchState(TypedDict):
    question: str
    needs_search: bool
    search_query: str
    papers: list
    answer: str


llm = get_llm()


def researcher_node(state: ResearchState):
    print("\n[Researcher Node]")
    print("Current question:")
    print(state["question"])

    prompt = f"""
You are a biomedical research assistant.

Answer the following scientific question clearly and concisely.

Question:
{state["question"]}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


graph_builder = StateGraph(ResearchState)

graph_builder.add_node("researcher", researcher_node)

graph_builder.add_edge(START, "researcher")
graph_builder.add_edge("researcher", END)

graph = graph_builder.compile()


if __name__ == "__main__":

    initial_state = {
        "question": "What are recent applications of LLMs in protein design?",
        "answer": ""
    }

    result = graph.invoke(initial_state)

    print("\nGraph finished.")

    print("\nFinal answer:")
    print(result["answer"])
