import json

from pydantic import BaseModel

from llm.deepseek import get_llm


class ResearchPlan(BaseModel):
    needs_search: bool
    search_query: str
    reason: str


def plan_research(question: str) -> ResearchPlan:
    """
    Analyze a biomedical research question and decide
    whether PubMed literature search is required.
    """

    llm = get_llm()

    prompt = f"""
You are a biomedical research planning assistant.

Determine whether answering the user's question requires
searching scientific literature.

Return ONLY valid JSON.

The JSON must have exactly these fields:

{{
    "needs_search": true,
    "search_query": "PubMed search query",
    "reason": "reason for the decision"
}}

User question:

{question}
"""

    response = llm.bind(
        response_format={"type": "json_object"}
    ).invoke(prompt)

    data = json.loads(response.content)

    return ResearchPlan(**data)


if __name__ == "__main__":
    question = (
        "What are the recent advances in using "
        "large language models for protein design?"
    )

    result = plan_research(question)

    print("\nResearch Planner Result")
    print("-----------------------")
    print("Needs search:", result.needs_search)
    print("Search query:", result.search_query)
    print("Reason:", result.reason)
