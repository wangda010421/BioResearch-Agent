import json

from llm.deepseek import get_llm


def rerank_papers(question: str, papers: list, top_k: int = 5) -> list:
    """
    Rank PubMed papers by relevance to the user's research question.
    """

    llm = get_llm()

    scored_papers = []

    for paper in papers:

        prompt = f"""
You are evaluating whether a scientific paper is relevant
to a biomedical research question.

Return ONLY valid JSON.

Question:
{question}

Paper title:
{paper.get("title", "")}

Paper abstract:
{paper.get("abstract", "")}

Return exactly:

{{
    "score": 0,
    "reason": "short explanation"
}}

Score relevance from 0 to 10:

0 = completely irrelevant
5 = partially relevant
10 = directly answers the question
"""

        response = llm.bind(
            response_format={"type": "json_object"}
        ).invoke(prompt)

        result = json.loads(response.content)

        score = float(result["score"])

        paper["relevance_score"] = score
        paper["relevance_reason"] = result["reason"]

        scored_papers.append(paper)

    scored_papers.sort(
        key=lambda x: x["relevance_score"],
        reverse=True
    )

    return scored_papers[:top_k]
