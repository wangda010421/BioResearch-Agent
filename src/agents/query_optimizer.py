from llm.deepseek import get_llm


def optimize_query(question: str) -> str:
    """
    Convert a natural-language biomedical question
    into a PubMed-friendly search query.
    """

    llm = get_llm()

    prompt = f"""
You are a biomedical literature search expert.

Convert the user's research question into a concise PubMed search query.

Requirements:
1. Extract the core scientific concepts.
2. Include useful synonyms with OR.
3. Combine different concepts with AND.
4. Focus on terms likely to appear in titles or abstracts.
5. Do not answer the research question.
6. Return ONLY the PubMed query.
7. Do not use Markdown or explanations.

User question:
{question}
"""

    response = llm.invoke(prompt)

    query = response.content.strip()

    return query


if __name__ == "__main__":

    question = "What are recent applications of LLMs in protein design?"

    query = optimize_query(question)

    print("\nOriginal question:")
    print(question)

    print("\nOptimized PubMed query:")
    print(query)
