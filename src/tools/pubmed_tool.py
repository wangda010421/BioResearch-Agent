from langchain_core.tools import tool

from tools.pubmed_search import (
    search_pubmed,
    fetch_pubmed_details,
)


@tool
def pubmed_search(query: str) -> str:
    """
    Search PubMed for biomedical scientific literature.

    Use this tool when the user's question requires
    current or evidence-based biomedical literature.

    Args:
        query: A PubMed-compatible search query.

    Returns:
        Relevant PubMed papers including title,
        abstract, journal, PMID and DOI.
    """

    pmids = search_pubmed(
        query,
        max_results=5
    )

    if not pmids:
        return "No PubMed papers found."

    papers = fetch_pubmed_details(pmids)

    results = []

    for i, paper in enumerate(papers, start=1):

        text = f"""
Paper {i}

Title: {paper['title']}

Journal: {paper['journal']}

PMID: {paper['pmid']}

DOI: {paper['doi']}

Abstract:
{paper['abstract']}
"""

        results.append(text)

    return "\n\n".join(results)
