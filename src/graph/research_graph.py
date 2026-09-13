from agents.reranker import rerank_papers
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from agents.research_planner import plan_research
from llm.deepseek import get_llm
from tools.pubmed_search import search_pubmed, fetch_pubmed_details

# =========================
# 1. 定义 Graph State
# =========================

class ResearchState(TypedDict):
    question: str
    needs_search: bool
    search_query: str
    reason: str
    papers: list   
    answer: str


# =========================
# 2. Planner Node
# =========================
def route_after_planner(state: ResearchState):
    if state["needs_search"]:
        print("\n[Router] Literature search required.")
        return "pubmed"

    print("\n[Router] Literature search not required.")
    return "researcher"
def planner_node(state: ResearchState):
    print("\n[Planner Node]")

    question = state["question"]

    plan = plan_research(question)

    print("Needs search:", plan.needs_search)
    print("Search query:", plan.search_query)
    print("Reason:", plan.reason)

    return {
        "needs_search": plan.needs_search,
        "search_query": plan.search_query,
        "reason": plan.reason,
    }


# =========================
# 3. Researcher Node
# =========================

def researcher_node(state: ResearchState):
    print("\n[Researcher Node]")

    question = state["question"]
    papers = state.get("papers", [])

    llm = get_llm()

    # 如果 PubMed 搜到了论文，就整理成 evidence
    if papers:
        evidence_parts = []

        for i, paper in enumerate(papers, start=1):
            evidence_parts.append(
                f"""
Paper {i}
Title: {paper.get('title', '')}
Authors: {', '.join(paper.get('authors', [])[:5])}
Journal: {paper.get('journal', '')}
PMID: {paper.get('pmid', '')}
DOI: {paper.get('doi', '')}
Abstract:
{paper.get('abstract', '')}
"""
            )

        evidence = "\n".join(evidence_parts)

    else:
        evidence = "No PubMed literature was retrieved."

    prompt = f"""
You are a biomedical research assistant.

Answer the user's question using the PubMed literature provided below.

Question:
{question}

PubMed evidence:
{evidence}

Instructions:

1. Base the scientific claims primarily on the provided PubMed evidence.
2. Do not invent papers, PMID numbers, DOI numbers, authors, or experimental results.
3. If the retrieved evidence is insufficient to support a claim, explicitly say so.
4. Cite supporting papers in the text using [Paper 1], [Paper 2], etc.
5. Clearly distinguish evidence from general background knowledge.
6. At the end, provide a section called "References".
7. In References, list the title, journal, PMID and DOI of the provided papers that were actually used.
8. Answer clearly and scientifically.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# =========================
# 4. 构建 Graph
# =========================
def pubmed_node(state: ResearchState):
    print("\n[PubMed Node]")

    query = state["search_query"]

    print("Search query:", query)

    pmids = search_pubmed(query, max_results=10)

    print("Found PMIDs:", pmids)

    papers = fetch_pubmed_details(pmids)

    print("Fetched papers:", len(papers))

    print("\n[Reranker] Ranking papers...")

    papers = rerank_papers(
        state["question"],
        papers,
        top_k=5
    )

    print("Top ranked papers:")

    for i, paper in enumerate(papers, start=1):
        print(
            f"{i}. Score={paper['relevance_score']:.1f} | "
            f"{paper['title']}"
        )

    return {
        "papers": papers
    }
graph_builder = StateGraph(ResearchState)

graph_builder.add_node("planner", planner_node)
graph_builder.add_node("pubmed", pubmed_node)
graph_builder.add_node("researcher", researcher_node)

graph_builder.add_edge(START, "planner")

graph_builder.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "pubmed": "pubmed",
        "researcher": "researcher",
    },
)
graph_builder.add_edge("pubmed", "researcher")
graph_builder.add_edge("researcher", END)

graph = graph_builder.compile()


# =========================
# 5. 测试
# =========================

if __name__ == "__main__":

    initial_state = {
        "question": "What are recent applications of LLMs in protein design?",
        "needs_search": False,
        "search_query": "",
        "reason": "",
        "papers": [],        
        "answer": "",
    }

    result = graph.invoke(initial_state)

    print("\nGraph finished.")

    print("\nFinal answer:")
    print(result["answer"])
