import os
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agents.research_planner import ResearchPlan
from tools.pubmed_search import (
    search_pubmed,
    fetch_pubmed_details,
)


load_dotenv()


# ==============================
# 1. 初始化 DeepSeek
# ==============================

planner_llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
    timeout=30,
    max_retries=1,
    model_kwargs={
        "response_format": {
            "type": "json_object"
        }
    }
)
# Answer 专用：正常文本输出
answer_llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
    timeout=30,
    max_retries=1,
)

# ==============================
# 2. 用户问题
# ==============================

question = """
What are the recent advances in using
large language models for protein design?
"""


# ==============================
# 3. Research Planner
# ==============================

planner_prompt = f"""
You are a biomedical research planning assistant.

Determine whether answering the following question
requires searching scientific literature.

Return ONLY valid JSON.

Format:

{{
    "needs_search": true,
    "search_query": "PubMed query",
    "reason": "reason"
}}

Question:

{question}
"""


print("\n[1/4] Research Planner working...")


response = planner_llm.invoke(planner_prompt)

data = json.loads(response.content)

plan = ResearchPlan(**data)


print("Needs search:", plan.needs_search)
print("PubMed query:", plan.search_query)


# ==============================
# 4. PubMed Search
# ==============================

papers = []

if plan.needs_search:

    print("\n[2/4] Searching PubMed...")

    pmids = search_pubmed(
        plan.search_query,
        max_results=5
    )

    print("PMIDs:", pmids)

    papers = fetch_pubmed_details(pmids)

    print("Papers retrieved:", len(papers))


# ==============================
# 5. 整理文献
# ==============================

literature_text = ""

for i, paper in enumerate(papers, start=1):

    literature_text += f"""
Paper {i}

Title:
{paper['title']}

Journal:
{paper['journal']}

PMID:
{paper['pmid']}

DOI:
{paper['doi']}

Abstract:
{paper['abstract']}

------------------------------
"""


# ==============================
# 6. DeepSeek 文献总结
# ==============================

print("\n[3/4] Analyzing literature...")


answer_prompt = f"""
You are a biomedical research assistant.

Answer the user's question using ONLY the
scientific literature provided below.

Do not invent papers or results.

For important claims, cite the corresponding
paper using [Paper 1], [Paper 2], etc.

User question:

{question}


Scientific literature:

{literature_text}


Provide:

1. A concise overview
2. Major recent advances
3. Important methods or models
4. Current limitations
5. References to the provided papers
"""


answer = answer_llm.invoke(answer_prompt)


# ==============================
# 7. 输出最终答案
# ==============================

print("\n[4/4] Final answer")
print("=" * 70)

print(answer.content)
