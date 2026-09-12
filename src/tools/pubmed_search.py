import requests
from xml.etree import ElementTree


def search_pubmed(query, max_results=5):
    """
    使用 NCBI PubMed E-utilities 搜索论文
    """

    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }

    response = requests.get(
        search_url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    pmids = data["esearchresult"]["idlist"]

    return pmids
def fetch_pubmed_details(pmids):
    """
    根据 PMID 获取论文详细信息
    """

    if not pmids:
        return []

    fetch_url = (
        "https://eutils.ncbi.nlm.nih.gov/"
        "entrez/eutils/efetch.fcgi"
    )

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }

    response = requests.get(
        fetch_url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    root = ElementTree.fromstring(response.content)

    papers = []

    for article in root.findall(".//PubmedArticle"):

        # 标题
        title_node = article.find(".//ArticleTitle")
        title = (
            "".join(title_node.itertext())
            if title_node is not None
            else ""
        )

        # 摘要
        abstract_parts = []

        for node in article.findall(".//Abstract/AbstractText"):
            text = "".join(node.itertext())

            label = node.attrib.get("Label")

            if label:
                text = f"{label}: {text}"

            abstract_parts.append(text)

        abstract = " ".join(abstract_parts)

        # 作者
        authors = []

        for author in article.findall(".//Author"):
            last = author.findtext("LastName", "")
            fore = author.findtext("ForeName", "")

            name = f"{fore} {last}".strip()

            if name:
                authors.append(name)

        # 期刊
        journal = article.findtext(
            ".//Journal/Title",
            ""
        )

        # PMID
        pmid = article.findtext(
            ".//PMID",
            ""
        )

        # DOI
        doi = ""

        for article_id in article.findall(".//ArticleId"):
            if article_id.attrib.get("IdType") == "doi":
                doi = article_id.text or ""
                break

        paper = {
            "pmid": pmid,
            "title": title,
            "authors": authors,
            "journal": journal,
            "abstract": abstract,
            "doi": doi,
        }

        papers.append(paper)

    return papers

if __name__ == "__main__":

    test_query = (
        '"large language model"[Title/Abstract] '
        'AND "protein design"[Title/Abstract]'
    )

    pmids = search_pubmed(
        test_query,
        max_results=5
    )


    print("\n找到 PMID:")
    print(pmids)

    papers = fetch_pubmed_details(pmids)

    print("\n找到论文数量:", len(papers))

    for i, paper in enumerate(papers, start=1):

        print("\n" + "=" * 70)
        print(f"Paper {i}")
        print("=" * 70)

        print("PMID:")
        print(paper["pmid"])

        print("\nTitle:")
        print(paper["title"])

        print("\nAuthors:")
        print(", ".join(paper["authors"][:5]))

        print("\nJournal:")
        print(paper["journal"])

        print("\nDOI:")
        print(paper["doi"])

        print("\nAbstract:")
        print(paper["abstract"][:500])
