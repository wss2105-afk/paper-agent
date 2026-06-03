import requests

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,authors,year,abstract,citationCount,url,externalIds"


def search_papers(query, limit=10):
    params = {
        "query": query,
        "fields": FIELDS,
        "limit": limit,
    }
    try:
        response = requests.get(SEMANTIC_SCHOLAR_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except requests.exceptions.Timeout:
        raise Exception("검색 시간이 초과됐어요. 다시 시도해주세요.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"검색 중 오류 발생: {e}")


def format_paper(paper):
    authors = ", ".join(a.get("name", "") for a in paper.get("authors", [])[:3])
    if len(paper.get("authors", [])) > 3:
        authors += " et al."
    year = paper.get("year", "연도 미상")
    title = paper.get("title", "제목 없음")
    abstract = paper.get("abstract") or "초록 없음"
    citations = paper.get("citationCount", 0)
    url = paper.get("url", "")

    doi = ""
    ext_ids = paper.get("externalIds", {})
    if ext_ids and ext_ids.get("DOI"):
        doi = f"https://doi.org/{ext_ids['DOI']}"

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": abstract[:500] + ("..." if len(abstract or "") > 500 else ""),
        "citations": citations,
        "url": doi or url,
    }
