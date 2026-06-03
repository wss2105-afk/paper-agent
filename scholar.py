import os
import time
import requests
import arxiv as arxiv_lib

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,authors,year,abstract,citationCount,url,externalIds"


def _get_headers():
    headers = {
        "User-Agent": "PaperAgent/1.0 (educational research tool)",
        "Accept": "application/json",
    }
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def search_semantic_scholar(query, limit=10, retries=2):
    params = {"query": query, "fields": FIELDS, "limit": limit}
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                SEMANTIC_SCHOLAR_URL, params=params, headers=_get_headers(), timeout=15
            )
            if response.status_code == 429:
                if attempt < retries:
                    time.sleep(3)
                    continue
                raise Exception("Semantic Scholar API 한도 초과. 잠시 후 다시 시도해주세요.")
            if response.status_code == 400:
                raise Exception("검색어를 확인해주세요.")
            response.raise_for_status()
            papers = response.json().get("data", [])
            return [format_paper(p) for p in papers]
        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(2)
                continue
            raise Exception("Semantic Scholar 검색 시간 초과.")
        except requests.exceptions.ConnectionError:
            raise Exception("네트워크 연결 오류.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"검색 오류: {e}")
    raise Exception("검색 실패. 잠시 후 다시 시도해주세요.")


def search_arxiv(query, limit=10):
    try:
        client = arxiv_lib.Client()
        search = arxiv_lib.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv_lib.SortCriterion.Relevance,
        )
        results = list(client.results(search))
        papers = []
        for r in results:
            authors = ", ".join(a.name for a in r.authors[:3])
            if len(r.authors) > 3:
                authors += " et al."
            abstract = r.summary or "초록 없음"
            papers.append({
                "title": r.title,
                "authors": authors,
                "year": r.published.year if r.published else "연도 미상",
                "abstract": abstract[:500] + ("..." if len(abstract) > 500 else ""),
                "citations": "N/A",
                "url": r.entry_id,
                "source": "arXiv",
            })
        return papers
    except Exception as e:
        raise Exception(f"arXiv 검색 오류: {e}")


def search_papers(query, limit=10, source="Semantic Scholar"):
    if source == "Semantic Scholar":
        return search_semantic_scholar(query, limit)
    elif source == "arXiv":
        return search_arxiv(query, limit)
    else:  # 둘 다
        half = max(limit // 2, 3)
        results = []
        try:
            results += search_semantic_scholar(query, half)
        except Exception:
            pass
        try:
            results += search_arxiv(query, half)
        except Exception:
            pass
        if not results:
            raise Exception("검색 결과가 없어요. 다른 키워드로 시도해주세요.")
        return results


def format_paper(paper):
    """Semantic Scholar raw API 응답을 정형화"""
    authors = ", ".join(a.get("name", "") for a in paper.get("authors", [])[:3])
    if len(paper.get("authors", [])) > 3:
        authors += " et al."
    abstract = paper.get("abstract") or "초록 없음"
    ext_ids = paper.get("externalIds") or {}
    doi_url = f"https://doi.org/{ext_ids['DOI']}" if ext_ids.get("DOI") else ""
    return {
        "title": paper.get("title", "제목 없음"),
        "authors": authors,
        "year": paper.get("year", "연도 미상"),
        "abstract": abstract[:500] + ("..." if len(abstract) > 500 else ""),
        "citations": paper.get("citationCount", 0),
        "url": doi_url or paper.get("url", ""),
        "source": "Semantic Scholar",
    }
