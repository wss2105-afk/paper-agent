import os
import time
import requests

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


def search_papers(query, limit=10, retries=2):
    params = {
        "query": query,
        "fields": FIELDS,
        "limit": limit,
    }
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                SEMANTIC_SCHOLAR_URL,
                params=params,
                headers=_get_headers(),
                timeout=15,
            )
            if response.status_code == 429:
                if attempt < retries:
                    time.sleep(3)
                    continue
                raise Exception("API 요청 한도 초과예요. 잠시 후(1~2분) 다시 시도해주세요.")
            if response.status_code == 400:
                raise Exception("검색어를 확인해주세요. 영어 키워드를 권장해요.")
            response.raise_for_status()
            data = response.json()
            papers = data.get("data", [])
            if not papers:
                raise Exception("검색 결과가 없어요. 다른 키워드로 시도해보세요.")
            return papers
        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(2)
                continue
            raise Exception("검색 시간이 초과됐어요. 다시 시도해주세요.")
        except requests.exceptions.ConnectionError:
            raise Exception("네트워크 연결 오류가 발생했어요. 잠시 후 다시 시도해주세요.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"검색 중 오류 발생: {e}")
    raise Exception("검색에 실패했어요. 잠시 후 다시 시도해주세요.")


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
