import os
import re
import time
import requests
import arxiv as arxiv_lib

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,authors,year,abstract,citationCount,url,externalIds,venue,journal,publicationVenue"
OPENALEX_URL = "https://api.openalex.org/works"
OPENALEX_SELECT = ("id,doi,title,display_name,publication_year,cited_by_count,"
                   "authorships,primary_location,abstract_inverted_index")


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
    # Semantic Scholar는 따옴표 구문을 지원하지 않으므로 제거
    params = {"query": query.replace('"', " "), "fields": FIELDS, "limit": limit}
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


def _abstract_from_inverted(inv):
    """OpenAlex의 abstract_inverted_index(단어→위치 목록)를 원문 문자열로 복원"""
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def search_openalex(query, limit=10, retries=2):
    """OpenAlex 검색 — API 키 불필요, 요청 제한이 관대해 기본 소스로 적합.
    전문(full-text) 검색은 인용 가중치가 커서 무관한 유명 논문이 상위에 오므로,
    제목+초록 한정 검색으로 주제 적합 풀을 만든 뒤 관련도순·인용순을 섞는다."""
    q = re.sub(r"[,:]", " ", query).strip()  # 콤마·콜론은 filter 문법과 충돌
    params = {
        "filter": f"title_and_abstract.search:{q},type:article",
        "per-page": 25,
        "select": OPENALEX_SELECT,
        "mailto": "wss2105@gmail.com",  # polite pool → 더 안정적인 응답
    }
    for attempt in range(retries + 1):
        try:
            response = requests.get(OPENALEX_URL, params=params,
                                    headers=_get_headers(), timeout=15)
            if response.status_code == 429:
                if attempt < retries:
                    time.sleep(2)
                    continue
                raise Exception("OpenAlex API 한도 초과. 잠시 후 다시 시도해주세요.")
            response.raise_for_status()
            # 관련도순(기본 정렬) 그대로 사용 — 인용순을 섞으면 경계선 매치의
            # 유명 논문이 끼어들어 주제 적합성이 떨어진다 (영향력 판단은 하류의
            # Claude 재정렬·SSCI 모드가 담당)
            works = response.json().get("results", [])
            if not works and '"' in q:
                # 따옴표 구문 검색이 과하게 좁으면 따옴표를 풀고 한 번 더
                params_loose = dict(params)
                params_loose["filter"] = (
                    f"title_and_abstract.search:{q.replace(chr(34), ' ')},type:article")
                loose = requests.get(OPENALEX_URL, params=params_loose,
                                     headers=_get_headers(), timeout=15)
                if loose.ok:
                    works = loose.json().get("results", [])
            papers = []
            for w in works[:limit]:
                authorships = w.get("authorships") or []
                names = [a.get("author", {}).get("display_name", "")
                         for a in authorships[:3]]
                authors = ", ".join(n for n in names if n)
                if len(authorships) > 3:
                    authors += " et al."
                abstract = _abstract_from_inverted(
                    w.get("abstract_inverted_index")) or "초록 없음"
                src = (w.get("primary_location") or {}).get("source") or {}
                papers.append({
                    "title": w.get("display_name") or w.get("title") or "제목 없음",
                    "authors": authors or "저자 미상",
                    "year": w.get("publication_year") or "연도 미상",
                    "abstract": abstract[:500] + ("..." if len(abstract) > 500 else ""),
                    "citations": w.get("cited_by_count", 0),
                    "url": w.get("doi") or w.get("id") or "",
                    "journal": src.get("display_name") or "",
                    "source": "OpenAlex",
                })
            return papers
        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(2)
                continue
            raise Exception("OpenAlex 검색 시간 초과.")
        except requests.exceptions.ConnectionError:
            raise Exception("네트워크 연결 오류.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"OpenAlex 검색 오류: {e}")
    raise Exception("OpenAlex 검색 실패. 잠시 후 다시 시도해주세요.")


def _norm_title(title):
    return re.sub(r"[^a-z0-9가-힣]", "", str(title).lower())


def dedup_papers(papers):
    """DOI → 정규화 제목 순으로 중복 제거 (앞선 항목 우선)"""
    seen, out = set(), []
    for p in papers:
        keys = []
        url = str(p.get("url") or "").lower()
        if "doi.org" in url:
            keys.append(url)
        tkey = _norm_title(p.get("title", ""))
        if tkey:
            keys.append("t:" + tkey)
        if any(k in seen for k in keys):
            continue
        seen.update(keys)
        out.append(p)
    return out


def interleave(lists):
    """여러 결과 목록을 라운드로빈으로 섞어 소스·검색어 다양성 유지"""
    merged = []
    longest = max((len(l) for l in lists), default=0)
    for i in range(longest):
        for l in lists:
            if i < len(l):
                merged.append(l[i])
    return merged


def search_papers(query, limit=10, source="Semantic Scholar"):
    if source == "Semantic Scholar":
        return search_semantic_scholar(query, limit)
    elif source == "arXiv":
        return search_arxiv(query, limit)
    elif source == "OpenAlex":
        return search_openalex(query, limit)
    else:  # 전체 / 둘 다: OpenAlex + Semantic Scholar 병합
        # arXiv는 교육 분야 주제에서 무관한 CS 프리프린트가 섞여 기본 병합에서 제외
        # (소스 선택에서 arXiv를 직접 고르면 여전히 사용 가능)
        buckets = []
        for fn, n in ((search_openalex, limit),
                      (search_semantic_scholar, max(limit // 2, 3))):
            try:
                buckets.append(fn(query, n))
            except Exception:
                pass
        results = dedup_papers(interleave(buckets))
        if not results:
            raise Exception("검색 결과가 없어요. 다른 키워드로 시도해주세요.")
        return results[:limit]


def format_paper(paper):
    """Semantic Scholar raw API 응답을 정형화"""
    authors = ", ".join(a.get("name", "") for a in paper.get("authors", [])[:3])
    if len(paper.get("authors", [])) > 3:
        authors += " et al."
    abstract = paper.get("abstract") or "초록 없음"
    ext_ids = paper.get("externalIds") or {}
    doi_url = f"https://doi.org/{ext_ids['DOI']}" if ext_ids.get("DOI") else ""

    # 학술지명: publicationVenue → journal.name → venue 순으로 시도
    pub_venue = paper.get("publicationVenue") or {}
    journal_info = paper.get("journal") or {}
    journal_name = (
        pub_venue.get("name")
        or journal_info.get("name")
        or paper.get("venue")
        or ""
    )
    journal_vol = journal_info.get("volume") or ""
    journal_pages = journal_info.get("pages") or ""
    journal_display = journal_name
    if journal_vol:
        journal_display += f", Vol.{journal_vol}"
    if journal_pages:
        journal_display += f", pp.{journal_pages}"

    return {
        "title": paper.get("title", "제목 없음"),
        "authors": authors,
        "year": paper.get("year", "연도 미상"),
        "abstract": abstract[:500] + ("..." if len(abstract) > 500 else ""),
        "citations": paper.get("citationCount", 0),
        "url": doi_url or paper.get("url", ""),
        "journal": journal_display,
        "source": "Semantic Scholar",
    }
