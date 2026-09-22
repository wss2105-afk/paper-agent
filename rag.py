import json
import pickle
import re
import threading
from pathlib import Path

import pdfplumber
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+|[가-힣]+")


def tokenize(text):
    """검색용 토큰화. 한국어는 글자 2-gram으로 쪼개 조사 차이(러닝이/러닝을)에도
    매칭되게 하고, 영문·숫자는 단어 단위로 처리한다. (BM25 recall 향상)"""
    tokens = []
    for m in _TOKEN_RE.findall(text.lower()):
        if "가" <= m[0] <= "힣":  # 한글 구간 → 글자 bigram (+ 단일글자는 그대로)
            if len(m) == 1:
                tokens.append(m)
            else:
                tokens.extend(m[i:i + 2] for i in range(len(m) - 1))
        else:
            tokens.append(m)
    return tokens


class ReferenceLibrary:
    def __init__(self, db_path="./reference_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        self.index_file = self.db_path / "index.pkl"
        self.metadata_file = self.db_path / "metadata.json"
        self.manifest_file = self.db_path / "manifest.json"  # 색인을 '시도한' 파일 목록
        self.documents = []
        self.metadata = []
        self._doc_tokens = []
        self.bm25 = None
        self.manifest = set()
        self._lock = threading.Lock()  # 동시 재색인 방지 (여러 세션이 동시에 열어도 1회만)
        self.indexing = False
        self.progress = None            # 색인 중 (처리한 수, 전체 수, 현재 파일명) — 화면 폴링용
        self.last_result = None         # 마지막 색인 결과 (indexed, errors)
        self._load()

    def _load(self):
        if self.index_file.exists() and self.metadata_file.exists():
            with open(self.index_file, "rb") as f:
                data = pickle.load(f)
                self.documents = data.get("documents", [])
                # 저장된 토큰이 아니라 원문에서 현재 토크나이저로 재생성 →
                # 토크나이저를 개선하면 재색인 없이도 다음 로드에 바로 반영됨.
                if self.documents:
                    self._doc_tokens = [tokenize(d) for d in self.documents]
                    self.bm25 = BM25Okapi(self._doc_tokens)
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, "r", encoding="utf-8") as f:
                    self.manifest = set(json.load(f).get("files", []))
            except Exception:
                self.manifest = self.indexed_filenames()
        else:
            # 구버전 색인(manifest 없음) → 성공한 파일 목록으로 대체
            self.manifest = self.indexed_filenames()

    def _save(self):
        with open(self.index_file, "wb") as f:
            pickle.dump({"documents": self.documents}, f)
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump({"files": sorted(self.manifest)}, f, ensure_ascii=False, indent=2)

    def index_folder(self, folder_path, progress_callback=None):
        """폴더의 PDF를 전부 다시 색인. 진행 중인 색인이 있으면 새로 시작하지 않는다.
        결과는 로컬 변수에 모아 마지막에 한 번에 교체 → 색인 도중에도 search()가 안전."""
        if not self._lock.acquire(blocking=False):
            return [], ["이미 색인이 진행 중이에요"]
        self.indexing = True
        try:
            pdf_files = list(Path(folder_path).glob("**/*.pdf"))
            documents, metadata, indexed, errors = [], [], [], []

            for i, pdf_path in enumerate(pdf_files):
                try:
                    text = self._extract_text(pdf_path)
                    if not text.strip():
                        errors.append(f"{pdf_path.name}: 텍스트 추출 실패")
                        continue
                    chunks = self._chunk(text)
                    for j, chunk in enumerate(chunks):
                        documents.append(chunk)
                        metadata.append({
                            "source": pdf_path.stem,
                            "filename": pdf_path.name,
                            "path": str(pdf_path),
                            "chunk_id": j,
                        })
                    indexed.append(pdf_path.stem)
                except Exception as e:
                    errors.append(f"{pdf_path.name}: {e}")

                self.progress = (i + 1, len(pdf_files), pdf_path.name)
                if progress_callback:
                    progress_callback(i + 1, len(pdf_files), pdf_path.name)

            doc_tokens = [tokenize(d) for d in documents]
            # 문서가 없으면 bm25=None으로 비우고도 저장 → 파일을 다 지운 뒤 옛 색인이 남지 않게
            bm25 = BM25Okapi(doc_tokens) if documents else None
            # 실패한 파일도 manifest에 넣어야 needs_reindex()가 매번 True로 남지 않음
            self.documents, self.metadata, self._doc_tokens, self.bm25 = documents, metadata, doc_tokens, bm25
            self.manifest = {p.name for p in pdf_files}
            self._save()
            self.last_result = (indexed, errors)
            return indexed, errors
        finally:
            self.indexing = False
            self.progress = None
            self._lock.release()

    def _extract_text(self, pdf_path):
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:30]:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def _chunk(self, text, chunk_size=600, overlap=100):
        words = text.split()
        chunks = []
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + chunk_size])
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
        return chunks

    def search(self, query, top_k=5, per_source=1):
        """상위 top_k개 논문에서 관련 청크를 찾는다.
        per_source>1이면 논문당 최대 그만큼의 청크를 원문 순서로 이어붙여
        더 넓은 문맥을 제공한다(단락 작성 품질 향상용)."""
        if not self.bm25 or not self.documents:
            return []

        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        if len(scores) and max(scores) <= 0:
            # 청크가 아주 적을 때(논문 1편 등) BM25 IDF가 0 이하로 떨어져 점수가 전부 0이 됨
            # → 질의 토큰 일치 개수로 대체 (색인된 내 논문이 한 편이어도 검색되게)
            qset = set(q_tokens)
            # '에서' 같은 흔한 글자쌍 하나로 매칭되는 걸 막기 위해 최소 일치 수 요구
            min_hits = max(2, -(-len(qset) * 3 // 10))  # ceil(30%)
            hits = [len(qset & set(toks)) for toks in self._doc_tokens]
            scores = [h if h >= min_hits else 0 for h in hits]

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # 논문당 최고 점수 청크(들) 선택 — dict 삽입 순서로 최고점 우선 유지
        seen = {}
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            source = self.metadata[idx]["source"]
            if source not in seen:
                if len(seen) >= top_k:
                    continue
                seen[source] = {
                    "chunks": [(self.metadata[idx]["chunk_id"], self.documents[idx])],
                    "source": source,
                    "filename": self.metadata[idx]["filename"],
                    "score": scores[idx],
                }
            elif len(seen[source]["chunks"]) < per_source:
                seen[source]["chunks"].append(
                    (self.metadata[idx]["chunk_id"], self.documents[idx]))

        results = []
        for item in seen.values():
            # 논문 안에서는 원문 순서(chunk_id)로 이어붙여 읽기 흐름 유지
            chunks = sorted(item.pop("chunks"))
            item["text"] = "\n(...)\n".join(c for _, c in chunks)
            results.append(item)
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def get_head(self, source, max_chars=700):
        """논문 첫머리(제목·저자·연도·학술지가 보통 담긴 부분)를 반환.
        단락 작성 시 실제 서지 정보 기반 (저자, 연도) 인용에 사용한다."""
        for doc, meta in zip(self.documents, self.metadata):
            if meta["source"] == source and meta["chunk_id"] == 0:
                return doc[:max_chars]
        return ""

    def indexed_filenames(self):
        """색인에 들어 있는 PDF 파일명 집합"""
        return {m["filename"] for m in self.metadata}

    def needs_reindex(self, folder_path):
        """폴더의 PDF 목록과 마지막에 색인을 시도한 파일 목록이 다르면 True (추가·삭제 감지).
        색인 진행 중이면 False (중복 시작 방지)."""
        if self.indexing:
            return False
        current = {p.name for p in Path(folder_path).glob("**/*.pdf")}
        return current != self.manifest

    def count_papers(self):
        sources = {m["source"] for m in self.metadata}
        return len(sources)

    def is_ready(self):
        return self.bm25 is not None and len(self.documents) > 0
