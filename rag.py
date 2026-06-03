import json
import pickle
from pathlib import Path

import pdfplumber
from rank_bm25 import BM25Okapi


class ReferenceLibrary:
    def __init__(self, db_path="./reference_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        self.index_file = self.db_path / "index.pkl"
        self.metadata_file = self.db_path / "metadata.json"
        self.documents = []
        self.metadata = []
        self.bm25 = None
        self._load()

    def _load(self):
        if self.index_file.exists() and self.metadata_file.exists():
            with open(self.index_file, "rb") as f:
                data = pickle.load(f)
                self.documents = data.get("documents", [])
                tokenized = data.get("tokenized", [])
                if tokenized:
                    self.bm25 = BM25Okapi(tokenized)
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def _save(self):
        tokenized = [doc.lower().split() for doc in self.documents]
        with open(self.index_file, "wb") as f:
            pickle.dump({"documents": self.documents, "tokenized": tokenized}, f)
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def index_folder(self, folder_path, progress_callback=None):
        self.documents = []
        self.metadata = []
        pdf_files = list(Path(folder_path).glob("**/*.pdf"))
        indexed, errors = [], []

        for i, pdf_path in enumerate(pdf_files):
            try:
                text = self._extract_text(pdf_path)
                if not text.strip():
                    errors.append(f"{pdf_path.name}: 텍스트 추출 실패")
                    continue
                chunks = self._chunk(text)
                for j, chunk in enumerate(chunks):
                    self.documents.append(chunk)
                    self.metadata.append({
                        "source": pdf_path.stem,
                        "filename": pdf_path.name,
                        "path": str(pdf_path),
                        "chunk_id": j,
                    })
                indexed.append(pdf_path.stem)
            except Exception as e:
                errors.append(f"{pdf_path.name}: {e}")

            if progress_callback:
                progress_callback(i + 1, len(pdf_files), pdf_path.name)

        if self.documents:
            tokenized = [doc.lower().split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized)
            self._save()

        return indexed, errors

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

    def search(self, query, top_k=5):
        if not self.bm25 or not self.documents:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # 논문당 가장 관련 높은 청크 하나씩만 선택
        seen = {}
        for idx in top_indices:
            source = self.metadata[idx]["source"]
            if source not in seen and scores[idx] > 0:
                seen[source] = {
                    "text": self.documents[idx],
                    "source": source,
                    "filename": self.metadata[idx]["filename"],
                    "score": scores[idx],
                }
            if len(seen) >= top_k:
                break

        return sorted(seen.values(), key=lambda x: x["score"], reverse=True)

    def count_papers(self):
        sources = {m["source"] for m in self.metadata}
        return len(sources)

    def is_ready(self):
        return self.bm25 is not None and len(self.documents) > 0
