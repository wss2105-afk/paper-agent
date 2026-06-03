import os
import tempfile
from pathlib import Path

import anthropic
import pdfplumber
import streamlit as st
from dotenv import load_dotenv

from rag import ReferenceLibrary
from scholar import search_papers, format_paper

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")
PDF_DIR = Path(DATA_DIR) / "pdfs"
DB_DIR = Path(DATA_DIR) / "reference_db"
PDF_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """당신은 교육공학 분야의 학술 논문 작성을 전문적으로 돕는 AI 에이전트입니다.

주요 역할:
1. 논문 구조 설계: 서론/이론적 배경/연구방법/결과/논의/결론 구성 안내
2. 문헌 요약/분석: 업로드된 PDF 논문의 핵심 내용 정리
3. 글쓰기 보조: 문장 다듬기, 학술적 표현으로 변환
4. 참고문헌 형식: APA, MLA 등 인용 형식 변환 및 생성
5. 단락 작성: 제공된 참고문헌 내용을 기반으로 학술적 단락 작성

답변 원칙:
- 항상 한국어로 답변
- 학술적이고 정확한 표현 사용
- 구체적인 예시와 함께 설명
- 교육공학 분야 용어와 맥락을 잘 반영
"""


@st.cache_resource
def get_library():
    return ReferenceLibrary(db_path=str(DB_DIR))


def extract_pdf_text(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chat_with_claude(messages):
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


def write_paragraph_with_refs(topic, style, results):
    ref_texts = "\n\n".join(
        f"[출처 {i+1}: {r['source']}]\n{r['text']}"
        for i, r in enumerate(results)
    )
    source_list = "\n".join(f"- {r['source']}" for r in results)
    prompt = f"""아래 참고문헌 내용들을 바탕으로 "{topic}" 주제에 대한 학술적 단락을 작성해주세요.

작성 유형: {style}

[참고문헌 내용]
{ref_texts}

작성 지침:
1. 반드시 제공된 참고문헌 내용만을 근거로 작성하세요.
2. 인용 시 괄호 안에 출처 파일명을 표시하세요. 예: (Smith et al., 2023)
3. 단락은 3~5문장으로 작성하세요.
4. 단락 아래에 "**참고문헌:**" 항목으로 사용한 출처 목록을 나열하세요.
5. 교육공학 분야의 학술적 문체를 사용하세요.

사용된 출처:
{source_list}
"""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ── 페이지 설정 ───────────────────────────────────────────────
st.set_page_config(page_title="논문 작성 도우미", page_icon="📝", layout="wide")
st.title("📝 논문 작성 도우미")
st.caption("교육공학 논문 작성을 위한 AI 에이전트")

library = get_library()

# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.header("기능 선택")
    mode = st.radio(
        "원하는 작업을 선택하세요",
        ["💬 자유 질문", "📚 단락 작성 (RAG)", "🔍 문헌 추천",
         "📄 PDF 분석", "🏗️ 논문 구조 설계", "✍️ 글쓰기 교정", "🔖 참고문헌 변환"],
    )

    st.divider()

    # 참고문헌 라이브러리 관리
    st.subheader("📁 참고문헌 라이브러리")

    saved_pdfs = list(PDF_DIR.glob("*.pdf"))
    if library.is_ready():
        st.success(f"✅ {library.count_papers()}개 논문 학습 완료")
    elif saved_pdfs:
        st.warning(f"⚠️ {len(saved_pdfs)}개 파일 있음 — 학습 필요")
    else:
        st.warning("⚠️ 업로드된 논문 없음")

    # PDF 업로드
    uploaded_files = st.file_uploader(
        "PDF 업로드 (여러 개 가능)",
        type="pdf",
        accept_multiple_files=True,
        help="참고문헌으로 사용할 논문 PDF를 업로드하세요",
    )

    if uploaded_files:
        new_files = []
        for uf in uploaded_files:
            dest = PDF_DIR / uf.name
            if not dest.exists():
                dest.write_bytes(uf.read())
                new_files.append(uf.name)
            else:
                uf.seek(0)
        if new_files:
            st.info(f"{len(new_files)}개 파일 저장됨")

    # 저장된 파일 목록
    saved_pdfs = list(PDF_DIR.glob("*.pdf"))
    if saved_pdfs:
        with st.expander(f"저장된 논문 {len(saved_pdfs)}개 보기"):
            for p in saved_pdfs:
                col1, col2 = st.columns([4, 1])
                col1.caption(p.name[:40])
                if col2.button("🗑️", key=f"del_{p.name}"):
                    p.unlink()
                    st.rerun()

    # 학습 시작 버튼
    if saved_pdfs:
        if st.button("🔄 문헌 학습 시작", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def on_progress(current, total, name):
                progress_bar.progress(current / total)
                status_text.text(f"처리 중: {name[:30]}...")

            with st.spinner("PDF 분석 중..."):
                indexed, errors = library.index_folder(str(PDF_DIR), on_progress)

            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ {len(indexed)}개 논문 학습 완료!")
            if errors:
                with st.expander(f"⚠️ 오류 {len(errors)}건"):
                    for e in errors:
                        st.text(e)
            st.rerun()

    # 전체 초기화
    st.divider()
    if st.button("🗑️ 전체 초기화 (PDF + 인덱스)", use_container_width=True, type="secondary"):
        st.session_state.confirm_reset = True

    if st.session_state.get("confirm_reset"):
        st.warning("정말 삭제할까요? 모든 PDF와 학습 데이터가 사라져요.")
        col1, col2 = st.columns(2)
        if col1.button("✅ 확인", use_container_width=True):
            import shutil
            for f in PDF_DIR.glob("*.pdf"):
                f.unlink()
            if DB_DIR.exists():
                shutil.rmtree(DB_DIR)
                DB_DIR.mkdir(parents=True, exist_ok=True)
            get_library.clear()
            st.session_state.confirm_reset = False
            st.rerun()
        if col2.button("❌ 취소", use_container_width=True):
            st.session_state.confirm_reset = False
            st.rerun()

    st.divider()
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

# ── 세션 초기화 ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 단락 작성 (RAG) ───────────────────────────────────────────
if mode == "📚 단락 작성 (RAG)":
    st.subheader("📚 참고문헌 기반 단락 작성")

    if not library.is_ready():
        st.warning("먼저 사이드바에서 PDF를 업로드하고 '문헌 학습 시작'을 눌러주세요.")
    else:
        topic = st.text_input(
            "작성할 주제를 입력하세요",
            placeholder="예: 블렌디드 러닝이 학습 동기에 미치는 영향",
        )
        col1, col2 = st.columns(2)
        with col1:
            style = st.selectbox(
                "단락 유형",
                ["이론적 배경", "서론", "선행연구 검토", "논의", "결론"],
            )
        with col2:
            top_k = st.slider("참고할 논문 수", 3, 8, 5)

        if st.button("✍️ 단락 작성", use_container_width=True, disabled=not topic):
            with st.spinner("관련 문헌 검색 중..."):
                results = library.search(topic, top_k=top_k)

            if not results:
                st.error("관련 문헌을 찾지 못했어요. 다른 키워드로 시도해보세요.")
            else:
                with st.expander(f"🔍 검색된 참고문헌 {len(results)}개", expanded=False):
                    for r in results:
                        st.markdown(f"**{r['source']}** (관련도: {r['score']:.2f})")
                        st.caption(r["text"][:200] + "...")
                        st.divider()

                with st.spinner("단락 작성 중..."):
                    paragraph = write_paragraph_with_refs(topic, style, results)

                st.markdown("### 작성된 단락")
                st.markdown(paragraph)
                st.download_button(
                    "📋 텍스트로 저장",
                    paragraph,
                    file_name=f"{topic[:20]}_단락.txt",
                    mime="text/plain",
                )

# ── 문헌 추천 ────────────────────────────────────────────────
elif mode == "🔍 문헌 추천":
    st.subheader("🔍 Semantic Scholar 문헌 추천")
    st.caption("연구 주제를 입력하면 관련 논문을 검색하고 Claude가 추천해드려요.")

    topic = st.text_input(
        "연구 주제 또는 키워드 입력",
        placeholder="예: blended learning student motivation",
        help="영어 키워드로 입력하면 검색 결과가 더 풍부해요",
    )
    col1, col2 = st.columns(2)
    with col1:
        num_results = st.slider("검색 논문 수", 5, 20, 10)
    with col2:
        my_topic = st.text_input("내 연구 주제 (선택)", placeholder="Claude 추천 기준으로 사용")

    if st.button("🔍 문헌 검색 및 추천", use_container_width=True, disabled=not topic):
        with st.spinner("Semantic Scholar 검색 중..."):
            try:
                raw_papers = search_papers(topic, limit=num_results)
                papers = [format_paper(p) for p in raw_papers]
            except Exception as e:
                st.error(f"❌ {e}")
                st.info("💡 팁: 키워드를 영어로 입력하면 결과가 더 잘 나와요.\n예) blended learning, flipped classroom, AI tutoring")
                papers = []

        if papers:
            # Claude에게 추천 분석 요청
            paper_list = "\n\n".join(
                f"[{i+1}] {p['title']}\n"
                f"저자: {p['authors']} ({p['year']}) | 인용: {p['citations']}회\n"
                f"초록: {p['abstract']}"
                for i, p in enumerate(papers)
            )
            research_context = f"\n내 연구 주제: {my_topic}" if my_topic else ""
            prompt = f"""다음 논문 목록을 분석하여 연구에 유용한 순서로 추천해주세요.{research_context}

검색 키워드: {topic}

[논문 목록]
{paper_list}

각 논문에 대해:
1. 핵심 내용 한 줄 요약
2. 연구에서 어떻게 활용할 수 있는지
3. 추천 여부 (⭐ 강추 / 👍 참고 / 💡 선택적)

마지막에 가장 중요한 논문 3편을 선정하고 이유를 설명해주세요."""

            with st.spinner("Claude가 논문 분석 중..."):
                recommendation = chat_with_claude([{"role": "user", "content": prompt}])

            st.markdown("### Claude 추천 분석")
            st.markdown(recommendation)

            st.divider()
            st.markdown("### 📄 검색된 논문 전체 목록")
            for i, p in enumerate(papers):
                with st.expander(f"{i+1}. {p['title']} ({p['year']})"):
                    st.markdown(f"**저자:** {p['authors']}")
                    st.markdown(f"**인용 횟수:** {p['citations']}회")
                    st.markdown(f"**초록:** {p['abstract']}")
                    if p["url"]:
                        st.markdown(f"**링크:** [{p['url']}]({p['url']})")

# ── PDF 분석 ─────────────────────────────────────────────────
elif mode == "📄 PDF 분석":
    uploaded_file = st.file_uploader("분석할 논문 PDF를 업로드하세요", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        with st.spinner("PDF 읽는 중..."):
            pdf_text = extract_pdf_text(tmp_path)
        os.unlink(tmp_path)
        st.success(f"PDF 로드 완료 ({len(pdf_text)}자)")
        analyze_option = st.selectbox(
            "분석 유형 선택",
            ["핵심 내용 요약", "연구 방법 분석", "이론적 배경 정리", "연구 결과 요약", "비판적 검토"],
        )
        if st.button("분석 시작"):
            prompt = f"다음 논문을 '{analyze_option}' 관점에서 분석해주세요:\n\n{pdf_text[:8000]}"
            st.session_state.messages.append({"role": "user", "content": prompt})

# ── 논문 구조 설계 ────────────────────────────────────────────
elif mode == "🏗️ 논문 구조 설계":
    st.subheader("논문 구조 설계")
    topic = st.text_input("연구 주제를 입력하세요",
                          placeholder="예: AI 튜터링 시스템이 학습 동기에 미치는 영향")
    research_type = st.selectbox("연구 유형",
                                 ["양적 연구", "질적 연구", "혼합 연구", "문헌 연구", "개발 연구"])
    if st.button("구조 설계 시작") and topic:
        prompt = f"다음 연구 주제로 {research_type} 논문 구조를 설계해주세요.\n\n주제: {topic}\n\n각 섹션별 핵심 내용과 작성 전략을 포함해주세요."
        st.session_state.messages.append({"role": "user", "content": prompt})

# ── 글쓰기 교정 ───────────────────────────────────────────────
elif mode == "✍️ 글쓰기 교정":
    st.subheader("글쓰기 교정")
    text_input = st.text_area("교정할 문장/문단을 입력하세요", height=200)
    correction_type = st.selectbox(
        "교정 유형", ["학술체로 변환", "문장 명확성 개선", "논리 흐름 개선", "전체 교정"]
    )
    if st.button("교정 시작") and text_input:
        prompt = f"다음 글을 '{correction_type}' 관점에서 교정해주세요. 원문과 수정본을 나란히 보여주고, 수정 이유도 설명해주세요:\n\n{text_input}"
        st.session_state.messages.append({"role": "user", "content": prompt})

# ── 참고문헌 변환 ─────────────────────────────────────────────
elif mode == "🔖 참고문헌 변환":
    st.subheader("참고문헌 형식 변환")
    ref_input = st.text_area("참고문헌 정보를 입력하세요", height=150,
                              placeholder="예: 저자명, 출판연도, 제목, 학술지명, 권호, 페이지")
    ref_format = st.selectbox("변환할 형식", ["APA 7판", "MLA", "Chicago", "한국 학술지 형식"])
    if st.button("변환 시작") and ref_input:
        prompt = f"다음 정보를 {ref_format} 형식의 참고문헌으로 변환해주세요:\n\n{ref_input}"
        st.session_state.messages.append({"role": "user", "content": prompt})

# ── 대화 기록 표시 ────────────────────────────────────────────
if mode in ["💬 자유 질문", "📄 PDF 분석", "🏗️ 논문 구조 설계", "✍️ 글쓰기 교정", "🔖 참고문헌 변환", "🔍 문헌 추천"]:
    for msg in st.session_state.messages:
        display_content = msg["content"]
        if len(display_content) > 500 and msg["role"] == "user":
            display_content = display_content[:300] + "\n\n...(내용 생략)..."
        with st.chat_message(msg["role"]):
            st.write(display_content)

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = chat_with_claude(st.session_state.messages)
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    if mode == "💬 자유 질문":
        user_input = st.chat_input("논문 작성에 대해 무엇이든 물어보세요...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.rerun()
