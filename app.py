import html as _html
import os
import tempfile
from pathlib import Path

import anthropic
import pdfplumber
import streamlit as st
from dotenv import load_dotenv

import streamlit.components.v1 as components
from rag import ReferenceLibrary
from scholar import search_papers
from export import to_word, to_markdown
from data_analyzer import load_file, summarize_dataframe, summarize_interview, get_preview, get_basic_stats
from stats_runner import (
    run_ttest_ind, run_ttest_rel, run_anova, run_correlation,
    run_chisquare, run_cronbach, run_regression, run_hlm,
    run_lca, run_sem, run_sem_multigroup, SEM_EXAMPLE,
)
from style_analyzer import (
    load_my_papers, build_style_prompt, save_style_profile,
    load_style_profile, build_style_instruction
)
from sjr_data import quartile_badge

load_dotenv()

# 저장 경로 결정 우선순위:
#   1) RAILWAY_VOLUME_MOUNT_PATH (볼륨 붙으면 Railway가 자동 주입) — 볼륨이 있으면 무조건 여기 저장
#   2) DATA_DIR (직접 지정)
#   3) ./data (로컬 기본값)
# 볼륨 경로를 최우선으로 둔다. 과거 DATA_DIR=./data 로 잘못 설정돼 볼륨을 두고도
# 임시 디스크에 저장 → 재배포마다 데이터 소실되던 사고 재발 방지.
DATA_DIR = os.getenv("RAILWAY_VOLUME_MOUNT_PATH") or os.getenv("DATA_DIR") or "./data"
PDF_DIR        = Path(DATA_DIR) / "pdfs"
DB_DIR         = Path(DATA_DIR) / "reference_db"
MY_PAPERS_DIR  = Path(DATA_DIR) / "my_papers"
STYLE_PROFILE  = Path(DATA_DIR) / "style_profile.json"

for d in [PDF_DIR, DB_DIR, MY_PAPERS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """당신은 교육공학 분야의 학술 논문 작성을 전문적으로 돕는 AI 에이전트입니다.

⛔ 최우선 절대 원칙 — 사실성(정확성)은 완성도보다 언제나 우선합니다:
- 확실하지 않은 사실·통계·수치·인용·저자·연도·논문 제목·출처를 절대 지어내지 마세요.
- 근거가 없거나 모르면, 그럴듯하게 채우지 말고 "확실한 근거가 없습니다" 또는 "제공된 자료에서는 확인할 수 없습니다"라고 솔직히 밝히세요.
- 글을 매끄럽게 완성하는 것보다, 틀린 내용을 쓰지 않는 것이 훨씬 중요합니다. **쓰지 못할지언정 지어내지 않습니다.**
- 참고문헌·인용은 제공된 자료에 실제로 존재하는 것만 사용하고, 존재하지 않는 문헌·저자·연도를 만들어내지 마세요.
- 불가피하게 추정할 때는 반드시 "추정" 또는 "추측"이라고 명시하세요.

주요 역할:
1. 논문 구조 설계: 서론/이론적 배경/연구방법/결과/논의/결론 구성 안내
2. 문헌 요약/분석: 업로드된 PDF 논문의 핵심 내용 정리
3. 글쓰기 보조: 문장 다듬기, 학술적 표현으로 변환
4. 참고문헌 형식: APA, MLA 등 인용 형식 변환 및 생성
5. 단락 작성: 제공된 참고문헌 내용을 기반으로 학술적 단락 작성

답변 원칙:
- 항상 한국어로 답변
- 학술적이고 정확한 표현 사용
- 확인되지 않은 내용은 지어내지 말고 한계를 솔직히 밝힐 것 (위 절대 원칙 준수)
- 구체적인 예시와 함께 설명
- 교육공학 분야 용어와 맥락을 잘 반영
"""


@st.cache_resource
def get_library():
    return ReferenceLibrary(db_path=str(DB_DIR))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_search(query, limit, source, version=5):
    return search_papers(query, limit=limit, source=source)


def extract_pdf_text(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def _call_claude(**kwargs):
    """client.messages.create를 감싸 오류를 사용자 친화적으로 처리하는 공용 함수.
    성공하면 응답 객체를 반환하고, 오류가 나면 안내 메시지를 띄운 뒤 실행을 안전하게 멈춘다."""
    try:
        return client.messages.create(**kwargs)
    except anthropic.RateLimitError:
        st.error("⏳ 지금 요청이 많아 잠시 제한됐어요. 1~2분 후 다시 시도해주세요.")
    except anthropic.APIConnectionError:
        st.error("🌐 AI 서버에 연결하지 못했어요. 인터넷 연결을 확인하고 다시 시도해주세요.")
    except anthropic.APIStatusError as e:
        st.error(f"⚠️ AI 서버 오류가 발생했어요 (코드 {getattr(e, 'status_code', '?')}). 잠시 후 다시 시도해주세요.")
    except Exception as e:
        st.error(f"⚠️ AI 처리 중 오류가 발생했어요: {e}")
    st.stop()


def chat_with_claude(messages, return_truncated=False):
    response = _call_claude(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    text = response.content[0].text
    if return_truncated:
        return text, response.stop_reason == "max_tokens"
    return text


def write_paragraph_with_refs(topic, style, results, style_profile=None):
    ref_texts = "\n\n".join(
        f"[출처 {i+1}: {r['source']}]\n{r['text']}"
        for i, r in enumerate(results)
    )
    source_list = "\n".join(f"- {r['source']}" for r in results)
    style_section = build_style_instruction(style_profile) if style_profile else ""
    prompt = f"""아래 [참고문헌 내용]만을 근거로 "{topic}" 주제에 대한 학술적 단락을 작성해주세요.

작성 유형: {style}
{style_section}
[참고문헌 내용]
{ref_texts}

작성 지침:
1. 반드시 위 [참고문헌 내용]에 실제로 담긴 정보만 사용하세요. 참고문헌에 없는 사실·수치·주장은 절대 지어내지 마세요.
2. 참고문헌이 주제를 충분히 뒷받침하지 못하면, 억지로 쓰지 말고 "제공된 참고문헌만으로는 이 주제를 충분히 다루기 어렵습니다"라고 먼저 밝힌 뒤 가능한 범위에서만 작성하세요.
3. 여러 출처를 단순 나열하지 말고, 논리적으로 연결·종합하여 하나의 매끄러운 단락으로 작성하세요.
4. 각 주장 문장 끝에 근거가 된 출처를 괄호로 표기하세요. 출처 이름은 아래 [사용 가능한 출처]에 있는 이름을 그대로 사용합니다. 예: (출처 1). 참고문헌 안에서 저자·연도를 확인할 수 있으면 (저자, 연도) 형식을 우선 쓰되, 확인되지 않으면 출처 이름을 그대로 쓰세요.
5. 객관적이고 학술적인 문체로 3~5문장 작성하세요.
6. 단락 아래에 "**참고문헌:**" 항목으로 실제 인용한 출처만 나열하세요.

[사용 가능한 출처]
{source_list}
"""
    response = _call_claude(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def insert_citations(draft_text, search_results, style_profile=None):
    ref_texts = "\n\n".join(
        f"[논문 {i+1}: {r['source']}]\n{r['text']}"
        for i, r in enumerate(search_results)
    )
    style_section = build_style_instruction(style_profile) if style_profile else ""
    prompt = f"""다음 초안 텍스트에서 인용이 필요한 주장이나 사실을 찾아 참고문헌을 자동으로 삽입해주세요.
{style_section}
[초안 텍스트]
{draft_text}

[사용 가능한 참고문헌]
{ref_texts}

작업 지침:
1. [사용 가능한 참고문헌]에 실제로 있는 자료만 인용하세요. 목록에 없는 저자·연도·문헌을 절대 지어내지 마세요.
2. 초안의 주장이 제공된 참고문헌으로 실제 뒷받침될 때만 (저자, 연도) 형식으로 삽입하세요. 저자·연도를 자료에서 확인할 수 없으면 출처 이름을 그대로 쓰고, 뒷받침할 자료가 아예 없으면 인용을 넣지 말고 그 문장 뒤에 "[근거 자료 없음]"으로 표시하세요.
3. 참고문헌 내용과 실제로 관련된 곳에만 삽입하세요.
4. 수정된 전체 텍스트를 출력하세요.
5. 마지막에 "**참고문헌:**" 섹션에 실제 인용한 자료만 나열하세요.
6. 인용을 삽입한 위치와 이유를 간단히 설명하세요.
"""
    response = _call_claude(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def export_buttons(content, topic, key_prefix):
    """단락 결과 아래에 내보내기 버튼 표시"""
    st.markdown("**내보내기**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "📄 Word (.docx)",
            data=to_word("논문 단락", content, topic),
            file_name=f"{topic[:20]}_단락.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key=f"{key_prefix}_docx",
        )
    with col2:
        st.download_button(
            "📝 Markdown (.md)",
            data=to_markdown("논문 단락", content, topic),
            file_name=f"{topic[:20]}_단락.md",
            mime="text/markdown",
            key=f"{key_prefix}_md",
        )
    with col3:
        st.download_button(
            "📋 텍스트 (.txt)",
            data=content,
            file_name=f"{topic[:20]}_단락.txt",
            mime="text/plain",
            key=f"{key_prefix}_txt",
        )


# ── 페이지 설정 ───────────────────────────────────────────────
st.set_page_config(page_title="논문 작성 도우미", page_icon="📝", layout="wide")
st.markdown("""
<style>
details summary p { font-size: 0.9rem !important; font-weight: 500 !important; }
[data-testid="stStatusWidget"] svg { display: none !important; }
.paper-title { font-size: 0.92rem !important; font-weight: 600 !important; margin-bottom: 2px !important; line-height: 1.4 !important; }
.paper-meta  { font-size: 0.80rem !important; color: gray !important; margin: 0 !important; }
.paper-abs   { font-size: 0.85rem !important; margin-top: 4px !important; }
/* 추천 분석 출력 영역 헤딩 크기 축소 */
[data-testid="stMarkdownContainer"] h1 { font-size: 1.1rem !important; }
[data-testid="stMarkdownContainer"] h2 { font-size: 1.0rem !important; }
[data-testid="stMarkdownContainer"] h3 { font-size: 0.95rem !important; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { font-size: 0.88rem !important; }
</style>
""", unsafe_allow_html=True)

components.html("""
<script>
const icons = ['📝','📚','🔍','✏️','📖','🎓','🔬','📊','💡','📋'];
let idx = 0;
function updateIcon() {
    const doc = window.parent.document;
    const widget = doc.querySelector('[data-testid="stStatusWidget"]');
    if (!widget) return;
    let span = doc.querySelector('#academic-icon');
    if (!span) {
        span = doc.createElement('span');
        span.id = 'academic-icon';
        span.style.cssText = 'font-size:1.3rem;vertical-align:middle;line-height:1;';
        widget.insertBefore(span, widget.firstChild);
    }
    span.textContent = icons[idx % icons.length];
    idx++;
}
updateIcon();
setInterval(updateIcon, 1500);
</script>
""", height=0)
st.title("📝 논문 작성 도우미")
st.caption("교육공학 논문 작성을 위한 AI 에이전트")

library = get_library()

# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.header("기능 선택")
    mode = st.radio(
        "원하는 작업을 선택하세요",
        ["💬 자유 질문", "📚 단락 작성 (RAG)", "✒️ 인용 자동 삽입",
         "🔍 문헌 추천", "📊 데이터 분석 설계", "📄 PDF 분석",
         "🏗️ 논문 구조 설계", "✍️ 글쓰기 교정", "🔖 참고문헌 변환"],
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

    uploaded_files = st.file_uploader(
        "PDF 업로드 (여러 개 가능)",
        type="pdf",
        accept_multiple_files=True,
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

    saved_pdfs = list(PDF_DIR.glob("*.pdf"))
    if saved_pdfs:
        with st.expander(f"저장된 논문 {len(saved_pdfs)}개 보기"):
            for p in saved_pdfs:
                col1, col2 = st.columns([4, 1])
                col1.caption(p.name[:40])
                if col2.button("🗑️", key=f"del_{p.name}"):
                    p.unlink()
                    st.rerun()

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

    # ── 내 논문 스타일 ──────────────────────────────────────────
    st.subheader("🖊️ 내 논문 스타일")
    # 저장 경로 진단: 볼륨이 실제로 붙어 영구 저장 중인지 확인
    _vol_mount = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
    _persistent = bool(_vol_mount) and str(DATA_DIR).startswith(_vol_mount)
    with st.expander("💾 저장소 상태", expanded=False):
        st.caption(f"저장 경로: `{DATA_DIR}`")
        if _persistent:
            st.success("영구 볼륨에 저장 중 — 재배포·재시작해도 유지됩니다.")
        elif _vol_mount:
            st.warning(f"볼륨({_vol_mount})은 있으나 저장 경로가 볼륨 밖입니다. "
                       "DATA_DIR을 볼륨 경로로 맞추세요.")
        else:
            st.error("임시 저장 상태 — 볼륨 미마운트. 재배포 시 데이터가 삭제됩니다.")
    profile = load_style_profile(str(STYLE_PROFILE))
    if profile:
        st.success("✅ 스타일 프로필 분석 완료")
        with st.expander("프로필 내용 보기"):
            st.markdown(profile)
    else:
        st.info("내 논문을 업로드하면 문체·관점을 학습해요.")

    my_papers_upload = st.file_uploader(
        "내 논문 PDF 업로드",
        type="pdf",
        accept_multiple_files=True,
        key="my_papers_uploader",
    )
    if my_papers_upload:
        for uf in my_papers_upload:
            dest = MY_PAPERS_DIR / uf.name
            if not dest.exists():
                dest.write_bytes(uf.read())
        st.info(f"{len(my_papers_upload)}개 저장됨")

    my_pdfs = list(MY_PAPERS_DIR.glob("*.pdf"))
    if my_pdfs:
        st.caption(f"업로드된 내 논문: {len(my_pdfs)}개")
        if st.button("🔍 스타일 분석 시작", use_container_width=True):
            with st.spinner("논문 읽는 중..."):
                papers = load_my_papers(str(MY_PAPERS_DIR))
            if not papers:
                st.error("텍스트를 추출할 수 없어요.")
            else:
                prompt = build_style_prompt(papers)
                with st.spinner("Claude가 문체 분석 중... (1~2분)"):
                    result = _call_claude(
                        model="claude-sonnet-4-6",
                        max_tokens=3000,
                        system=SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    analysis = result.content[0].text
                save_style_profile(analysis, str(STYLE_PROFILE))
                st.success("✅ 스타일 분석 완료!")
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
        topic = st.text_input("작성할 주제를 입력하세요",
                              placeholder="예: 블렌디드 러닝이 학습 동기에 미치는 영향")
        col1, col2 = st.columns(2)
        with col1:
            style = st.selectbox("단락 유형",
                                 ["이론적 배경", "서론", "선행연구 검토", "논의", "결론"])
        with col2:
            top_k = st.slider("참고할 논문 수", 3, 8, 5)

        rag_style_profile = load_style_profile(str(STYLE_PROFILE))
        use_my_style_rag = st.checkbox(
            "🖊️ 내 스타일로 작성",
            value=bool(rag_style_profile),
            disabled=not rag_style_profile,
            help="사이드바에서 내 논문 스타일을 분석한 뒤 사용 가능해요." if not rag_style_profile else "내 문체·관점을 반영해 작성합니다.",
        )

        if st.button("✍️ 단락 작성", use_container_width=True, disabled=not topic):
            with st.spinner("관련 문헌 검색 중..."):
                results = library.search(topic, top_k=top_k)

            if not results:
                st.error("관련 문헌을 찾지 못했어요.")
            else:
                with st.expander(f"🔍 검색된 참고문헌 {len(results)}개", expanded=False):
                    for r in results:
                        st.markdown(f"**{r['source']}** (관련도: {r['score']:.2f})")
                        st.caption(r["text"][:200] + "...")
                        st.divider()

                with st.spinner("단락 작성 중..."):
                    applied_profile = rag_style_profile if use_my_style_rag else None
                    paragraph = write_paragraph_with_refs(topic, style, results, applied_profile)

                st.markdown("### 작성된 단락")
                st.markdown(paragraph)
                export_buttons(paragraph, topic, "rag")

# ── 인용 자동 삽입 ────────────────────────────────────────────
elif mode == "✒️ 인용 자동 삽입":
    st.subheader("✒️ 인용 자동 삽입")
    st.caption("초안 텍스트를 붙여넣으면 Claude가 인용이 필요한 부분을 찾아 참고문헌을 자동으로 삽입해요.")

    cite_source = st.radio(
        "참고문헌 출처 선택",
        ["📁 내 라이브러리만", "🌐 외부 검색 포함 (Semantic Scholar + arXiv)"],
        horizontal=True,
    )
    use_external = cite_source.startswith("🌐")

    if not use_external and not library.is_ready():
        st.warning("먼저 사이드바에서 PDF를 업로드하고 '문헌 학습 시작'을 눌러주세요.")
    else:
        draft = st.text_area("초안 텍스트를 입력하세요", height=250,
                             placeholder="인용을 넣고 싶은 글을 여기에 붙여넣으세요...")
        top_k = st.slider("참고할 논문 수", 3, 10, 5)

        cite_style_profile = load_style_profile(str(STYLE_PROFILE))
        use_my_style_cite = st.checkbox(
            "🖊️ 내 스타일 유지",
            value=bool(cite_style_profile),
            disabled=not cite_style_profile,
            help="사이드바에서 내 논문 스타일을 분석한 뒤 사용 가능해요." if not cite_style_profile else "인용 삽입 후에도 내 문체를 유지합니다.",
        )

        if st.button("✒️ 인용 삽입", use_container_width=True, disabled=not draft):
            results = []

            if not use_external:
                # 내 라이브러리에서 검색
                with st.spinner("라이브러리에서 관련 문헌 검색 중..."):
                    lib_results = library.search(draft[:300], top_k=top_k)
                results = [
                    {"source": r["source"], "text": r["text"]}
                    for r in lib_results
                ]
            else:
                # 외부 검색 (Semantic Scholar + arXiv)
                with st.spinner("외부 검색 중 (Semantic Scholar + arXiv)..."):
                    try:
                        ext_papers = cached_search(draft[:200], top_k, "둘 다", version=3)
                        results = [
                            {
                                "source": f"{p['authors']} ({p['year']})",
                                "text": p["abstract"],
                            }
                            for p in ext_papers
                        ]
                    except Exception as e:
                        st.error(f"외부 검색 오류: {e}")

                # 라이브러리도 함께 활용 (있으면)
                if library.is_ready():
                    lib_results = library.search(draft[:300], top_k=3)
                    lib_formatted = [
                        {"source": r["source"], "text": r["text"]}
                        for r in lib_results
                    ]
                    results = lib_formatted + results

            if not results:
                st.error("관련 문헌을 찾지 못했어요.")
            else:
                with st.expander(f"🔍 활용할 참고문헌 {len(results)}개", expanded=False):
                    for r in results:
                        st.markdown(f"**{r['source']}**")
                        st.caption(r["text"][:200] + "...")
                        st.divider()

                with st.spinner("인용 삽입 중..."):
                    applied_cite_profile = cite_style_profile if use_my_style_cite else None
                    result_text = insert_citations(draft, results, applied_cite_profile)

                st.markdown("### 인용이 삽입된 텍스트")
                st.markdown(result_text)
                export_buttons(result_text, "인용삽입결과", "cite")

# ── 문헌 추천 ────────────────────────────────────────────────
elif mode == "🔍 문헌 추천":
    st.subheader("🔍 문헌 추천")
    st.caption("연구 주제를 입력하면 Semantic Scholar / arXiv에서 관련 논문을 검색하고 Claude가 추천해드려요.")

    topic = st.text_input("연구 주제 또는 키워드 입력",
                          placeholder="예: blended learning student motivation")
    col1, col2, col3 = st.columns(3)
    with col1:
        source = st.selectbox("검색 소스", ["Semantic Scholar", "arXiv", "둘 다"])
    with col2:
        num_results = st.slider("검색 논문 수", 5, 20, 10)
    with col3:
        my_topic = st.text_input("내 연구 주제 (선택)", placeholder="Claude 추천 기준")

    ssci_priority = st.checkbox(
        "🏆 SSCI급 학술지 우선 추천",
        help="교육학·교육공학 분야 주요 SSCI 등재 학술지 논문을 우선 추천하고, SSCI 여부를 표시해요.",
    )

    if st.button("🔍 문헌 검색 및 추천", use_container_width=True, disabled=not topic):
        with st.spinner(f"{source} 검색 중..."):
            try:
                papers = cached_search(topic, num_results, source, version=5)
            except Exception as e:
                st.error(f"❌ {e}")
                st.info("💡 영어 키워드로 입력해보세요.\n예) blended learning, flipped classroom")
                papers = []

        if papers:
            # SSCI 우선 모드: 인용 수 있는 논문 먼저 정렬
            if ssci_priority:
                def _cite_key(p):
                    c = p.get("citations")
                    return c if isinstance(c, int) else 0
                papers = sorted(papers, key=_cite_key, reverse=True)

            paper_list = "\n\n".join(
                f"[{i+1}] {p['title']}\n"
                f"저자: {p['authors']} ({p['year']}) | 학술지: {p.get('journal','') or p.get('source','')}\n"
                f"인용: {p.get('citations', 'N/A')}회\n"
                f"초록: {p['abstract']}"
                for i, p in enumerate(papers)
            )
            research_context = f"\n내 연구 주제: {my_topic}" if my_topic else ""

            SSCI_JOURNALS = """Computers & Education, British Journal of Educational Technology,
Educational Technology Research and Development, Learning and Instruction,
Computers in Human Behavior, Internet and Higher Education,
Journal of Computer Assisted Learning, Educational Researcher,
Review of Educational Research, American Educational Research Journal,
Contemporary Educational Psychology, Educational Psychologist,
Distance Education, Journal of Educational Technology & Society,
Instructional Science, Journal of the Learning Sciences,
Metacognition and Learning, Educational Psychology Review,
Behaviour & Information Technology, Higher Education,
Teaching and Teacher Education, Educational Research Review"""

            ssci_block = f"""
[SSCI 우선 추천 모드]
아래 교육학·교육공학 분야 주요 SSCI 학술지 목록을 참고하여, 해당 학술지 게재 논문을 우선 추천하세요.
각 논문 제목 앞에 🏆(SSCI 등재 학술지) 또는 📄(미확인)을 표시해주세요.
인용 수가 높을수록 영향력 있는 논문이므로 우선순위에 반영해주세요.

주요 SSCI 교육학·교육공학 학술지:
{SSCI_JOURNALS}
""" if ssci_priority else ""

            prompt = f"""다음 논문 목록을 분석하여 연구에 유용한 순서로 추천해주세요.{research_context}
{ssci_block}
검색 키워드: {topic}

[논문 목록]
{paper_list}

각 논문에 대해 아래 형식으로 작성해주세요:
**[번호] 논문 제목** | 저자 (연도) | 학술지명
1. 핵심 내용 한 줄 요약
2. 연구에서 어떻게 활용할 수 있는지
3. 추천 여부 (⭐ 강추 / 👍 참고 / 💡 선택적)

마지막에 가장 중요한 논문 3편을 선정하고 이유를 설명해주세요."""

            with st.spinner("Claude가 논문 분석 중..."):
                recommendation, truncated = chat_with_claude(
                    [{"role": "user", "content": prompt}], return_truncated=True
                )

            st.markdown("### Claude 추천 분석")
            st.markdown(recommendation)
            if truncated:
                st.warning("⚠️ 분석이 길어 출력이 일부 잘렸어요. 검색 논문 수를 줄이면 모든 논문이 끝까지 표시됩니다. (아래 '검색된 논문 전체 목록'에는 모든 논문이 나와요.)")
            st.download_button("📋 추천 결과 저장 (.txt)", recommendation,
                               file_name="문헌추천결과.txt", mime="text/plain")

            st.divider()
            st.markdown("### 📄 검색된 논문 전체 목록")

            for i, p in enumerate(papers):
                journal = p.get("journal") or ""
                paper_src = p.get("source", "")
                venue_label = journal if journal else paper_src

                citations = p.get("citations")
                cite_str = f"인용 {citations}회" if citations not in ("N/A", 0, None, "") else ""

                title_s   = _html.escape(str(p.get("title", "")))
                authors_s = _html.escape(str(p.get("authors", "")))
                venue_s   = _html.escape(venue_label)
                abstract_s = _html.escape((p["abstract"][:200].rsplit(" ", 1)[0] + "..."))

                sjr_badge = quartile_badge(journal)

                author_line = f"{authors_s} ({p['year']})"
                source_parts = [venue_s]
                if cite_str:
                    source_parts.append(cite_str)
                source_line = " · ".join(source_parts)

                url = p.get("url", "")
                link_tag = (
                    f' <a href="{url}" target="_blank" style="font-size:0.72rem;text-decoration:none">🔗</a>'
                    if url else ""
                )

                st.markdown(
                    f'<p style="font-size:0.78rem;font-weight:600;margin:0 0 1px 0;line-height:1.4">'
                    f'{i+1}. {title_s}{link_tag}</p>'
                    f'<p style="font-size:0.82rem;color:#777;margin:0">{author_line}</p>'
                    f'<p style="font-size:0.82rem;color:#555;margin:0 0 3px 0">'
                    f'출처: {source_line}{sjr_badge}</p>'
                    f'<p style="font-size:0.84rem;margin:2px 0 0 0">{abstract_s}</p>',
                    unsafe_allow_html=True,
                )
                st.divider()

# ── 데이터 분석 설계 ──────────────────────────────────────────
elif mode == "📊 데이터 분석 설계":
    st.subheader("📊 데이터 분석 설계")
    st.caption("Excel, SPSS, 인터뷰 텍스트를 업로드하면 데이터 구조를 파악하고 연구문제와 분석 방법을 제안해드려요.")

    uploaded_data = st.file_uploader(
        "데이터 파일 업로드",
        type=["xlsx", "xls", "sav", "csv", "txt", "docx"],
        help="Excel(.xlsx), SPSS(.sav), CSV(.csv), 인터뷰 텍스트(.txt/.docx) 지원",
    )
    research_context = st.text_input(
        "연구 맥락 (선택)",
        placeholder="예: 대학생 블렌디드 러닝 경험 설문조사",
        help="어떤 연구인지 간단히 설명하면 더 정확한 제안을 드릴 수 있어요",
    )

    if uploaded_data:
        try:
            with st.spinner("파일 읽는 중..."):
                data_type, data, meta = load_file(uploaded_data)

            if data_type == "quantitative":
                st.success(f"✅ 정량 데이터 로드 완료 — {len(data)}행 × {len(data.columns)}열")

                tab1, tab2 = st.tabs(["📋 데이터 미리보기", "📈 기술통계"])
                with tab1:
                    st.dataframe(get_preview(data), use_container_width=True)
                with tab2:
                    stats = get_basic_stats(data)
                    if stats is not None:
                        st.dataframe(stats, use_container_width=True)
                    else:
                        st.info("수치형 변수가 없어요.")

                summary = summarize_dataframe(data, meta)

            else:  # qualitative
                st.success(f"✅ 인터뷰 텍스트 로드 완료")
                with st.expander("텍스트 미리보기"):
                    st.text(data[:1000] + ("..." if len(data) > 1000 else ""))
                summary = summarize_interview(data)

            analysis_goal = st.selectbox(
                "분석 목적",
                ["연구문제 제안", "분석 방법 추천", "연구문제 + 분석 방법 모두"],
            )

            if st.button("🔍 분석 설계 시작", use_container_width=True):
                context_line = f"\n연구 맥락: {research_context}" if research_context else ""
                data_type_label = "정량(설문/측정) 데이터" if data_type == "quantitative" else "질적(인터뷰) 데이터"

                prompt = f"""다음 {data_type_label}의 구조를 분석하여 {analysis_goal}을 제안해주세요.{context_line}

## 데이터 구조
{summary}

제안 지침:
1. **연구문제**: 이 데이터로 탐구할 수 있는 구체적인 연구문제 3~5개를 제안하세요.
   - 각 연구문제는 "~은 ~에 어떤 영향을 미치는가?" 형식으로 작성
   - 변수 이름을 직접 활용할 것

2. **분석 방법**: 각 연구문제에 맞는 통계/분석 방법을 추천하세요.
   - 정량: t검정, ANOVA, 회귀분석, 구조방정식 등
   - 질적: 주제 분석, 근거이론, 내러티브 분석 등
   - 분석 소프트웨어(SPSS, R, NVivo 등)도 함께 추천

3. **주의사항**: 데이터의 한계나 분석 시 유의할 점을 언급하세요.

교육공학 연구 맥락에서 답변해주세요."""

                with st.spinner("Claude가 분석 설계 중..."):
                    result = chat_with_claude([{"role": "user", "content": prompt}])

                st.markdown("### 분석 설계 제안")
                st.markdown(result)
                st.download_button(
                    "📋 분석 설계 저장 (.txt)",
                    result,
                    file_name="분석설계제안.txt",
                    mime="text/plain",
                )

            # ── 통계 직접 실행 ────────────────────────────────
            if data_type == "quantitative":
                st.divider()
                st.subheader("📈 통계 직접 실행")
                st.caption("설계 제안에서 그치지 않고, 선택한 분석을 업로드한 데이터로 바로 계산하고 논문용 결과 문장까지 작성해드려요.")

                num_cols = list(data.select_dtypes(include="number").columns)
                all_cols = list(data.columns)

                analysis = st.selectbox("분석 방법", [
                    "독립표본 t검정 (두 집단 평균 비교)",
                    "대응표본 t검정 (사전·사후 등 짝지은 비교)",
                    "일원분산분석 ANOVA (3개 이상 집단)",
                    "상관분석 (Pearson/Spearman)",
                    "카이제곱 검정 (범주형 × 범주형)",
                    "신뢰도 분석 (Cronbach's α)",
                    "회귀분석 (중다회귀)",
                    "다층모형 HLM (임의절편)",
                    "잠재계층분석 LCA",
                    "구조방정식 SEM",
                ])

                run_fn = None
                if not num_cols and not analysis.startswith("카이제곱"):
                    st.warning("수치형 변수가 없어 이 분석을 실행할 수 없어요.")
                elif analysis.startswith("독립표본"):
                    dv = st.selectbox("종속변수 (수치형)", num_cols)
                    grp = st.selectbox("집단변수 (정확히 2개 집단)", all_cols)
                    run_fn = lambda: run_ttest_ind(data, dv, grp)
                elif analysis.startswith("대응표본"):
                    v1 = st.selectbox("변수 1 (예: 사전점수)", num_cols)
                    v2 = st.selectbox("변수 2 (예: 사후점수)", num_cols,
                                      index=min(1, len(num_cols) - 1))
                    run_fn = lambda: run_ttest_rel(data, v1, v2)
                elif analysis.startswith("일원분산분석"):
                    dv = st.selectbox("종속변수 (수치형)", num_cols)
                    grp = st.selectbox("집단변수 (3개 이상 집단)", all_cols)
                    run_fn = lambda: run_anova(data, dv, grp)
                elif analysis.startswith("상관분석"):
                    cols = st.multiselect("분석할 변수 (2개 이상)", num_cols,
                                          default=num_cols[:min(3, len(num_cols))])
                    method = st.radio("상관계수", ["Pearson", "Spearman"], horizontal=True)
                    run_fn = lambda: run_correlation(data, cols, method.lower())
                elif analysis.startswith("카이제곱"):
                    v1 = st.selectbox("변수 1 (범주형)", all_cols)
                    v2 = st.selectbox("변수 2 (범주형)", all_cols,
                                      index=min(1, len(all_cols) - 1))
                    run_fn = lambda: run_chisquare(data, v1, v2)
                elif analysis.startswith("신뢰도"):
                    cols = st.multiselect("같은 척도의 문항들 (2개 이상)", num_cols)
                    run_fn = lambda: run_cronbach(data, cols)
                elif analysis.startswith("회귀분석"):
                    dv = st.selectbox("종속변수", num_cols)
                    ivs = st.multiselect("독립변수 (1개 이상)",
                                         [c for c in num_cols if c != dv])
                    run_fn = lambda: run_regression(data, dv, ivs)
                elif analysis.startswith("다층모형"):
                    dv = st.selectbox("종속변수 (1수준)", num_cols)
                    ivs = st.multiselect("독립변수 (1개 이상)",
                                         [c for c in num_cols if c != dv])
                    grp = st.selectbox("상위수준(2수준) 집단변수 — 예: 학교ID, 학급ID", all_cols)
                    run_fn = lambda: run_hlm(data, dv, ivs, grp)
                elif analysis.startswith("잠재계층"):
                    cols = st.multiselect("지표 변수 (2개 이상)", num_cols)
                    meas_label = st.radio("지표 유형",
                                          ["연속형 (리커트 평균, 점수 등)", "범주형 (예/아니오, 선택지 등)"],
                                          horizontal=True)
                    meas = "continuous" if meas_label.startswith("연속형") else "categorical"
                    k_label = st.selectbox("계층 수",
                                           ["자동 (BIC 최적 모형 선택)", "2", "3", "4", "5", "6"])
                    k_sel = 0 if k_label.startswith("자동") else int(k_label)
                    max_k = st.slider("탐색할 최대 계층 수", 3, 6, 5,
                                      help="1부터 이 수까지 모형을 적합해 AIC/BIC로 비교해요.")
                    run_fn = lambda: run_lca(data, cols, n_classes=k_sel,
                                             max_classes=max_k, measurement=meas)
                else:  # SEM
                    st.caption("사용 가능한 변수: " + ", ".join(num_cols[:30])
                               + ("..." if len(num_cols) > 30 else ""))
                    model_spec = st.text_area(
                        "SEM 모델식 (lavaan 스타일)",
                        placeholder=SEM_EXAMPLE, height=170,
                        help="=~ : 잠재변수 정의(측정모형) / ~ : 회귀 경로(구조모형) / ~~ : 공분산. 변수명은 데이터 열 이름과 일치해야 해요.",
                    )
                    mg_col = st.selectbox(
                        "다집단 비교 (선택)", ["사용 안 함"] + all_cols,
                        help="집단변수를 선택하면 집단별로 모형을 적합하고, 구조 경로가 집단 간에 다른지 z검정으로 비교해요. (예: 성별, 학교급)",
                    )
                    if mg_col == "사용 안 함":
                        run_fn = lambda: run_sem(data, model_spec)
                    else:
                        run_fn = lambda: run_sem_multigroup(data, model_spec, mg_col)

                interpret = st.checkbox("🤖 결과를 논문용 문장으로 해석", value=True,
                                        help="계산된 결과를 APA 표기를 포함한 '연구 결과' 절 문장으로 작성해드려요.")

                if run_fn and st.button("▶️ 분석 실행", use_container_width=True, type="primary"):
                    res = None
                    try:
                        with st.spinner("통계 계산 중..."):
                            res = run_fn()
                    except ValueError as ve:
                        st.error(f"⚠️ {ve}")
                    except Exception as ex:
                        st.error(f"❌ 분석 실패: {ex}")
                    if res:
                        interp = None
                        if interpret:
                            context_line = f"\n연구 맥락: {research_context}" if research_context else ""
                            interp_prompt = f"""다음은 실제 데이터로 방금 실행한 통계 분석 결과입니다. 이 결과를 논문의 '연구 결과' 절에 쓸 수 있도록 해석해주세요.{context_line}

## 분석 결과
{res['summary']}

작성 지침:
1. APA 스타일 통계치 표기(t, F, χ², β, p, 효과크기 등)를 포함한 학술적 한국어 문장으로 서술하세요.
2. 효과크기의 실질적 의미를 함께 해석하세요.
3. 유의하지 않은 결과도 있는 그대로 서술하세요. 결과를 긍정적으로 왜곡하지 마세요.
4. 위 분석 결과에 없는 수치·통계량은 절대 만들어내지 마세요.
5. 마지막에 '해석 시 유의점' 1~2가지를 덧붙이세요."""
                            with st.spinner("Claude가 결과를 논문 문장으로 작성 중..."):
                                interp = chat_with_claude(
                                    [{"role": "user", "content": interp_prompt}])
                        st.session_state["stats_run"] = {
                            "analysis": analysis, "result": res, "interp": interp,
                        }

                saved = st.session_state.get("stats_run")
                if saved:
                    st.markdown(f"#### 결과 — {saved['analysis']}")
                    for tname, tdf in saved["result"]["tables"].items():
                        st.markdown(f"**{tname}**")
                        st.dataframe(tdf, use_container_width=True)
                    with st.expander("결과 요약 (텍스트)"):
                        st.text(saved["result"]["summary"])
                    if saved["interp"]:
                        st.markdown("#### 📝 논문용 결과 해석")
                        st.markdown(saved["interp"])
                        st.download_button(
                            "📋 결과 해석 저장 (.txt)", saved["interp"],
                            file_name="통계결과해석.txt", mime="text/plain",
                            key="stats_interp_dl",
                        )

        except Exception as e:
            st.error(f"❌ 파일 로드 오류: {e}")

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
        analyze_option = st.selectbox("분석 유형 선택",
            ["핵심 내용 요약", "연구 방법 분석", "이론적 배경 정리", "연구 결과 요약", "비판적 검토"])
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
    correction_type = st.selectbox("교정 유형",
        ["학술체로 변환", "문장 명확성 개선", "논리 흐름 개선", "전체 교정"])

    corr_style_profile = load_style_profile(str(STYLE_PROFILE))
    use_my_style_corr = st.checkbox(
        "🖊️ 내 문체 반영",
        value=bool(corr_style_profile),
        disabled=not corr_style_profile,
        help="사이드바에서 내 논문 스타일을 분석한 뒤 사용 가능해요." if not corr_style_profile else "교정할 때 내 고유 문체·어휘를 유지합니다.",
    )

    if st.button("교정 시작") and text_input:
        style_section = build_style_instruction(corr_style_profile) if use_my_style_corr else ""
        prompt = f"""다음 글을 '{correction_type}' 관점에서 교정해주세요. 원문과 수정본을 나란히 보여주고, 수정 이유도 설명해주세요.
{style_section}
[원문]
{text_input}"""
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
if mode in ["💬 자유 질문", "📄 PDF 분석", "🏗️ 논문 구조 설계", "✍️ 글쓰기 교정", "🔖 참고문헌 변환"]:
    for msg in st.session_state.messages:
        display_content = msg["content"]
        if len(display_content) > 500 and msg["role"] == "user":
            display_content = display_content[:300] + "\n\n...(내용 생략)..."
        with st.chat_message(msg["role"]):
            st.write(display_content)

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response, truncated = chat_with_claude(
                    st.session_state.messages, return_truncated=True
                )
                st.write(response)
                if truncated:
                    st.warning("⚠️ 답변이 길어 출력이 일부 잘렸어요. 항목 수를 나눠서 다시 시도하면 전부 표시됩니다.")
        st.session_state.messages.append({"role": "assistant", "content": response})

    if mode == "💬 자유 질문":
        user_input = st.chat_input("논문 작성에 대해 무엇이든 물어보세요...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.rerun()
