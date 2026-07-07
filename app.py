import html as _html
import io
import json
import os
import re
import tempfile
from pathlib import Path

import anthropic
import pdfplumber
import streamlit as st
from dotenv import load_dotenv

import streamlit.components.v1 as components
from rag import ReferenceLibrary
from scholar import search_papers
from export import to_word, to_markdown, to_word_redline, to_hwpx_redline, diff_segments
import inplace_redline as ir
import journal_format as jf
from data_analyzer import load_file, summarize_dataframe, summarize_interview, get_preview, get_basic_stats, load_codebook_text
from stats_runner import (
    run_ttest_ind, run_ttest_rel, run_anova, run_correlation,
    run_chisquare, run_cronbach, run_regression, run_hlm,
    run_lca, run_sem, run_sem_multigroup, run_lcsm, run_lcsm_multigroup,
    SEM_EXAMPLE,
)
from style_analyzer import (
    load_my_papers, build_style_prompt, save_style_profile,
    load_style_profile, build_style_instruction
)
from sjr_data import quartile_badge
from projects import (
    load_projects, save_projects, project_dir, migrate_legacy,
    load_analyses, save_analysis, delete_analysis, tables_from_record,
    MAX_PROJECTS,
)

load_dotenv()

# 저장 경로 결정 우선순위:
#   1) RAILWAY_VOLUME_MOUNT_PATH (볼륨 붙으면 Railway가 자동 주입) — 볼륨이 있으면 무조건 여기 저장
#   2) DATA_DIR (직접 지정)
#   3) ./data (로컬 기본값)
# 볼륨 경로를 최우선으로 둔다. 과거 DATA_DIR=./data 로 잘못 설정돼 볼륨을 두고도
# 임시 디스크에 저장 → 재배포마다 데이터 소실되던 사고 재발 방지.
DATA_DIR = os.getenv("RAILWAY_VOLUME_MOUNT_PATH") or os.getenv("DATA_DIR") or "./data"

# ── 논문 프로젝트(워크스페이스) ───────────────────────────────
# 참고문헌 PDF / RAG 인덱스 / 분석 결과는 프로젝트별로 분리.
# 내 문체 프로필은 사람에 속하므로 전역 유지.
try:
    migrate_legacy(DATA_DIR)
except Exception:
    pass  # 레거시 이전 실패가 앱 구동을 막지 않도록 (다음 실행에 재시도)
PROJECTS = load_projects(DATA_DIR)
if st.session_state.get("project_id") not in [p["id"] for p in PROJECTS]:
    st.session_state["project_id"] = PROJECTS[0]["id"]

PROJ_DIR       = project_dir(DATA_DIR, st.session_state["project_id"])
PDF_DIR        = PROJ_DIR / "pdfs"
DB_DIR         = PROJ_DIR / "reference_db"
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
- 참고문헌 표기 시 학술지 이름에 따옴표(" ", ' ', 「」, "" 등)를 붙이지 않음.
  APA 등 표기 규정에 따라 학술지명·권 번호는 이탤릭(*학술지명*)으로 표기
"""


@st.cache_resource
def get_library(db_path):
    # 프로젝트별 인덱스 경로가 캐시 키 → 프로젝트 전환 시 해당 라이브러리 로드
    return ReferenceLibrary(db_path=db_path)


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


def write_paragraph_with_refs(topic, style, results, style_profile=None, language="한국어"):
    ref_texts = "\n\n".join(
        f"[출처 {i+1}: {r['source']}]\n{r['text']}"
        for i, r in enumerate(results)
    )
    source_list = "\n".join(f"- {r['source']}" for r in results)
    style_section = build_style_instruction(style_profile) if style_profile else ""
    lang_line = ("반드시 영어(English)로 작성하세요. 학술 논문에 적합한 영어 문체를 사용하세요."
                 if language == "English"
                 else "반드시 한국어로 작성하세요.")
    prompt = f"""아래 [참고문헌 내용]만을 근거로 "{topic}" 주제에 대한 학술적 단락을 작성해주세요.

작성 언어: {lang_line}
작성 유형: {style}
{style_section}
[참고문헌 내용]
{ref_texts}

작성 지침:
1. 반드시 위 [참고문헌 내용]에 실제로 담긴 정보만 사용하세요. 참고문헌에 없는 사실·수치·주장은 절대 지어내지 마세요.
2. 참고문헌이 주제를 충분히 뒷받침하지 못하면, 억지로 쓰지 말고 그 사실을 먼저 밝힌 뒤 가능한 범위에서만 작성하세요.
3. 여러 출처를 단순 나열하지 말고, 논리적으로 연결·종합하여 하나의 매끄러운 단락으로 작성하세요.
4. 각 주장 문장 끝에 근거가 된 출처를 괄호로 표기하세요. 출처 이름은 아래 [사용 가능한 출처]에 있는 이름을 그대로 사용합니다. 예: (출처 1). 참고문헌 안에서 저자·연도를 확인할 수 있으면 (저자, 연도) 형식을 우선 쓰되, 확인되지 않으면 출처 이름을 그대로 쓰세요.
5. 객관적이고 학술적인 문체로 3~5문장 작성하세요. ({lang_line})
6. 단락 아래에 참고문헌 항목(영어면 "References:", 한국어면 "**참고문헌:**")으로 실제 인용한 출처만 나열하세요.

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


def _split_for_correction(text, limit=3000):
    """문단 경계를 지키며 limit자 내외 구간으로 분할 (전체 문서 교정용)"""
    paras = [p for p in text.splitlines() if p.strip()]
    chunks, cur, cur_len = [], [], 0
    for p in paras:
        if cur and cur_len + len(p) > limit:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += len(p)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks or [text]


def _parse_correction(resp):
    """[교정문]...[교정 끝][수정 설명]... 형식 응답 파싱"""
    m = re.search(r"\[교정문\]\s*(.*?)\s*\[교정 끝\]", resp, re.S)
    if not m:
        return resp.strip(), ""
    corrected = m.group(1).strip()
    explanation = resp.split("[교정 끝]", 1)[1].replace("[수정 설명]", "").strip()
    return corrected, explanation


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


def render_library_manager():
    """참고문헌 라이브러리 관리 UI (PDF 업로드 + 학습 + 목록 + 초기화).
    단락 작성/PDF 분석 통합 모드 안에서 사용."""
    saved_pdfs = list(PDF_DIR.glob("*.pdf"))
    if library.is_ready():
        st.success(f"✅ {library.count_papers()}개 논문 학습 완료")
    elif saved_pdfs:
        st.warning(f"⚠️ {len(saved_pdfs)}개 파일 있음 — 아래 '문헌 학습 시작'을 눌러주세요")
    else:
        st.info("논문 PDF를 업로드하면 단락 작성·분석에 활용해요. (여러 개 가능)")

    uploaded_files = st.file_uploader(
        "논문 PDF 업로드 (여러 개 가능)", type="pdf",
        accept_multiple_files=True, key="lib_uploader",
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
        with st.expander(f"저장된 논문 {len(saved_pdfs)}개 보기 / 삭제"):
            for p in saved_pdfs:
                c1, c2 = st.columns([5, 1])
                c1.caption(p.name[:50])
                if c2.button("🗑️", key=f"del_{p.name}"):
                    p.unlink()
                    st.rerun()
        col_a, col_b = st.columns([2, 1])
        if col_a.button("🔄 문헌 학습 시작", use_container_width=True,
                        help="업로드한 PDF를 색인해 단락 작성(RAG)에 사용할 수 있게 해요."):
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
        if col_b.button("🗑️ 전체 초기화", use_container_width=True, type="secondary"):
            st.session_state.confirm_reset = True

    if st.session_state.get("confirm_reset"):
        st.warning("정말 삭제할까요? 이 프로젝트의 모든 PDF와 학습 데이터가 사라져요.")
        c1, c2 = st.columns(2)
        if c1.button("✅ 확인", use_container_width=True):
            import shutil
            for f in PDF_DIR.glob("*.pdf"):
                f.unlink()
            if DB_DIR.exists():
                shutil.rmtree(DB_DIR)
                DB_DIR.mkdir(parents=True, exist_ok=True)
            get_library.clear()
            st.session_state.confirm_reset = False
            st.rerun()
        if c2.button("❌ 취소", use_container_width=True):
            st.session_state.confirm_reset = False
            st.rerun()


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

library = get_library(str(DB_DIR))

# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    # ── 논문 프로젝트 선택 ──────────────────────────────────
    st.subheader("📂 논문 프로젝트")
    proj_cols = st.columns(len(PROJECTS))
    for i, p in enumerate(PROJECTS):
        is_cur = p["id"] == st.session_state["project_id"]
        if proj_cols[i].button(
            p["id"], key=f"proj_btn_{p['id']}",
            type="primary" if is_cur else "secondary",
            use_container_width=True, help=p["name"],
        ) and not is_cur:
            st.session_state["project_id"] = p["id"]
            st.session_state.pop("stats_run", None)  # 이전 프로젝트 결과 잔상 제거
            st.rerun()
    _cur_proj = next(p for p in PROJECTS if p["id"] == st.session_state["project_id"])
    st.caption(f"현재: **{_cur_proj['id']}. {_cur_proj['name']}**")
    with st.expander("✏️ 이름 바꾸기 / 프로젝트 추가"):
        _new_name = st.text_input("현재 프로젝트 이름", value=_cur_proj["name"],
                                  key=f"pname_{_cur_proj['id']}", max_chars=20)
        c1, c2 = st.columns(2)
        if c1.button("💾 이름 저장", use_container_width=True):
            _cur_proj["name"] = _new_name.strip() or _cur_proj["name"]
            save_projects(DATA_DIR, PROJECTS)
            st.rerun()
        if len(PROJECTS) < MAX_PROJECTS:
            if c2.button("➕ 프로젝트 추가", use_container_width=True):
                _nid = str(max(int(p["id"]) for p in PROJECTS) + 1)
                PROJECTS.append({"id": _nid, "name": f"프로젝트 {_nid}"})
                save_projects(DATA_DIR, PROJECTS)
                st.session_state["project_id"] = _nid
                st.rerun()
        else:
            c2.caption(f"최대 {MAX_PROJECTS}개")

    st.divider()
    st.header("기능 선택")
    mode = st.radio(
        "원하는 작업을 선택하세요",
        ["💬 자유 질문", "📚 단락 작성 · 논문 분석", "✒️ 인용 자동 삽입",
         "🔍 문헌 추천", "📊 데이터 분석 설계",
         "🏗️ 논문 구조 설계", "✍️ 글쓰기 교정", "🔖 학술지 형식 · 참고문헌 변환"],
    )

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

# ── 단락 작성 · 논문 분석 (통합) ──────────────────────────────
if mode == "📚 단락 작성 · 논문 분석":
    st.subheader("📚 참고문헌 기반 단락 작성 · 논문 분석")

    # 참고문헌 라이브러리 (업로드 + 학습 + 관리) — 이 모드 안에 배치
    with st.container(border=True):
        st.markdown("##### 📁 참고문헌 라이브러리")
        render_library_manager()

    tab_write, tab_analyze = st.tabs(["✍️ 단락 작성", "📄 논문 PDF 분석"])

    with tab_write:
        if not library.is_ready():
            st.warning("먼저 위에서 PDF를 업로드하고 '문헌 학습 시작'을 눌러주세요.")
        else:
            topic = st.text_input("작성할 주제를 입력하세요",
                                  placeholder="예: 블렌디드 러닝이 학습 동기에 미치는 영향")
            col1, col2, col3 = st.columns(3)
            with col1:
                style = st.selectbox("단락 유형",
                                     ["이론적 배경", "서론", "선행연구 검토", "논의", "결론"])
            with col2:
                rag_lang = st.selectbox("작성 언어", ["한국어", "English"])
            with col3:
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
                        # 논문이 다른 언어(예: 영어)일 수 있어 주제를 번역해 재검색
                        try:
                            alt = chat_with_claude([{"role": "user", "content":
                                f"다음 논문 주제를 검색용으로 번역해줘. 한국어면 영어로, 영어면 한국어로. "
                                f"핵심 키워드 위주로, 번역문만 출력:\n{topic}"}])
                            alt_results = library.search(alt.strip(), top_k=top_k)
                            if alt_results:
                                results = alt_results
                                st.caption(f"💡 '{alt.strip()}'(으)로도 검색했어요 (업로드 논문 언어에 맞춰).")
                        except Exception:
                            pass

                if not results:
                    st.error("관련 문헌을 찾지 못했어요. 주제 키워드가 업로드한 논문의 표현·언어와 맞는지 확인해보세요. "
                             "(예: 논문이 영어면 영어 키워드로, 또는 더 일반적인 용어로)")
                else:
                    with st.expander(f"🔍 검색된 참고문헌 {len(results)}개", expanded=False):
                        for r in results:
                            st.markdown(f"**{r['source']}** (관련도: {r['score']:.2f})")
                            st.caption(r["text"][:200] + "...")
                            st.divider()

                    with st.spinner("단락 작성 중..."):
                        applied_profile = rag_style_profile if use_my_style_rag else None
                        paragraph = write_paragraph_with_refs(topic, style, results,
                                                              applied_profile, rag_lang)

                    st.markdown("### 작성된 단락")
                    st.markdown(paragraph)
                    export_buttons(paragraph, topic, "rag")

    with tab_analyze:
        st.caption("업로드한 논문을 골라 요약·분석해요. (위 라이브러리에 올린 PDF를 사용하거나 새로 올릴 수 있어요)")
        lib_pdfs = list(PDF_DIR.glob("*.pdf"))
        pdf_source = None
        if lib_pdfs:
            pick = st.selectbox("분석할 논문 선택",
                                ["(직접 업로드)"] + [p.name for p in lib_pdfs])
            if pick != "(직접 업로드)":
                pdf_source = PDF_DIR / pick
        one_off = st.file_uploader("또는 분석할 논문 PDF 업로드", type="pdf", key="analyze_pdf")

        analyze_option = st.selectbox("분석 유형 선택",
            ["핵심 내용 요약", "연구 방법 분석", "이론적 배경 정리", "연구 결과 요약", "비판적 검토"])

        if st.button("분석 시작", use_container_width=True, disabled=not (pdf_source or one_off)):
            try:
                if one_off:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(one_off.read())
                        tmp_path = tmp.name
                    with st.spinner("PDF 읽는 중..."):
                        pdf_text = extract_pdf_text(tmp_path)
                    os.unlink(tmp_path)
                else:
                    with st.spinner("PDF 읽는 중..."):
                        pdf_text = extract_pdf_text(str(pdf_source))
            except Exception as e:
                pdf_text = ""
                st.error(f"❌ PDF를 읽지 못했어요: {e}")
            if pdf_text.strip():
                st.success(f"PDF 로드 완료 ({len(pdf_text):,}자)")
                prompt = f"다음 논문을 '{analyze_option}' 관점에서 분석해주세요:\n\n{pdf_text[:8000]}"
                with st.spinner("분석 중..."):
                    analysis_out = chat_with_claude([{"role": "user", "content": prompt}])
                st.session_state["pdf_analysis"] = {
                    "name": (one_off.name if one_off else pdf_source.name),
                    "option": analyze_option, "result": analysis_out,
                }
        _pa = st.session_state.get("pdf_analysis")
        if _pa:
            st.markdown(f"### 분석 결과 — {_pa['name']} · {_pa['option']}")
            st.markdown(_pa["result"])
            st.download_button("📋 분석 저장 (.txt)", _pa["result"],
                               file_name=f"{_pa['name'][:20]}_분석.txt",
                               mime="text/plain", key="pdf_analysis_dl")

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
        st.warning("먼저 '📚 단락 작성 · 논문 분석' 모드에서 PDF를 업로드하고 '문헌 학습 시작'을 눌러주세요. (또는 위에서 🌐 외부 검색을 선택하세요)")
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
        type=["xlsx", "xls", "sav", "csv", "txt", "docx", "hwp", "hwpx"],
        help="Excel(.xlsx), SPSS(.sav), CSV(.csv), 인터뷰 텍스트(.txt/.docx/.hwp/.hwpx) 지원",
    )
    research_context = st.text_input(
        "연구 맥락 (선택)",
        placeholder="예: 대학생 블렌디드 러닝 경험 설문조사",
        help="어떤 연구인지 간단히 설명하면 더 정확한 제안을 드릴 수 있어요",
    )

    # ── 코딩북 (변수 설명서) — 프로젝트별 영구 저장 ─────────
    CODEBOOK_FILE = PROJ_DIR / "codebook.txt"
    _cb_exists = CODEBOOK_FILE.exists()
    with st.expander("📖 코딩북 (변수 설명서)" + (" — 등록됨 ✅" if _cb_exists else " — 올리면 분석 정확도가 크게 올라가요"),
                     expanded=False):
        st.caption("변수명이 Q1, V3처럼 돼 있으면 AI가 의미를 알 수 없어요. "
                   "문항 내용·값 의미(예: 성별 1=남, 2=여)가 담긴 코딩북을 올리면 "
                   "설계 제안·분석 실행·결과 해석 모두 변수의 실제 의미를 반영합니다. "
                   "이 프로젝트에 계속 저장돼요. (SPSS .sav는 레이블이 내장돼 있어 자동 반영)")
        cb_up = st.file_uploader(
            "코딩북 업로드",
            type=["xlsx", "xls", "csv", "txt", "docx", "hwp", "hwpx", "pdf"],
            key="codebook_up",
        )
        if cb_up:
            try:
                _cb_text = load_codebook_text(cb_up)
                if _cb_text.strip():
                    CODEBOOK_FILE.write_text(_cb_text, encoding="utf-8")
                    st.success(f"✅ 코딩북 저장됨 ({len(_cb_text):,}자)")
                else:
                    st.warning("파일에서 내용을 추출하지 못했어요.")
            except Exception as e:
                st.error(f"❌ 코딩북을 읽지 못했어요: {e}")
        if CODEBOOK_FILE.exists():
            _cb_preview = CODEBOOK_FILE.read_text(encoding="utf-8")
            st.text(_cb_preview[:800] + ("..." if len(_cb_preview) > 800 else ""))
            if st.button("🗑️ 코딩북 삭제", key="cb_del"):
                CODEBOOK_FILE.unlink()
                st.rerun()

    codebook_text = CODEBOOK_FILE.read_text(encoding="utf-8") if CODEBOOK_FILE.exists() else ""

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

                codebook_section = (
                    f"\n\n## 코딩북 (변수의 실제 의미 — 반드시 반영할 것)\n{codebook_text[:6000]}"
                    + ("\n(코딩북이 길어 일부만 표시)" if len(codebook_text) > 6000 else "")
                ) if codebook_text else ""

                prompt = f"""다음 {data_type_label}의 구조를 분석하여 {analysis_goal}을 제안해주세요.{context_line}

## 데이터 구조
{summary}{codebook_section}

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

                if data_type == "quantitative":
                    prompt += """

마지막에, 위에서 제안한 분석 중 아래 도구로 바로 실행 가능한 것을 JSON으로 정리하세요. 반드시 이 형식으로:

[실행 스펙]
```json
[
  {"method": "ttest_ind", "dv": "변수명", "group": "변수명", "설명": "집단에 따른 ○○ 차이 (독립표본 t검정)"}
]
```

사용 가능한 method와 인자:
- ttest_ind: dv(수치형), group(2개 집단)
- ttest_rel: var1, var2 (짝지은 수치형)
- anova: dv, group(3개 이상 집단)
- correlation: cols(수치형 변수명 배열, 2개 이상)
- chisquare: var1, var2 (범주형)
- cronbach: cols(같은 척도 문항 배열)
- regression: dv, ivs(독립변수명 배열)
- hlm: dv, ivs(배열), group(상위수준 집단변수)
- lcsm: waves(시점 순서 변수명 배열), group(선택, 다집단 비교)
- lca: cols(지표 변수명 배열)

규칙: 변수명은 [데이터 구조]의 열 이름과 정확히 일치 (코딩북의 문항 설명이 아니라 실제 열 이름). 실행 불가한 제안(SEM 등)은 JSON에서 제외. 최대 6개. "설명"은 코딩북의 변수 의미를 반영한 한 줄 한국어."""

                with st.spinner("Claude가 분석 설계 중..."):
                    result = chat_with_claude([{"role": "user", "content": prompt}])

                design_text, design_specs = result, []
                _mspec = re.search(r"\[실행 스펙\].*?```json\s*(.*?)```", result, re.S)
                if _mspec:
                    try:
                        design_specs = json.loads(_mspec.group(1))
                        if not isinstance(design_specs, list):
                            design_specs = []
                    except Exception:
                        design_specs = []
                    design_text = result[:_mspec.start()].strip()
                st.session_state["design_result"] = design_text
                st.session_state["design_specs"] = design_specs if data_type == "quantitative" else []

            if st.session_state.get("design_result"):
                st.markdown("### 분석 설계 제안")
                st.markdown(st.session_state["design_result"])
                st.download_button(
                    "📋 분석 설계 저장 (.txt)",
                    st.session_state["design_result"],
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

                def _interp_stats(res_summary):
                    """실행 결과를 논문용 결과 문장으로 해석 (수동/제안 실행 공용)"""
                    _ctx = f"\n연구 맥락: {research_context}" if research_context else ""
                    _cb = (f"\n\n## 코딩북 (변수의 실제 의미 — 해석에 반영할 것)\n{codebook_text[:3000]}"
                           if codebook_text else "")
                    _p = f"""다음은 실제 데이터로 방금 실행한 통계 분석 결과입니다. 이 결과를 논문의 '연구 결과' 절에 쓸 수 있도록 해석해주세요.{_ctx}{_cb}

## 분석 결과
{res_summary}

작성 지침:
1. APA 스타일 통계치 표기(t, F, χ², β, p, 효과크기 등)를 포함한 학술적 한국어 문장으로 서술하세요.
2. 효과크기의 실질적 의미를 함께 해석하세요.
3. 유의하지 않은 결과도 있는 그대로 서술하세요. 결과를 긍정적으로 왜곡하지 마세요.
4. 위 분석 결과에 없는 수치·통계량은 절대 만들어내지 마세요.
5. 마지막에 '해석 시 유의점' 1~2가지를 덧붙이세요."""
                    with st.spinner("Claude가 결과를 논문 문장으로 작성 중..."):
                        return chat_with_claude([{"role": "user", "content": _p}])

                def _proposal_fn(spec):
                    """설계 제안 JSON 스펙 → 실행 함수 (변수 검증 포함). 불가하면 None."""
                    _m = spec.get("method")
                    _cols = set(data.columns)

                    def ok(*keys, list_keys=()):
                        for k in keys:
                            v = spec.get(k)
                            if not v or (isinstance(v, str) and v not in _cols):
                                return False
                        for k in list_keys:
                            v = spec.get(k)
                            if not isinstance(v, list) or not v or not all(c in _cols for c in v):
                                return False
                        return True

                    if _m == "ttest_ind" and ok("dv", "group"):
                        return lambda: run_ttest_ind(data, spec["dv"], spec["group"])
                    if _m == "ttest_rel" and ok("var1", "var2"):
                        return lambda: run_ttest_rel(data, spec["var1"], spec["var2"])
                    if _m == "anova" and ok("dv", "group"):
                        return lambda: run_anova(data, spec["dv"], spec["group"])
                    if _m == "correlation" and ok(list_keys=("cols",)):
                        return lambda: run_correlation(data, spec["cols"],
                                                       spec.get("corr_method", "pearson"))
                    if _m == "chisquare" and ok("var1", "var2"):
                        return lambda: run_chisquare(data, spec["var1"], spec["var2"])
                    if _m == "cronbach" and ok(list_keys=("cols",)):
                        return lambda: run_cronbach(data, spec["cols"])
                    if _m == "regression" and ok("dv", list_keys=("ivs",)):
                        return lambda: run_regression(data, spec["dv"], spec["ivs"])
                    if _m == "hlm" and ok("dv", "group", list_keys=("ivs",)):
                        return lambda: run_hlm(data, spec["dv"], spec["ivs"], spec["group"])
                    if _m == "lcsm" and ok(list_keys=("waves",)):
                        if spec.get("group") and spec["group"] in _cols:
                            return lambda: run_lcsm_multigroup(data, spec["waves"], spec["group"])
                        return lambda: run_lcsm(data, spec["waves"])
                    if _m == "lca" and ok(list_keys=("cols",)):
                        return lambda: run_lca(data, spec["cols"],
                                               n_classes=int(spec.get("n_classes") or 0))
                    return None

                _specs = st.session_state.get("design_specs") or []
                if _specs:
                    st.markdown("#### ⚡ 제안된 분석 바로 실행")
                    st.caption("분석 설계에서 제안된 방법을 클릭 한 번으로 같은 데이터에 바로 실행해요. 결과는 아래와 '저장된 분석 결과'에 나타납니다.")
                    for _pi, _sp in enumerate(_specs[:6]):
                        _plabel = str(_sp.get("설명") or _sp.get("method", "분석"))
                        _pfn = _proposal_fn(_sp) if isinstance(_sp, dict) else None
                        if _pfn is None:
                            st.caption(f"▫️ {_plabel} — 변수명 불일치로 실행 불가 (아래에서 직접 변수를 선택해 실행하세요)")
                            continue
                        if st.button(f"▶️ {_plabel}", key=f"prop_run_{_pi}", use_container_width=True):
                            _pres = None
                            try:
                                with st.spinner("통계 계산 중..."):
                                    _pres = _pfn()
                            except ValueError as _ve:
                                st.error(f"⚠️ {_ve}")
                            except Exception as _ex:
                                st.error(f"❌ 분석 실패: {_ex}")
                            if _pres:
                                _pint = _interp_stats(_pres["summary"])
                                st.session_state["stats_run"] = {
                                    "analysis": _plabel, "result": _pres, "interp": _pint,
                                }
                                try:
                                    save_analysis(PROJ_DIR, _plabel, _pres, _pint,
                                                  note=uploaded_data.name)
                                except Exception:
                                    pass
                    st.divider()

                analysis = st.selectbox("분석 방법", [
                    "독립표본 t검정 (두 집단 평균 비교)",
                    "대응표본 t검정 (사전·사후 등 짝지은 비교)",
                    "일원분산분석 ANOVA (3개 이상 집단)",
                    "상관분석 (Pearson/Spearman)",
                    "카이제곱 검정 (범주형 × 범주형)",
                    "신뢰도 분석 (Cronbach's α)",
                    "회귀분석 (중다회귀)",
                    "다층모형 HLM (임의절편)",
                    "잠재변화점수 LCSM (사전-사후 변화)",
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
                elif analysis.startswith("잠재변화점수"):
                    waves = st.multiselect(
                        "시점 변수 — 시간 순서대로 선택 (예: 사전점수 → 사후점수 → 추후점수)",
                        num_cols,
                        help="선택한 순서가 시간 순서로 사용돼요. 2개(사전-사후)부터 가능합니다.",
                    )
                    lcsm_mg = st.selectbox("다집단 비교 (선택)", ["사용 안 함"] + all_cols,
                                           key="lcsm_mg",
                                           help="집단변수를 선택하면 집단별 변화량·비례변화를 비교해요.")
                    if lcsm_mg == "사용 안 함":
                        run_fn = lambda: run_lcsm(data, waves)
                    else:
                        run_fn = lambda: run_lcsm_multigroup(data, waves, lcsm_mg)
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
                        interp = _interp_stats(res["summary"]) if interpret else None
                        st.session_state["stats_run"] = {
                            "analysis": analysis, "result": res, "interp": interp,
                        }
                        # 프로젝트별 영구 저장 (화면 이동·재접속해도 유지)
                        try:
                            save_analysis(PROJ_DIR, analysis, res, interp,
                                          note=uploaded_data.name)
                        except Exception:
                            st.warning("결과 저장에 실패했어요. (분석 결과는 아래에 표시됩니다)")

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

    # ── 저장된 분석 결과 (프로젝트별, 영구 보관) ──────────────
    st.divider()
    saved_records = load_analyses(PROJ_DIR)
    st.subheader(f"🗂️ 저장된 분석 결과 — {len(saved_records)}건")
    st.caption("분석을 실행하면 자동 저장돼요. 화면을 옮기거나 새로 접속해도 프로젝트별로 유지됩니다. (최근 30건)")
    if not saved_records:
        st.info("아직 저장된 분석이 없어요. 위에서 데이터를 업로드하고 분석을 실행해보세요.")
    for ridx, rec in enumerate(saved_records):
        note = f" · {rec['note']}" if rec.get("note") else ""
        with st.expander(f"📌 {rec['time']} · {rec['analysis']}{note}"):
            try:
                for tname, tdf in tables_from_record(rec).items():
                    st.markdown(f"**{tname}**")
                    st.dataframe(tdf, use_container_width=True)
            except Exception:
                st.text(rec.get("summary", "(표 복원 실패)"))
            if rec.get("interp"):
                st.markdown("**📝 논문용 결과 해석**")
                st.markdown(rec["interp"])
            dcol1, dcol2 = st.columns([1, 1])
            export_text = (
                f"[{rec['time']}] {rec['analysis']}{note}\n\n"
                f"== 결과 요약 ==\n{rec.get('summary', '')}\n\n"
                + (f"== 논문용 해석 ==\n{rec['interp']}" if rec.get("interp") else "")
            )
            dcol1.download_button(
                "📋 저장 (.txt)", export_text,
                file_name=f"분석결과_{rec['time'].replace(':', '').replace(' ', '_')}.txt",
                mime="text/plain", key=f"rec_dl_{ridx}",
            )
            if dcol2.button("🗑️ 삭제", key=f"rec_del_{ridx}"):
                delete_analysis(PROJ_DIR, ridx)
                st.rerun()

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

    corr_file = st.file_uploader(
        "Word(.docx), 한글(.hwp/.hwpx), 텍스트(.txt) 파일 업로드 (선택)",
        type=["docx", "hwp", "hwpx", "txt"],
        help="파일을 올리면 내용을 추출해 교정해요. 아래에 직접 입력해도 됩니다.",
    )
    corr_source = None
    if corr_file:
        try:
            _cname = corr_file.name.lower()
            if _cname.endswith(".docx"):
                from docx import Document
                _doc = Document(corr_file)
                corr_source = "\n".join(p.text for p in _doc.paragraphs if p.text.strip())
            elif _cname.endswith(".hwpx"):
                from data_analyzer import extract_hwpx_text
                corr_source = extract_hwpx_text(corr_file)
            elif _cname.endswith(".hwp"):
                from data_analyzer import extract_hwp_text
                corr_source = extract_hwp_text(corr_file)
            else:
                corr_source = corr_file.read().decode("utf-8", errors="ignore")
        except Exception as e:
            st.error(f"❌ 파일을 읽지 못했어요: {e}")
        if corr_source is not None and not corr_source.strip():
            st.warning("파일에서 텍스트를 찾지 못했어요. (스캔 이미지 문서는 지원되지 않아요)")
            corr_source = None

    corr_inplace_mode = False
    if corr_source and corr_file.name.lower().endswith((".docx", ".hwpx")):
        corr_method = st.radio(
            "교정 방식",
            ["🧩 원본 서식 유지 — 표·제목·레이아웃 보존 (권장)",
             "📝 텍스트만 — 범위 선택 가능"],
            horizontal=True,
            help="원본 서식 유지: 문서 구조는 그대로 두고 문단 텍스트만 교정해요. 표는 표대로 유지되고 수정된 단어만 빨간색이 됩니다.",
        )
        corr_inplace_mode = corr_method.startswith("🧩")

    if corr_source:
        paras = [p.strip() for p in corr_source.splitlines() if p.strip()]
        st.success(f"✅ '{corr_file.name}' 불러옴 — 문단 {len(paras)}개, 총 {len(corr_source):,}자")
        if corr_inplace_mode:
            text_input = corr_source  # 버튼 활성화용 (실제 교정은 원본 구조 기준)
        elif len(corr_source) > 6000:
            corr_scope = st.radio(
                "교정 범위", ["📄 전체 문서 (자동 분할 교정)", "✂️ 문단 범위 선택"],
                horizontal=True,
                help="전체 문서: 긴 글을 여러 구간으로 자동 분할해 차례로 교정한 뒤 이어붙여요.",
            )
            if corr_scope.startswith("✂️"):
                _acc, _end_default = 0, 1
                for _i, _p in enumerate(paras):
                    _acc += len(_p)
                    if _acc > 6000:
                        break
                    _end_default = _i + 1
                _rng = st.slider("교정할 문단 범위", 1, len(paras), (1, _end_default))
                text_input = "\n\n".join(paras[_rng[0] - 1:_rng[1]])
                st.caption(f"선택된 분량: 문단 {_rng[1] - _rng[0] + 1}개, {len(text_input):,}자")
            else:
                text_input = "\n\n".join(paras)
                _n_chunks = len(_split_for_correction(text_input))
                st.info(f"전체 {len(text_input):,}자를 {_n_chunks}개 구간으로 나눠 차례로 교정해요. "
                        f"(구간당 30초~1분, 예상 {_n_chunks}~{_n_chunks*2}분)")
        else:
            text_input = "\n\n".join(paras)
        with st.expander("불러온 내용 미리보기"):
            st.text(text_input[:2000] + ("..." if len(text_input) > 2000 else ""))
    else:
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

    if corr_inplace_mode and st.button("교정 시작", key="corr_inplace_start") and text_input:
        style_section = build_style_instruction(corr_style_profile) if use_my_style_corr else ""
        _raw = corr_file.getvalue()
        _is_docx = corr_file.name.lower().endswith(".docx")
        try:
            if _is_docx:
                _doc, _pobjs, _texts = ir.collect_docx(io.BytesIO(_raw))
            else:
                _state, _texts = ir.collect_hwpx(_raw)
        except Exception as e:
            st.error(f"❌ 문서 구조를 읽지 못했어요: {e}")
            _texts = []
        if _texts:
            _chunks = ir.chunk_numbered(_texts)
            _cmap, _notes, _trunc_any = {}, [], False
            _prog = st.progress(0, text=f"교정 중... (0/{len(_chunks)})") if len(_chunks) > 1 else None
            for _ci, (_start, _block) in enumerate(_chunks):
                _prompt = f"""다음은 문서에서 추출한 문단들입니다 (⟦번호⟧ 문단). 각 문단을 '{correction_type}' 관점에서 교정해주세요.
{style_section}
[문단들]
{_block}

규칙 (반드시 지킬 것):
1. 출력도 같은 형식으로: ⟦번호⟧ 교정된 문단. 입력의 모든 번호를 빠짐없이 같은 번호로 출력.
2. 문단을 합치거나 나누지 말고, 새 문단을 추가하지 마세요.
3. 제목, 표 셀의 숫자·변수명·짧은 항목명은 교정하지 말고 그대로 출력하세요.
4. 수정할 필요 없는 문단도 번호와 함께 원문 그대로 출력하세요.

마지막에:
[수정 설명]
- 주요 수정 내용을 항목별로 간단히"""
                with st.spinner(f"교정 중... ({_ci + 1}/{len(_chunks)} 구간)"):
                    _resp, _tr = chat_with_claude(
                        [{"role": "user", "content": _prompt}], return_truncated=True)
                _trunc_any = _trunc_any or _tr
                _m, _expl = ir.parse_numbered(_resp, _texts)
                _cmap.update(_m)
                if _expl:
                    _notes.append(f"[구간 {_ci + 1}]\n{_expl}" if len(_chunks) > 1 else _expl)
                if _prog:
                    _prog.progress((_ci + 1) / len(_chunks),
                                   text=f"교정 중... ({_ci + 1}/{len(_chunks)})")
            if _prog:
                _prog.empty()
            if _trunc_any:
                st.warning("⚠️ 일부 구간 출력이 잘렸을 수 있어요. 결과를 확인해주세요.")
            _corrected = [_cmap.get(_i, _t) for _i, _t in enumerate(_texts)]
            try:
                if _is_docx:
                    _out, _n = ir.apply_docx(_doc, _pobjs, _corrected)
                    _fname, _mime = "교정_서식유지.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                else:
                    _out, _n = ir.apply_hwpx(_state, _corrected)
                    _fname, _mime = "교정_서식유지.hwpx", "application/octet-stream"
                st.session_state["corr_inplace"] = {
                    "bytes": _out, "fname": _fname, "mime": _mime, "n": _n,
                    "texts": _texts, "corrected": _corrected,
                    "explanation": "\n\n".join(_notes),
                }
                st.session_state.pop("corr_result", None)
            except Exception as e:
                st.error(f"❌ 교정 결과를 문서에 반영하지 못했어요: {e}")

    if st.session_state.get("corr_inplace"):
        _ci_res = st.session_state["corr_inplace"]
        st.markdown(f"#### 교정 결과 (원본 서식 유지) — 문단 {_ci_res['n']}개 수정, "
                    f"<span style='color:#d32f2f'>빨간색</span> = 수정 부분",
                    unsafe_allow_html=True)
        _changed_pairs = [(i, o, c) for i, (o, c) in
                          enumerate(zip(_ci_res["texts"], _ci_res["corrected"]))
                          if o.strip() != (c or "").strip()]
        with st.expander(f"수정된 문단 미리보기 ({len(_changed_pairs)}개)", expanded=True):
            for _i, _o, _c in _changed_pairs[:50]:
                _ps = []
                for _seg, _chg in diff_segments(_o, _c):
                    _e = _html.escape(_seg)
                    _ps.append(f"<span style='color:#d32f2f;font-weight:600'>{_e}</span>" if _chg else _e)
                st.markdown(f"<div style='margin-bottom:10px'><b>문단 {_i + 1}</b><br>"
                            + "".join(_ps) + "</div>", unsafe_allow_html=True)
            if len(_changed_pairs) > 50:
                st.caption(f"...외 {len(_changed_pairs) - 50}개 (다운로드 파일에는 전부 반영)")
        if _ci_res["explanation"]:
            with st.expander("수정 설명 보기"):
                st.markdown(_ci_res["explanation"])
        st.download_button(
            f"📄 원본 서식 그대로 저장 ({_ci_res['fname'].split('.')[-1]}, 빨간 표시)",
            data=_ci_res["bytes"], file_name=_ci_res["fname"], mime=_ci_res["mime"],
            key="corr_inplace_dl",
        )

    if not corr_inplace_mode and st.button("교정 시작", key="corr_text_start") and text_input:
        style_section = build_style_instruction(corr_style_profile) if use_my_style_corr else ""
        chunks = _split_for_correction(text_input)
        correcteds, notes, truncated_any = [], [], False
        prog = st.progress(0, text=f"교정 중... (0/{len(chunks)} 구간)") if len(chunks) > 1 else None
        for ci, chunk in enumerate(chunks):
            prompt = f"""다음 글을 '{correction_type}' 관점에서 교정해주세요.
{style_section}
[원문]
{chunk}

출력 형식 (반드시 지킬 것):
[교정문]
(교정된 전체 글만 출력. 머리말·설명·마크다운 서식 없이 순수한 글만. 원문의 모든 문단을 빠짐없이 다룰 것)
[교정 끝]
[수정 설명]
- 무엇을 왜 바꿨는지 항목별로 간단히"""
            with st.spinner(f"교정 중... ({ci + 1}/{len(chunks)} 구간)"):
                corr_resp, _trunc = chat_with_claude(
                    [{"role": "user", "content": prompt}], return_truncated=True)
            truncated_any = truncated_any or _trunc
            _corr, _expl = _parse_correction(corr_resp)
            correcteds.append(_corr)
            if _expl:
                notes.append(f"[구간 {ci + 1}]\n{_expl}" if len(chunks) > 1 else _expl)
            if prog:
                prog.progress((ci + 1) / len(chunks),
                              text=f"교정 중... ({ci + 1}/{len(chunks)} 구간)")
        if prog:
            prog.empty()
        if truncated_any:
            st.warning("⚠️ 일부 구간의 출력이 잘렸을 수 있어요. 결과에서 빠진 부분이 보이면 해당 부분만 범위 선택으로 다시 교정해주세요.")
        st.session_state["corr_result"] = {
            "original": text_input,
            "corrected": "\n\n".join(correcteds),
            "explanation": "\n\n".join(notes),
        }
        st.session_state.pop("corr_inplace", None)

    corr_res = st.session_state.get("corr_result")
    if corr_res:
        st.markdown("#### 교정 결과 — <span style='color:#d32f2f'>빨간색</span> = 수정된 부분",
                    unsafe_allow_html=True)
        _parts = []
        for _seg, _chg in diff_segments(corr_res["original"], corr_res["corrected"]):
            _esc = _html.escape(_seg).replace("\n", "<br>")
            _parts.append(
                f"<span style='color:#d32f2f;font-weight:600'>{_esc}</span>" if _chg else _esc)
        st.markdown(
            "<div style='line-height:1.9;border:1px solid #ddd;border-radius:8px;"
            "padding:14px;background:#fafafa'>" + "".join(_parts) + "</div>",
            unsafe_allow_html=True)
        if corr_res["explanation"]:
            with st.expander("수정 설명 보기", expanded=True):
                st.markdown(corr_res["explanation"])
        dl1, dl2, dl3 = st.columns(3)
        dl1.download_button(
            "📄 Word (.docx, 빨간 표시)",
            data=to_word_redline(corr_res["original"], corr_res["corrected"],
                                 corr_res["explanation"]),
            file_name="교정결과_빨간표시.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="corr_docx",
        )
        dl2.download_button(
            "📄 한글 (.hwpx, 빨간 표시)",
            data=to_hwpx_redline(corr_res["original"], corr_res["corrected"],
                                 corr_res["explanation"]),
            file_name="교정결과_빨간표시.hwpx",
            mime="application/octet-stream",
            key="corr_hwpx",
        )
        dl3.download_button(
            "📋 교정문만 (.txt)", corr_res["corrected"],
            file_name="교정결과.txt", mime="text/plain", key="corr_txt",
        )

# ── 학술지 형식 · 참고문헌 변환 (통합) ────────────────────────
elif mode == "🔖 학술지 형식 · 참고문헌 변환":
    st.subheader("🔖 학술지 형식 맞추기 · 참고문헌 변환")
    tab_journal, tab_ref = st.tabs(["📐 학술지 형식 맞추기", "🔖 참고문헌 변환"])

    # ── 탭 1: 학술지 형식 맞추기 ──────────────────────────────
    with tab_journal:
        JFMT_FILE = PROJ_DIR / "journal_format.json"
        st.markdown("##### 1) 학술지 형식 규정 등록")
        st.caption("학술지 투고규정 웹주소를 넣거나 규정·템플릿 파일을 올리면, 그 형식(인용·구조·레이아웃)을 학습해요. 프로젝트에 저장됩니다.")

        jf_url = st.text_input("투고규정 웹주소 (선택)",
                               placeholder="예: https://학술지사이트/author-guidelines")
        jf_file = st.file_uploader("또는 규정·템플릿 파일 업로드 (선택)",
                                   type=["pdf", "docx", "hwp", "hwpx", "txt", "xlsx", "csv"],
                                   key="jfmt_src")
        if st.button("📖 형식 규칙 분석", use_container_width=True,
                     disabled=not (jf_url.strip() or jf_file)):
            src_text = ""
            try:
                if jf_file:
                    with st.spinner("규정 파일 읽는 중..."):
                        src_text = load_codebook_text(jf_file)
                elif jf_url.strip():
                    with st.spinner("웹페이지에서 규정 가져오는 중..."):
                        src_text = jf.fetch_url_text(jf_url)
            except Exception as e:
                st.error(f"❌ 규정을 가져오지 못했어요: {e}")
            if src_text.strip():
                rule_prompt = f"""다음은 한 학술지의 투고 규정(author guidelines)입니다. 원고를 이 학술지 형식에 맞게 편집하기 위한 '형식 규칙 요약'을 작성하세요.

[투고 규정 원문]
{src_text[:12000]}

다음을 포함해 한국어로 정리하세요:
1. 인용·참고문헌 스타일 (본문 인용 방식 + 참고문헌 목록 형식, 가능하면 예시)
2. 원고 구조 (제목/초록 단어수/키워드 개수/섹션 순서/소제목 단계)
3. 표·그림 캡션 규칙
4. 기타 표기 규칙(숫자, 약어, 단위 등)
규정에 없는 항목은 "규정에 명시 없음"이라고 쓰세요. 지어내지 마세요.

마지막에 문서 레이아웃을 아래 형식의 JSON으로 출력하세요(규정에 없으면 일반 기본값 추정, 값은 숫자만):
[레이아웃]
```json
{{"font": "한글 글꼴명 또는 빈문자열", "size_pt": 10, "line_spacing": 1.6, "margin_cm": {{"top": 2.5, "bottom": 2.5, "left": 2.5, "right": 2.5}}}}
```"""
                with st.spinner("Claude가 형식 규칙 분석 중..."):
                    rule_resp = chat_with_claude([{"role": "user", "content": rule_prompt}])
                layout = dict(jf.DEFAULT_LAYOUT)
                _lm = re.search(r"\[레이아웃\].*?```json\s*(.*?)```", rule_resp, re.S)
                rulebook = rule_resp
                if _lm:
                    try:
                        layout = {**jf.DEFAULT_LAYOUT, **json.loads(_lm.group(1))}
                    except Exception:
                        pass
                    rulebook = rule_resp[:_lm.start()].strip()
                JFMT_FILE.write_text(json.dumps(
                    {"rulebook": rulebook, "layout": layout,
                     "source": (jf_file.name if jf_file else jf_url)},
                    ensure_ascii=False), encoding="utf-8")
                st.success("✅ 형식 규칙 저장 완료")
                st.rerun()

        saved_fmt = None
        if JFMT_FILE.exists():
            try:
                saved_fmt = json.loads(JFMT_FILE.read_text(encoding="utf-8"))
            except Exception:
                saved_fmt = None
        if saved_fmt:
            st.success(f"✅ 등록된 형식: {saved_fmt.get('source', '(파일)')}")
            with st.expander("형식 규칙 보기"):
                st.markdown(saved_fmt.get("rulebook", ""))
                st.caption(f"레이아웃: {saved_fmt.get('layout')}")
            if st.button("🗑️ 형식 삭제", key="jfmt_del"):
                JFMT_FILE.unlink()
                st.rerun()

            st.divider()
            st.markdown("##### 2) 원고를 이 형식으로 편집")
            ms_file = st.file_uploader("원고 업로드 (docx/hwpx/txt)",
                                       type=["docx", "hwp", "hwpx", "txt"], key="jfmt_ms")
            if st.button("✨ 형식 맞추기 실행", use_container_width=True, disabled=not ms_file):
                try:
                    with st.spinner("원고 읽는 중..."):
                        ms_text = load_codebook_text(ms_file)
                except Exception as e:
                    ms_text = ""
                    st.error(f"❌ 원고를 읽지 못했어요: {e}")
                if ms_text.strip():
                    chunks = _split_for_correction(ms_text, limit=4000)
                    if len(chunks) > 1:
                        st.info(f"원고가 길어 {len(chunks)}개 구간으로 나눠 편집해요.")
                    parts = []
                    prog = st.progress(0.0) if len(chunks) > 1 else None
                    for ci, chunk in enumerate(chunks):
                        fmt_prompt = f"""아래 [형식 규칙]에 맞게 [원고]를 재편집하세요.

[형식 규칙]
{saved_fmt['rulebook'][:6000]}

[원고]
{chunk}

지침:
1. 인용·참고문헌 표기를 규칙의 스타일로 바꾸세요. 학술지명에 따옴표 금지.
2. 섹션 구조와 소제목을 규칙에 맞게 정리하되, 제목은 마크다운으로 표시: 대제목 "# ", 중제목 "## ", 소제목 "### ". 본문은 그대로 문단.
3. 원고의 내용을 빠짐없이 포함하고, 없는 사실·인용을 지어내지 마세요.
4. 재편집된 원고만 출력하세요(설명 없이)."""
                        with st.spinner(f"형식 맞추는 중... ({ci + 1}/{len(chunks)})"):
                            parts.append(chat_with_claude([{"role": "user", "content": fmt_prompt}]))
                        if prog:
                            prog.progress((ci + 1) / len(chunks))
                    if prog:
                        prog.empty()
                    formatted_md = "\n\n".join(parts)
                    st.session_state["jfmt_result"] = {
                        "md": formatted_md, "layout": saved_fmt.get("layout"),
                        "name": ms_file.name.rsplit(".", 1)[0],
                    }

            jr = st.session_state.get("jfmt_result")
            if jr:
                st.divider()
                st.markdown("##### 편집 결과")
                st.markdown(jr["md"][:4000] + ("\n\n...(이하 생략, 파일에는 전체 포함)" if len(jr["md"]) > 4000 else ""))
                d1, d2, d3 = st.columns(3)
                d1.download_button(
                    "📄 Word (.docx, 형식 적용)",
                    data=jf.build_docx_from_markdown(jr["md"], jr["layout"]),
                    file_name=f"{jr['name']}_형식적용.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="jfmt_docx")
                d2.download_button(
                    "📄 한글 (.hwpx)",
                    data=jf.build_hwpx_from_markdown(jr["md"]),
                    file_name=f"{jr['name']}_형식적용.hwpx",
                    mime="application/octet-stream", key="jfmt_hwpx")
                d3.download_button(
                    "📋 텍스트 (.txt)", jr["md"],
                    file_name=f"{jr['name']}_형식적용.txt",
                    mime="text/plain", key="jfmt_txt")
                st.caption("※ 레이아웃(여백·글꼴·줄간격)은 Word(.docx)에 정확히 적용돼요. hwpx는 내용 위주입니다.")
        else:
            st.info("먼저 위에서 학술지 형식 규정을 등록하세요.")

    # ── 탭 2: 참고문헌 변환 ───────────────────────────────────
    with tab_ref:
        ref_input = st.text_area("참고문헌 정보를 입력하세요", height=150,
                                 placeholder="예: 저자명, 출판연도, 제목, 학술지명, 권호, 페이지")
        ref_format = st.selectbox("변환할 형식",
                                  ["APA 7판", "MLA", "Chicago", "한국 학술지 형식",
                                   "등록한 학술지 형식"])
        if st.button("변환 시작", key="ref_convert") and ref_input:
            fmt_desc = ref_format
            if ref_format == "등록한 학술지 형식":
                _sf = None
                if (PROJ_DIR / "journal_format.json").exists():
                    try:
                        _sf = json.loads((PROJ_DIR / "journal_format.json").read_text(encoding="utf-8"))
                    except Exception:
                        _sf = None
                if _sf:
                    fmt_desc = f"아래 학술지 형식 규칙에 따른 참고문헌 형식:\n{_sf['rulebook'][:3000]}"
                else:
                    st.warning("등록된 학술지 형식이 없어요. '학술지 형식 맞추기' 탭에서 먼저 등록하세요.")
                    fmt_desc = "APA 7판"
            prompt = f"""다음 정보를 {fmt_desc} 으로 참고문헌을 변환해주세요:

{ref_input}

표기 규칙:
- 학술지 이름에 따옴표를 붙이지 마세요. 규정에 따라 학술지명과 권 번호는 이탤릭으로 표기하세요.
- 논문 제목에도 따옴표를 붙이지 마세요 (해당 형식이 요구하는 경우에만 예외)."""
            with st.spinner("변환 중..."):
                st.session_state["ref_result"] = chat_with_claude(
                    [{"role": "user", "content": prompt}])
        if st.session_state.get("ref_result"):
            st.markdown("#### 변환 결과")
            st.markdown(st.session_state["ref_result"])
            st.download_button("📋 저장 (.txt)", st.session_state["ref_result"],
                               file_name="참고문헌.txt", mime="text/plain", key="ref_dl")

# ── 대화 기록 표시 ────────────────────────────────────────────
if mode in ["💬 자유 질문", "🏗️ 논문 구조 설계"]:
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
