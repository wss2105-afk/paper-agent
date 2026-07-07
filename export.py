import difflib
import io
import re

from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


def to_word(title: str, content: str, topic: str = "") -> bytes:
    doc = Document()

    # 제목
    heading = doc.add_heading(title, level=1)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER

    if topic:
        sub = doc.add_paragraph(f"주제: {topic}")
        sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
        sub.runs[0].font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    doc.add_paragraph()

    # 본문 — "**참고문헌:**" 기준으로 본문/참고문헌 분리
    if "**참고문헌:**" in content:
        body, refs = content.split("**참고문헌:**", 1)
    elif "참고문헌:" in content:
        body, refs = content.split("참고문헌:", 1)
    else:
        body, refs = content, ""

    # 본문 단락
    for line in body.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        p = doc.add_paragraph(line)
        p.style.font.size = Pt(11)

    # 참고문헌 섹션
    if refs.strip():
        doc.add_paragraph()
        ref_heading = doc.add_heading("참고문헌", level=2)
        for line in refs.strip().splitlines():
            line = line.strip().lstrip("-").strip()
            if line:
                doc.add_paragraph(line, style="List Bullet")

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def diff_segments(original: str, corrected: str):
    """원문 대비 교정문의 단어 단위 diff.
    교정문 기준 [(텍스트, 변경여부)] 목록 반환 — 변경/추가된 조각만 True."""
    o = re.split(r"(\s+)", original)
    c = re.split(r"(\s+)", corrected)
    sm = difflib.SequenceMatcher(None, o, c, autojunk=False)
    segs = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        chunk = "".join(c[j1:j2])
        if not chunk:
            continue
        # 공백만 바뀐 조각은 빨간 표시 대상에서 제외
        changed = tag != "equal" and bool(chunk.strip())
        segs.append((chunk, changed))
    return segs


def to_word_redline(original: str, corrected: str, explanation: str = "",
                    title: str = "교정 결과") -> bytes:
    """교정문을 Word로 내보내되, 원문에서 수정된 부분을 빨간 글자로 표시."""
    doc = Document()
    heading = doc.add_heading(title, level=1)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    note = doc.add_paragraph("※ 빨간색 글자 = 원문에서 수정·추가된 부분")
    note.runs[0].font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    note.runs[0].font.size = Pt(9)
    doc.add_paragraph()

    para = doc.add_paragraph()
    for text, changed in diff_segments(original, corrected):
        parts = text.split("\n")
        for k, part in enumerate(parts):
            if k > 0:
                para = doc.add_paragraph()
            if part:
                run = para.add_run(part)
                run.font.size = Pt(11)
                if changed:
                    run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)

    if explanation.strip():
        doc.add_paragraph()
        doc.add_heading("수정 설명", level=2)
        for line in explanation.strip().splitlines():
            line = line.strip()
            if line:
                doc.add_paragraph(line)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ── HWPX(한글) 내보내기 ───────────────────────────────────────
# assets/hwpx_template.hwpx: 한글이 만든 빈 문서(hwpxlib 프로젝트, Apache-2.0).
# 유효한 실제 파일을 템플릿으로 쓰고 본문/글자속성만 주입해 호환성을 확보한다.

_HWPX_TEMPLATE = None
_RED_CHARPR_ID = "100"  # 템플릿의 charPr id(0~6)와 겹치지 않는 값


def _hwpx_template_bytes():
    global _HWPX_TEMPLATE
    if _HWPX_TEMPLATE is None:
        from pathlib import Path
        _HWPX_TEMPLATE = (Path(__file__).parent / "assets" / "hwpx_template.hwpx").read_bytes()
    return _HWPX_TEMPLATE


def _hwpx_escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _hwpx_paragraph(runs, para_id) -> str:
    """runs: [(텍스트, 빨간여부)] → <hp:p> XML"""
    run_xml = "".join(
        f'<hp:run charPrIDRef="{_RED_CHARPR_ID if red else "0"}">'
        f"<hp:t>{_hwpx_escape(t)}</hp:t></hp:run>"
        for t, red in runs if t
    ) or '<hp:run charPrIDRef="0"><hp:t></hp:t></hp:run>'
    return (f'<hp:p id="{para_id}" paraPrIDRef="3" styleIDRef="0" '
            f'pageBreak="0" columnBreak="0" merged="0">{run_xml}</hp:p>')


def to_hwpx_redline(original: str, corrected: str, explanation: str = "") -> bytes:
    """교정문을 한글(.hwpx)로 내보내되, 수정된 부분을 빨간 글자로 표시."""
    import zipfile

    src = zipfile.ZipFile(io.BytesIO(_hwpx_template_bytes()))
    header = src.read("Contents/header.xml").decode("utf-8")
    section = src.read("Contents/section0.xml").decode("utf-8")

    # 1) 빨간 글자 charPr 추가: 기본 charPr(id=0)을 복제해 색만 변경
    m = re.search(r'<hh:charPr id="0".*?</hh:charPr>', header, re.S)
    if not m:
        raise ValueError("HWPX 템플릿에서 기본 글자 속성을 찾지 못했어요.")
    red_charpr = (m.group(0)
                  .replace('id="0"', f'id="{_RED_CHARPR_ID}"', 1)
                  .replace('textColor="#000000"', 'textColor="#C00000"', 1))
    header = header.replace("</hh:charProperties>", red_charpr + "</hh:charProperties>", 1)
    header = re.sub(r'(<hh:charProperties itemCnt=")(\d+)(")',
                    lambda mm: mm.group(1) + str(int(mm.group(2)) + 1) + mm.group(3),
                    header, count=1)

    # 2) 본문 문단 구성 (문단 경계 = 개행)
    paras, cur = [], []
    for text, changed in diff_segments(original, corrected):
        parts = text.split("\n")
        for k, part in enumerate(parts):
            if k > 0:
                paras.append(cur)
                cur = []
            if part:
                cur.append((part, changed))
    paras.append(cur)

    pid = 90000000
    body_xml = _hwpx_paragraph([("※ 빨간색 글자 = 원문에서 수정·추가된 부분", False)], pid)
    for p_runs in paras:
        pid += 1
        body_xml += _hwpx_paragraph(p_runs, pid)
    if explanation.strip():
        pid += 1
        body_xml += _hwpx_paragraph([("", False)], pid)
        pid += 1
        body_xml += _hwpx_paragraph([("[수정 설명]", False)], pid)
        for line in explanation.strip().splitlines():
            if line.strip():
                pid += 1
                body_xml += _hwpx_paragraph([(line.strip(), False)], pid)

    section = section.replace("</hs:sec>", body_xml + "</hs:sec>", 1)

    # 3) zip 재조립 (mimetype은 관례대로 무압축 선두 배치)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", src.read("mimetype"), compress_type=zipfile.ZIP_STORED)
        for name in src.namelist():
            if name == "mimetype":
                continue
            elif name == "Contents/header.xml":
                zf.writestr(name, header)
            elif name == "Contents/section0.xml":
                zf.writestr(name, section)
            else:
                zf.writestr(name, src.read(name))
    src.close()
    return out.getvalue()


def to_markdown(title: str, content: str, topic: str = "") -> str:
    lines = [f"# {title}"]
    if topic:
        lines.append(f"> 주제: {topic}")
    lines.append("")
    lines.append(content)
    return "\n".join(lines)
