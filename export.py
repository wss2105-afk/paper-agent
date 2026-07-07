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


def to_markdown(title: str, content: str, topic: str = "") -> str:
    lines = [f"# {title}"]
    if topic:
        lines.append(f"> 주제: {topic}")
    lines.append("")
    lines.append(content)
    return "\n".join(lines)
