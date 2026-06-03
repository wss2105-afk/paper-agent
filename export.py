import io
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


def to_markdown(title: str, content: str, topic: str = "") -> str:
    lines = [f"# {title}"]
    if topic:
        lines.append(f"> 주제: {topic}")
    lines.append("")
    lines.append(content)
    return "\n".join(lines)
