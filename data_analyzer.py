import io
import re
import zipfile
from xml.etree import ElementTree

import pandas as pd


def extract_hwpx_text(file_like):
    """HWPX(한글 표준 XML 포맷) 본문 텍스트 추출.
    HWPX는 zip 압축 안에 Contents/section*.xml 로 본문을 담는다.
    문서 순서대로 순회하며 문단(<hp:p>) 단위로 텍스트(<hp:t>)를 모은다."""
    try:
        zf = zipfile.ZipFile(file_like)
    except zipfile.BadZipFile:
        raise ValueError(
            "HWPX 형식이 아니에요. 구형 .hwp 파일이면 한글에서 "
            "'다른 이름으로 저장 → HWPX'로 저장한 뒤 올려주세요."
        )
    with zf:
        sections = sorted(
            n for n in zf.namelist()
            if re.match(r"Contents/section\d+\.xml$", n)
        )
        if not sections:
            raise ValueError("HWPX 본문(section XML)을 찾을 수 없어요. 파일이 손상됐을 수 있어요.")
        paras = [""]
        for sec_name in sections:
            root = ElementTree.fromstring(zf.read(sec_name))
            for el in root.iter():
                tag = el.tag.split("}")[-1]
                if tag == "p":  # 새 문단 (표 안 문단 포함, 문서 순서 유지)
                    if paras[-1].strip():
                        paras.append("")
                elif tag == "t" and el.text:
                    paras[-1] += el.text
    return "\n".join(p for p in paras if p.strip())


def load_file(uploaded_file):
    """Excel, SPSS, CSV, 텍스트, Word, HWPX 파일 로드"""
    name = uploaded_file.name.lower()

    if name.endswith(".sav"):
        import pyreadstat
        with io.BytesIO(uploaded_file.read()) as buf:
            df, meta = pyreadstat.read_sav(buf)
        return "quantitative", df, meta

    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
        return "quantitative", df, None

    elif name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return "quantitative", df, None

    elif name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        return "qualitative", text, None

    elif name.endswith(".docx"):
        from docx import Document
        doc = Document(uploaded_file)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return "qualitative", text, None

    elif name.endswith(".hwpx"):
        text = extract_hwpx_text(uploaded_file)
        return "qualitative", text, None

    else:
        raise ValueError(
            "지원 형식: Excel(.xlsx/.xls), SPSS(.sav), CSV(.csv), "
            "텍스트(.txt), Word(.docx), 한글(.hwpx)"
        )


def summarize_dataframe(df, meta=None):
    """정량 데이터 구조 요약 (Claude에게 전달용)"""
    lines = []
    lines.append(f"- 총 행(응답자) 수: {len(df)}")
    lines.append(f"- 총 열(변수) 수: {len(df.columns)}")
    lines.append("")
    lines.append("## 변수 목록")

    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        missing_pct = missing / len(df) * 100

        label = ""
        if meta and hasattr(meta, "column_labels") and meta.column_labels:
            label_val = meta.column_labels.get(col, "")
            if label_val:
                label = f" [{label_val}]"

        if pd.api.types.is_numeric_dtype(dtype):
            mean = df[col].mean()
            std = df[col].std()
            min_v = df[col].min()
            max_v = df[col].max()
            lines.append(
                f"- **{col}**{label}: 수치형 | 평균={mean:.2f}, SD={std:.2f}, "
                f"범위={min_v}~{max_v}, 결측={missing}({missing_pct:.1f}%)"
            )
        else:
            unique = df[col].nunique()
            top_vals = df[col].value_counts().head(5).to_dict()
            lines.append(
                f"- **{col}**{label}: 범주형 | 고유값={unique}개, "
                f"주요값={top_vals}, 결측={missing}({missing_pct:.1f}%)"
            )

    return "\n".join(lines)


def summarize_interview(text):
    """인터뷰 텍스트 구조 요약"""
    lines = text.splitlines()
    non_empty = [l for l in lines if l.strip()]
    word_count = len(text.split())
    char_count = len(text)

    # 간단한 참여자 구분 시도 (Q:, A:, 면담자, 참여자 등 패턴)
    speakers = set()
    for line in non_empty:
        for prefix in ["Q:", "A:", "면담자:", "참여자:", "연구자:", "교사:", "학생:"]:
            if line.strip().startswith(prefix):
                speakers.add(prefix.rstrip(":"))

    summary = [
        f"- 총 글자 수: {char_count:,}",
        f"- 총 단어 수: {word_count:,}",
        f"- 총 줄 수: {len(non_empty)}",
    ]
    if speakers:
        summary.append(f"- 감지된 화자: {', '.join(sorted(speakers))}")
    summary.append("")
    summary.append("## 텍스트 앞부분 미리보기")
    summary.append(text[:800] + ("..." if len(text) > 800 else ""))

    return "\n".join(summary)


def get_preview(df, n=5):
    return df.head(n)


def get_basic_stats(df):
    numeric_cols = df.select_dtypes(include="number")
    if numeric_cols.empty:
        return None
    return numeric_cols.describe().round(2)
