import io
import pandas as pd


def load_file(uploaded_file):
    """Excel, SPSS, CSV, 텍스트, Word 파일 로드"""
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

    else:
        raise ValueError(
            "지원 형식: Excel(.xlsx/.xls), SPSS(.sav), CSV(.csv), "
            "텍스트(.txt), Word(.docx)"
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
