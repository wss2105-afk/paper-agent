"""
SJR (Scimago Journal Rankings) quartile lookup for education / educational technology.
Data sourced from scimagojr.com (Education & Educational Technology subject areas).
Covers journals most likely to appear in searches on learning, instruction, and ed-tech.
"""

# key: lowercase journal name (stripped), value: (quartile_str, subject_hint)
_RAW: dict[str, tuple[str, str]] = {
    # ── Q1 ──────────────────────────────────────────────────────
    "review of educational research": ("Q1", "Education"),
    "educational researcher": ("Q1", "Education"),
    "american educational research journal": ("Q1", "Education"),
    "learning and instruction": ("Q1", "Education"),
    "educational psychology review": ("Q1", "Education"),
    "journal of the learning sciences": ("Q1", "Education"),
    "educational psychologist": ("Q1", "Education"),
    "contemporary educational psychology": ("Q1", "Education"),
    "metacognition and learning": ("Q1", "Education"),
    "educational research review": ("Q1", "Education"),
    "higher education": ("Q1", "Education"),
    "teaching and teacher education": ("Q1", "Education"),
    "computers & education": ("Q1", "Educational Technology"),
    "computers and education": ("Q1", "Educational Technology"),
    "british journal of educational technology": ("Q1", "Educational Technology"),
    "educational technology research and development": ("Q1", "Educational Technology"),
    "internet and higher education": ("Q1", "Educational Technology"),
    "computers in human behavior": ("Q1", "Educational Technology"),
    "journal of computer assisted learning": ("Q1", "Educational Technology"),
    "instructional science": ("Q1", "Educational Technology"),
    "distance education": ("Q1", "Educational Technology"),
    "learning environments research": ("Q1", "Educational Technology"),
    "journal of educational psychology": ("Q1", "Educational Psychology"),
    "educational measurement issues and practice": ("Q1", "Education"),
    "journal of educational measurement": ("Q1", "Education"),
    "applied measurement in education": ("Q1", "Education"),
    "journal of research in science teaching": ("Q1", "Education"),
    "science education": ("Q1", "Education"),
    "cognition and instruction": ("Q1", "Education"),
    "reading research quarterly": ("Q1", "Education"),
    "journal of literacy research": ("Q1", "Education"),
    "early childhood education journal": ("Q1", "Education"),
    "early childhood research quarterly": ("Q1", "Education"),
    "child development": ("Q1", "Education"),
    "developmental psychology": ("Q1", "Education"),
    "journal of educational computing research": ("Q1", "Educational Technology"),
    "educational technology & society": ("Q1", "Educational Technology"),
    "journal of educational technology & society": ("Q1", "Educational Technology"),
    "journal of educational technology and society": ("Q1", "Educational Technology"),
    "behaviour & information technology": ("Q1", "Educational Technology"),
    "behavior and information technology": ("Q1", "Educational Technology"),
    "journal of research on technology in education": ("Q1", "Educational Technology"),
    "technology knowledge and learning": ("Q1", "Educational Technology"),
    "interactive learning environments": ("Q1", "Educational Technology"),
    "education and information technologies": ("Q1", "Educational Technology"),
    "australasian journal of educational technology": ("Q1", "Educational Technology"),
    "open learning": ("Q1", "Educational Technology"),
    "innovation in language learning and teaching": ("Q1", "Education"),
    "language learning & technology": ("Q1", "Education"),
    "language learning and technology": ("Q1", "Education"),
    "system": ("Q1", "Education"),
    "computers & education open": ("Q1", "Educational Technology"),
    # ── Q2 ──────────────────────────────────────────────────────
    "technology pedagogy and education": ("Q2", "Educational Technology"),
    "technology, pedagogy and education": ("Q2", "Educational Technology"),
    "journal of science education and technology": ("Q2", "Educational Technology"),
    "education sciences": ("Q2", "Education"),
    "frontiers in education": ("Q2", "Education"),
    "journal of educational research": ("Q2", "Education"),
    "educational studies": ("Q2", "Education"),
    "educational review": ("Q2", "Education"),
    "international journal of educational research": ("Q2", "Education"),
    "international journal of science education": ("Q2", "Education"),
    "school effectiveness and school improvement": ("Q2", "Education"),
    "assessment in education principles policy and practice": ("Q2", "Education"),
    "curriculum inquiry": ("Q2", "Education"),
    "journal of curriculum studies": ("Q2", "Education"),
    "teachers and teaching": ("Q2", "Education"),
    "teacher education and special education": ("Q2", "Education"),
    "journal of teacher education": ("Q2", "Education"),
    "active learning in higher education": ("Q2", "Education"),
    "studies in higher education": ("Q2", "Education"),
    "higher education research and development": ("Q2", "Education"),
    "journal of higher education": ("Q2", "Education"),
    "research in higher education": ("Q2", "Education"),
    "innovations in education and teaching international": ("Q2", "Education"),
    "journal of college student development": ("Q2", "Education"),
    "simulation & gaming": ("Q2", "Educational Technology"),
    "simulation and gaming": ("Q2", "Educational Technology"),
    "computers in the schools": ("Q2", "Educational Technology"),
    "journal of educational multimedia and hypermedia": ("Q2", "Educational Technology"),
    "online learning": ("Q2", "Educational Technology"),
    "international review of research in open and distributed learning": ("Q2", "Educational Technology"),
    "irreodl": ("Q2", "Educational Technology"),
    "asia pacific education researcher": ("Q2", "Education"),
    "asia-pacific education researcher": ("Q2", "Education"),
    "pedagogies an international journal": ("Q2", "Education"),
    "learning media and technology": ("Q2", "Educational Technology"),
    "e-learning and digital media": ("Q2", "Educational Technology"),
    "digital education review": ("Q2", "Educational Technology"),
    # ── Q3 ──────────────────────────────────────────────────────
    "international journal of educational technology in higher education": ("Q3", "Educational Technology"),
    "journal of e-learning and knowledge society": ("Q3", "Educational Technology"),
    "knowledge management & e-learning": ("Q3", "Educational Technology"),
    "smart learning environments": ("Q3", "Educational Technology"),
    "international journal of mobile and blended learning": ("Q3", "Educational Technology"),
    "international journal of emerging technologies in learning": ("Q3", "Educational Technology"),
    "journal of educational technology systems": ("Q3", "Educational Technology"),
    "turkish online journal of educational technology": ("Q3", "Educational Technology"),
    "educational technology international": ("Q3", "Educational Technology"),
    "international journal of computer-supported collaborative learning": ("Q3", "Educational Technology"),
    "international journal of computer supported collaborative learning": ("Q3", "Educational Technology"),
    "research and practice in technology enhanced learning": ("Q3", "Educational Technology"),
    "interdisciplinary journal of e-learning and learning objects": ("Q3", "Educational Technology"),
    "journal of information technology education research": ("Q3", "Educational Technology"),
    # ── Q4 ──────────────────────────────────────────────────────
    "international journal of education and development using ict": ("Q4", "Educational Technology"),
    "turkish online journal of distance education": ("Q4", "Educational Technology"),
    "merlot journal of online learning and teaching": ("Q4", "Educational Technology"),
}

# Normalize: lowercase + strip whitespace
SJR_LOOKUP: dict[str, tuple[str, str]] = {
    k.lower().strip(): v for k, v in _RAW.items()
}

# Common Semantic Scholar abbreviations → canonical key in SJR_LOOKUP.
_ABBREV: dict[str, str] = {
    "comput. educ.": "computers & education",
    "comput. hum. behav.": "computers in human behavior",
    "br. j. educ. technol.": "british journal of educational technology",
    "educ. technol. res. dev.": "educational technology research and development",
    "j. comput. assist. learn.": "journal of computer assisted learning",
    "internet high. educ.": "internet and higher education",
    "learn. instr.": "learning and instruction",
    "instr. sci.": "instructional science",
    "j. educ. psychol.": "journal of educational psychology",
    "contemp. educ. psychol.": "contemporary educational psychology",
    "educ. psychol. rev.": "educational psychology review",
    "educ. res. rev.": "educational research review",
    "rev. educ. res.": "review of educational research",
    "teach. teach. educ.": "teaching and teacher education",
    "behav. inf. technol.": "behaviour & information technology",
    "j. res. technol. educ.": "journal of research on technology in education",
    "educ. inf. technol.": "education and information technologies",
    "j. sci. educ. technol.": "journal of science education and technology",
    "comput. educ. open": "computers & education open",
}

_QUARTILE_COLOR = {"Q1": "#1a7f37", "Q2": "#0969da", "Q3": "#9a6700", "Q4": "#cf222e"}


def _normalize(journal_name: str) -> str:
    """Strip volume/page/subtitle noise that Semantic Scholar appends to venue names."""
    n = journal_name.lower().strip()
    # Cut off ", Vol.X" / ", pp.Y" suffixes added by format_paper(), and " : subtitle".
    for sep in (", vol", ", pp", " : ", ": ", " - vol"):
        idx = n.find(sep)
        if idx != -1:
            n = n[:idx]
    return n.strip().rstrip(",.").strip()


# Normalize abbreviation keys the same way lookups are normalized (drops trailing ".").
_ABBREV_LOOKUP: dict[str, str] = {_normalize(k): v for k, v in _ABBREV.items()}


def get_quartile(journal_name: str) -> tuple[str, str] | None:
    """Return (quartile, subject) or None if not found."""
    if not journal_name:
        return None
    key = _normalize(journal_name)
    key = _ABBREV_LOOKUP.get(key, key)
    return SJR_LOOKUP.get(key)


def quartile_badge(journal_name: str) -> str:
    """Return an HTML badge string for the quartile, or empty string."""
    result = get_quartile(journal_name)
    if not result:
        return ""
    q, _ = result
    color = _QUARTILE_COLOR.get(q, "#666")
    return (
        f'<span style="display:inline-block;padding:1px 6px;border-radius:3px;'
        f'background:{color};color:#fff;font-size:0.70rem;font-weight:700;'
        f'margin-left:4px;vertical-align:middle">{q}</span>'
    )
