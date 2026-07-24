"""논문 프로젝트(워크스페이스) 관리 + 분석 결과 영구 저장.

프로젝트마다 참고문헌 PDF, RAG 인덱스, 분석 결과를 따로 보관한다.
내 문체 프로필(my_papers, style_profile.json)은 사람에 속하므로 전역 유지.

저장 구조:
  /data/projects.json              프로젝트 목록 [{"id": "1", "name": "ai활용"}, ...]
  /data/projects/<id>/pdfs/        참고문헌 PDF
  /data/projects/<id>/reference_db RAG 인덱스
  /data/projects/<id>/analyses.json 통계 분석 결과 히스토리
"""

import json
import shutil
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd

DEFAULT_PROJECTS = [
    {"id": "1", "name": "프로젝트 1"},
    {"id": "2", "name": "프로젝트 2"},
    {"id": "3", "name": "프로젝트 3"},
]
MAX_PROJECTS = 6
MAX_SAVED_ANALYSES = 30


# ── 프로젝트 목록 ─────────────────────────────────────────────

def _projects_file(data_dir):
    return Path(data_dir) / "projects.json"


def load_projects(data_dir):
    f = _projects_file(data_dir)
    if f.exists():
        try:
            projects = json.loads(f.read_text(encoding="utf-8"))
            if projects:
                return projects
        except Exception:
            pass
    return [dict(p) for p in DEFAULT_PROJECTS]


def save_projects(data_dir, projects):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    _projects_file(data_dir).write_text(
        json.dumps(projects, ensure_ascii=False, indent=2), encoding="utf-8")


def project_dir(data_dir, pid):
    d = Path(data_dir) / "projects" / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    return d


def migrate_legacy(data_dir):
    """프로젝트 도입 전의 전역 pdfs/reference_db를 프로젝트 1로 1회 이동."""
    root = Path(data_dir)
    marker = root / "projects" / ".migrated"
    if marker.exists():
        return
    p1 = project_dir(data_dir, "1")
    for name in ["pdfs", "reference_db"]:
        src, dst = root / name, p1 / name
        if src.exists() and any(src.iterdir()) and not dst.exists():
            shutil.move(str(src), str(dst))
    marker.write_text("done", encoding="utf-8")


# ── 분석 결과 히스토리 ────────────────────────────────────────

def _analyses_file(proj_dir):
    return Path(proj_dir) / "analyses.json"


def load_analyses(proj_dir):
    f = _analyses_file(proj_dir)
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_analysis(proj_dir, analysis_name, result, interp=None, note=""):
    """분석 1건을 히스토리 맨 앞에 저장. 최근 MAX_SAVED_ANALYSES건 유지."""
    records = load_analyses(proj_dir)
    records.insert(0, {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "analysis": analysis_name,
        "note": note,
        "summary": result["summary"],
        "interp": interp,
        "tables": {k: v.to_json(orient="split", force_ascii=False)
                   for k, v in result["tables"].items()},
    })
    del records[MAX_SAVED_ANALYSES:]
    _analyses_file(proj_dir).write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8")


def update_analysis_interp(proj_dir, index, interp):
    """저장된 분석 기록에 논문용 해석을 나중에 채워넣는다."""
    records = load_analyses(proj_dir)
    if 0 <= index < len(records):
        records[index]["interp"] = interp
        _analyses_file(proj_dir).write_text(
            json.dumps(records, ensure_ascii=False), encoding="utf-8")


def delete_analysis(proj_dir, index):
    records = load_analyses(proj_dir)
    if 0 <= index < len(records):
        records.pop(index)
        _analyses_file(proj_dir).write_text(
            json.dumps(records, ensure_ascii=False), encoding="utf-8")


def tables_from_record(record):
    """저장된 레코드의 표들을 DataFrame으로 복원."""
    return {k: pd.read_json(StringIO(v), orient="split")
            for k, v in record["tables"].items()}
