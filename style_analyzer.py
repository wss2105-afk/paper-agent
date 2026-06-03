import json
import os
from pathlib import Path

import pdfplumber


def extract_text_from_pdf(pdf_path, max_pages=20):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:max_pages]:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def load_my_papers(my_papers_dir):
    """내 논문 폴더에서 텍스트 추출"""
    papers = []
    for pdf_path in Path(my_papers_dir).glob("*.pdf"):
        try:
            text = extract_text_from_pdf(pdf_path)
            if text.strip():
                papers.append({"filename": pdf_path.name, "text": text})
        except Exception:
            pass
    return papers


def build_style_prompt(papers):
    """Claude에게 보낼 스타일 분석 프롬프트 생성"""
    combined = "\n\n===논문 구분===\n\n".join(
        f"[논문: {p['filename']}]\n{p['text'][:3000]}"
        for p in papers
    )
    return f"""다음은 한 연구자가 직접 작성한 논문들입니다. 이 논문들을 분석하여 이 연구자만의 학술적 글쓰기 스타일 프로필을 작성해주세요.

[논문 텍스트]
{combined}

아래 항목별로 구체적으로 분석해주세요:

1. **문장 스타일**: 문장 길이, 구조, 수동태/능동태 선호도, 접속사 사용 패턴
2. **어휘/표현 특징**: 자주 사용하는 학술 표현, 선호하는 단어, 특징적인 어구
3. **학문적 관점**: 주로 다루는 이론적 프레임, 인용하는 학자/이론, 교육공학적 접근 방식
4. **논증 방식**: 주장을 전개하는 방식, 근거 제시 패턴, 반론 처리 방식
5. **단락 구성**: 도입-전개-마무리 패턴, 단락당 평균 문장 수
6. **인용 통합 방식**: 직접인용 vs 간접인용 선호, 인용 앞뒤 문장 패턴
7. **전체 문체 요약**: 위 분석을 바탕으로 이 연구자의 글쓰기를 2~3문장으로 요약

이 프로필은 향후 이 연구자 스타일로 글을 생성할 때 사용됩니다."""


def save_style_profile(profile_text, profile_path):
    data = {"profile": profile_text}
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_style_profile(profile_path):
    if not Path(profile_path).exists():
        return None
    with open(profile_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("profile", "")


def build_style_instruction(profile):
    """글쓰기 프롬프트에 삽입할 스타일 지시문 생성"""
    return f"""
[작성자 스타일 프로필 - 반드시 아래 스타일을 따를 것]
{profile}

위 스타일 프로필을 철저히 반영하여 글을 작성하세요.
작성자 고유의 문체, 어휘, 논증 방식, 학문적 관점을 최대한 살려주세요.
"""
