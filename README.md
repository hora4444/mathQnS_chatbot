# LLM & RAG 기반 수학 모의고사 챗봇

PDF 기반 수학 모의고사 데이터를 파싱하고,
RAG 검색과 LLM 응답을 활용하여
문제 및 해설을 제공하는 학습 지원 챗봇입니다.

## Tech Stack
- Python
- Django
- OpenAI API
- RAG
- Faiss
- BM25
- OCR
- Oracle Cloud

## Features
- PDF 문제/해설 자동 파싱
- OCR 기반 텍스트 추출
- 벡터 검색 + BM25 하이브리드 검색
- 자연어 기반 문제 질의응답
- Django 웹 서비스 배포(140.245.69.100:8000)

## System Architecture
PDF → OCR → JSONL → Embedding → Faiss/BM25 → RAG → LLM Response
