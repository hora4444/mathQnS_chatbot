# LLM & RAG 기반 수학 모의고사 챗봇
PDF 기반 수학 모의고사 데이터를 파싱하고,
RAG 기반 검색과 LLM 응답 생성을 활용하여
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
- Django 웹 서비스 배포

## Demo
http://140.245.69.100:8000

## System Architecture
PDF → OCR → JSONL → Embedding → Faiss/BM25 → RAG → LLM Response

## Deployment Note
Oracle Cloud Free Tier 환경에서 Django 기반 서비스를 배포했습니다.

현재 웹 서비스 접속은 정상 동작하지만,
LLM 응답 생성 과정에서 서버 성능 제한으로 인해 응답 지연이 발생하거나 응답을 하지 않을 수 있습니다.

향후:
- 서버 성능 개선
- 비동기 처리 적용
- 경량화 모델 및 캐싱 구조 도입
등을 통해 응답 속도를 개선할 예정입니다.
