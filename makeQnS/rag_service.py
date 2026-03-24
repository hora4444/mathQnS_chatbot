import os
import sys

# PyTorch나 외부 라이브러리가 GPU를 건드리지 못하게 차단
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["USE_TORCH"] = "0"# Torch 사용 시도 자체를 억제

import pickle
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings, ChatOllama

from django.conf import settings

FAISS_INSWX_PATH=os.path.join(settings.BASE_DIR, "faiss_db/math_index.faiss")
CONTENT_PKL_PATH=os.path.join(settings.BASE_DIR, "faiss_db/math_content.pkl")


# 1. 초기 로드 및 데이터 준비
EMBEDDINGS = OllamaEmbeddings(model="nomic-embed-text")
_rag_chain = None

# LLM설정
llm = ChatOllama(
    # model="qwen3.5:4b", # 9b사용시 djago shell에서 동작하지 않아 다운그레이드 함
    model="qwen3:0.6b", # oracle cloud용(VM.Standard.A1.Flex는 서버생성이 막혀있어 VM.Standard.E2.1.Micro사용)
    temperature=0,  # 수학 문제의 정답 일관성을 위해 0으로 설정
)

template = """당신은 수학 교육 전문가입니다. 
아래 제공된 [Context]의 문제 정보를 참고하여 사용자의 [Question]에 대해 논리적으로 답변하세요.

[규칙]
1. 모든 수식은 반드시 LaTeX 형식($...$ 또는 $$...$$)으로 작성하세요.
2. 답변은 반드시 한국어로 작성하세요.
3. 사용자가 요청한 문제의 개수를 반드시 엄격하게 지켜서 생성하세요. (예: 3개 요청 시 반드시 3개 생성)
4. 각 문제는 '문제 1.', '문제 2.'와 같이 명확한 번호를 매겨 구분하세요.
5. [Context]에 질문과 관련된 문제가 있다면 그 정보를 바탕으로 설명하세요.
6. 만약 [Context]만으로 답을 알 수 없다면, 아는 범위 내에서 답변하되 근거가 부족함을 알리세요.
7. 모든 핵심 수식은 텍스트와 분리하여 독립된 줄에 작성하고, 반드시 LaTeX의 \\begin{{aligned}} ... \\end{{aligned}} 환경을 사용하세요.
   - 단순한 대입이나 결과 도출 과정도 생략하지 말고 단계별로 작성합니다.
   - 대괄호([ ])나 단순 나열 대신, 수학적으로 올바른 LaTeX 문법만 사용하세요.
8. 수식 전개 시 가로로 길어지는 것을 방지하기 위해 \\begin{{aligned}} 환경을 적극 활용하세요.
   - 등호(=)가 반복되는 계산 과정
   - 여러 개의 조건식이나 방정식을 동시에 나열할 때
   - 계산의 '단계'가 바뀔 때
   반드시 각 식 끝에 줄바꿈(\\\\)을 넣어 수직으로 정렬하세요.
   
   (범용 예시:
   \\begin{{aligned}}
   (조건 1) &= (식 1) \\\\
   (조건 2) &= (식 2) \\\\
   \\therefore (결과) &= (최종 식)
   \\end{{aligned}}
   형식으로 작성하면 가독성이 높습니다.)
반드시 답변 마지막에 아래 형식으로 [RAG 분석 리포트]를 작성해주세요:

[RAG 분석 리포트]
- 참고 문제 ID: (DB에서 가져온 원본 문제 번호나 제목)
- 주요 변형 사항: (예: 숫자 변경, 조건 추가, 질문 방향 전환 등)
- 유사도 수준: (원본과 얼마나 유사한지 %로 표기)

[Context]
{context}

[Question]
{question}

답변:"""

prompt = ChatPromptTemplate.from_template(template)

def load_custom_faiss():
    # 저장된 pkl에서 텍스트와 메타데이터 복원
    with open(CONTENT_PKL_PATH, "rb") as f:
        all_content = pickle.load(f)
    
    # Document 객체 리스트 생성
    docs = [
        Document(
            page_content=item['text'], 
            metadata={**item['metadata'], "id": item['id']}
        ) for item in all_content
    ]
    
    # LangChain용 FAISS 객체로 변환 (이미 생성된 index 파일 활용)
    # ※ 주의: FAISS.load_local은 전용 포맷이 필요하므로, 
    # 처음엔 from_documents로 메모리에 올리는 것이 장고 서버 구동 시 더 안정적입니다.
    vectorstore = FAISS.from_documents(docs, EMBEDDINGS)
    return vectorstore, docs

# 3. 앙상블 로직 (Runnable용 함수)
def merge_documents(data: dict):
    """FAISS와 BM25 결과를 합치고 중복을 제거함"""
    faiss_docs = data["faiss"]
    bm25_docs = data["bm25"]
    
    combined = faiss_docs + bm25_docs
    # ID 기반 중복 제거
    unique_docs = []
    seen_ids = set()
    for doc in combined:
        doc_id = doc.metadata.get("id")
        if doc_id not in seen_ids:
            unique_docs.append(doc)
            seen_ids.add(doc_id)
    return unique_docs[:5] # 최종 TOP 5 반환

def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs):
        # 문서의 ID와 내용을 구분하여 LLM이 참조하기 쉽게 구성
        content = f"[문제 {i+1}] (ID: {doc.metadata.get('id')})\n{doc.page_content}"
        formatted.append(content)
    return "\n\n".join(formatted)

def get_rag_chain():
    """필요한 시점에 로딩하여 RAG 체인을 반환하는 함수"""
    global _rag_chain
    
    if _rag_chain is None:
        print("\n[시스템] RAG 리소스 로딩을 시작합니다. (최초 1회 실행)")
        
        # 1. 데이터 로드
        vectorstore, all_docs = load_custom_faiss()
        
        # 2. 리트리버 설정
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 3
        
        # 3. 앙상블 리트리버 구성
        ensemble_retriever = RunnableParallel(
            faiss=faiss_retriever,
            bm25=bm25_retriever
        ) | RunnableLambda(merge_documents)
        
        # 4. 최종 체인 구성
        _rag_chain = (
            {
                "context": ensemble_retriever | format_docs, 
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        print("[시스템] RAG 리소스 로딩 완료!\n")
        
    return _rag_chain

if __name__ == "__main__":
    test_query = "복소수 z에 대하여 z^2이 실수가 되도록 하는 값을 구하는 문제 찾아줘"
    print(f"질문: {test_query}\n")
    print("--- 답변 생성 중 ---")
    
    # 이제 전역 변수가 아닌 함수를 호출하여 체인을 가져옵니다.
    chain = get_rag_chain()
    for chunk in chain.stream(test_query):
        print(chunk, end="", flush=True)