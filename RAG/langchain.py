import os
import sys

# PyTorch나 외부 라이브러리가 GPU를 건드리지 못하게 차단
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["USE_TORCH"] = "0"

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

# 1. 초기 로드 및 데이터 준비
EMBEDDINGS = OllamaEmbeddings(model="nomic-embed-text")
FAISS_INDEX_PATH = "C:/ai/source/soloproject/faiss_db/math_index.faiss"
CONTENT_PKL_PATH = "C:/ai/source/soloproject/faiss_db/math_content.pkl"

# LLM설정
llm = ChatOllama(
    model="qwen3.5:9b",
    temperature=0,  # 수학 문제의 정답 일관성을 위해 0으로 설정
)

template = """당신은 수학 교육 전문가입니다. 
아래 제공된 [Context]의 문제 정보를 참고하여 사용자의 [Question]에 대해 논리적으로 답변하세요.

[규칙]
1. 모든 수식은 반드시 LaTeX 형식($...$ 또는 $$...$$)으로 작성하세요.
2. 답변은 반드시 한국어로 작성하세요.
3. [Context]에 질문과 관련된 문제가 있다면 그 정보를 바탕으로 설명하세요.
4. 만약 [Context]만으로 답을 알 수 없다면, 아는 범위 내에서 답변하되 근거가 부족함을 알리세요.

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

vectorstore, all_docs = load_custom_faiss()

# 2. 개별 리트리버 설정
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
bm25_retriever = BM25Retriever.from_documents(all_docs)
bm25_retriever.k = 3

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

# 4. LCEL 체인 구성 (앙상블 리트리버)
ensemble_retriever = RunnableParallel(
    faiss=faiss_retriever,
    bm25=bm25_retriever
) | RunnableLambda(merge_documents)

def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs):
        # 문서의 ID와 내용을 구분하여 LLM이 참조하기 쉽게 구성
        content = f"[문제 {i+1}] (ID: {doc.metadata.get('id')})\n{doc.page_content}"
        formatted.append(content)
    return "\n\n".join(formatted)

# 최종 RAG 체인 구성 (LCEL)
rag_chain = (
    {
        "context": ensemble_retriever | format_docs, 
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    # 실제 DB에 있을 법한 질문으로 테스트
    test_query = "복소수 z에 대하여 z^2이 실수가 되도록 하는 값을 구하는 문제 찾아줘"
    print(f"질문: {test_query}\n")
    print("--- 답변 생성 중 ---")
    
    # 9B 모델의 응답을 실시간으로 확인하기 위해 스트리밍 사용
    for chunk in rag_chain.stream(test_query):
        print(chunk, end="", flush=True)