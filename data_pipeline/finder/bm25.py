import faiss
import pickle
import numpy as np
import ollama
from pathlib import Path
from rank_bm25 import BM25Okapi  # pip install rank_bm25

# 설정
BASE_DIR = Path(r"C:\ai\source\soloproject")
SAVE_DIR = BASE_DIR / "faiss_db"
MODEL_NAME = "qwen3-embedding:0.6b"

def get_query_embedding(text):
    response = ollama.embed(model=MODEL_NAME, input=f"query: {text}")
    return response['embeddings'][0]

def hybrid_search(query_text, top_k=5):
    # 1. 데이터 로드
    index = faiss.read_index(str(SAVE_DIR / "math_index.faiss"))
    with open(SAVE_DIR / "math_content.pkl", "rb") as f:
        all_content = pickle.load(f)

    # 2. BM25 모델 생성 (검색할 때마다 생성하면 느리니 나중엔 따로 저장하세요)
    # 텍스트를 공백 기준으로 나눠 토큰화합니다.
    tokenized_corpus = [doc['text'].split() for doc in all_content]
    bm25 = BM25Okapi(tokenized_corpus)

    # --- [검색 과정] ---
    # (1) FAISS 벡터 검색
    query_emb = np.array([get_query_embedding(query_text)]).astype('float32')
    faiss_distances, faiss_indices = index.search(query_emb, top_k * 2) # 좀 넉넉하게 뽑음

    # (2) BM25 점수 계산
    tokenized_query = query_text.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # (3) 결과 통합 및 정렬 (단순 합산 방식)
    results = []
    for i, idx in enumerate(faiss_indices[0]):
        if idx == -1: continue
        
        # FAISS 점수 (낮을수록 좋음 -> 정규화 필요)
        f_score = 1 / (1 + faiss_distances[0][i]) 
        # BM25 점수 (높을수록 좋음)
        b_score = bm25_scores[idx] / (max(bm25_scores) + 1e-9)
        
        # 가중치 부여 (의미 검색 7 : 키워드 검색 3)
        final_score = (f_score * 0.6) + (b_score * 0.4)
        
        results.append((final_score, all_content[idx]))

    # 최종 점수로 정렬
    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n🚀 하이브리드 검색 결과 (검색어: {query_text})")
    for i, (score, item) in enumerate(results[:top_k]):
        print(f"[{i+1}] 점수: {score:.4f} | ID: {item['id']}")
        print(f"📝 내용: {item['text'][:100]}...")
        print("-" * 50)

if __name__ == "__main__":
    # 키워드가 섞인 검색어로 테스트해 보세요!
    hybrid_search("2024년 6월 24번 복소수 문제")