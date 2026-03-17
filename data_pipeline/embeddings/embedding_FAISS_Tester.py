import faiss
import pickle
import numpy as np
import ollama
from pathlib import Path

# 설정
BASE_DIR = Path(r"C:\ai\source\soloproject")
SAVE_DIR = BASE_DIR / "faiss_db"
MODEL_NAME = "qwen3-embedding:0.6b"

def get_query_embedding(text):
    # 검색 시에는 task_type을 'query'로 설정하는 것이 좋습니다.
    response = ollama.embed(model=MODEL_NAME, input=f"query: {text}")
    return response['embeddings'][0]

def test_search(query_text, top_k=3):
    # 1. 파일 로드
    print("📂 인덱스 로드 중...")
    index = faiss.read_index(str(SAVE_DIR / "math_index.faiss"))
    with open(SAVE_DIR / "math_content.pkl", "rb") as f:
        all_content = pickle.load(f)

    # 2. 질문 임베딩
    print(f"🔎 검색어: {query_text}")
    query_emb = np.array([get_query_embedding(query_text)]).astype('float32')

    # 3. 검색 실시 (D: 거리, I: 인덱스 번호)
    distances, indices = index.search(query_emb, top_k)

    # 4. 결과 출력
    print("\n🎯 가장 유사한 문제 Top 3:")
    for i, idx in enumerate(indices[0]):
        if idx < len(all_content):
            item = all_content[idx]
            print(f"[{i+1}] 유사도 거리: {distances[0][i]:.4f}")
            print(f"🆔 ID: {item['id']}")
            print(f"📝 내용: {item['text'][:100]}...")
            print("-" * 50)

if __name__ == "__main__":
    # 테스트하고 싶은 검색어를 넣어보세요
    test_search("복소수 z의 값을 구하는 문제")