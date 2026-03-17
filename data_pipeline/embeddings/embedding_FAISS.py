import json
import ollama
import faiss
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import os
import time

# --- 설정 구간 ---
BASE_DIR = Path(r"C:\ai\source\soloproject")
MODEL_NAME = "qwen3-embedding:0.6b"
# 저장 경로 (폴더가 없으면 생성)
SAVE_DIR = BASE_DIR / "faiss_db"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 1024차원 (Qwen-0.6b 임베딩 차원에 맞춤. 실행 후 에러나면 출력값 확인 후 수정)
DIMENSION = 1024 

def get_embedding(text, task_type="passage"):
    prefixed_text = f"{task_type}: {text}"
    response = ollama.embed(model=MODEL_NAME, input=prefixed_text)
    return response['embeddings'][0]

def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def build_text_for_embedding(item):
    q_text = item.get("question_text", "").strip()
    s_text = item.get("solution_text", "").strip()
    if "*주요 수정 사항*" in s_text:
        s_text = s_text.split("*주요 수정 사항*")[0].strip()
    return f"문제: {q_text}\n\n해설: {s_text}".strip()

def run_faiss_indexing():
    final_dir = BASE_DIR / "output" / "final_integrated"
    jsonl_files = sorted(list(final_dir.glob("*.jsonl")))
    
    # 1. FAISS 인덱스 초기화 (L2 거리 기반 검색)
    index = faiss.IndexFlatL2(DIMENSION)
    
    # 2. 메타데이터와 원문을 담을 리스트 (BM25 및 결과 노출용)
    all_content = []
    
    print(f"📂 총 {len(jsonl_files)}개 파일을 처리합니다.")

    for jsonl_path in jsonl_files:
        print(f"\n📦 작업 중: {jsonl_path.name}")
        items = list(load_jsonl(jsonl_path))
        
        # 이번 파일의 벡터들을 담을 리스트
        file_embeddings = []
        
        for item in tqdm(items, desc="임베딩 생성"):
            text = build_text_for_embedding(item)
            if not text: continue
            
            try:
                emb = get_embedding(text)
                file_embeddings.append(emb)
                
                # 원문과 메타데이터 저장 (나중에 검색 결과 확인용)
                all_content.append({
                    "id": f"{jsonl_path.stem}_{item.get('id')}",
                    "text": text,
                    "metadata": item.get("metadata", {}),
                    "file_source": jsonl_path.name
                })
            except Exception as e:
                print(f"❌ 실패: {e}")

        # 인덱스에 벡터 추가 (Numpy 배열로 변환 필요)
        if file_embeddings:
            embeddings_np = np.array(file_embeddings).astype('float32')
            index.add(embeddings_np)
            print(f"✅ {jsonl_path.name} 인덱스 추가 완료. (현재 총 {index.ntotal}개)")

    # 💾 최종 파일 저장
    faiss.write_index(index, str(SAVE_DIR / "math_index.faiss"))
    with open(SAVE_DIR / "math_content.pkl", "wb") as f:
        pickle.dump(all_content, f)

    print(f"\n🚀 모든 작업 완료! 총 {index.ntotal}개의 벡터가 저장되었습니다.")
    print(f"📍 저장 위치: {SAVE_DIR}")

if __name__ == "__main__":
    run_faiss_indexing()