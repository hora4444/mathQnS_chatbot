try:
    from langchain_core.runnables import RunnableParallel, RunnableLambda
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import Chroma
    print("✅ 모든 모듈을 성공적으로 불러왔습니다!")
except ImportError as e:
    print(f"❌ 임포트 실패: {e}")