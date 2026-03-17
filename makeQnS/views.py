from django.shortcuts import render

# Create your views here.
from django.http import StreamingHttpResponse
from .rag_service import EMBEDDINGS, rag_chain
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

last_chat = {
        "question": "",
        "answer": "",
        "embedding": None
    }

def get_similarity(query1, query2_emb):
        """현재 질문과 이전 질문의 임베딩 유사도를 계산"""
        if query2_emb is None:
            return 0
        curr_emb = np.array(EMBEDDINGS.embed_query(query1)).reshape(1, -1)
        prev_emb = np.array(query2_emb).reshape(1, -1)
        return cosine_similarity(curr_emb, prev_emb)[0][0]

def chat_stream(request):
    """
    사용자의 질문을 받아 실시간으로 답변을 스트리밍하는 View
    """
    user_question = request.GET.get('q', '')

    def event_stream():
        global last_chat
        full_answer = ""
        # 1. 유사도 검사 (0.7 이상이면 맥락 유지)
        similarity = get_similarity(user_question, last_chat["embedding"])
        
        if similarity > 0.7:
            # 맥락이 이어질 경우 질문을 보강함
            contextual_question = f"이전 질문: {last_chat['question']}\n이전 답변: {last_chat['answer']}\n\n위 내용을 참고해서 다음 질문에 답해줘: {user_question}"
        else:
            # 새로운 주제라면 바로 질문함
            contextual_question = user_question

        for chunk in rag_chain.stream(contextual_question):
            # chunk가 객체인 경우 content 속성만 추출, 문자열인 경우 그대로 사용
            if isinstance(chunk, str):
                content = chunk
            else:
                # LangChain 버전에 따라 다를 수 있으나 보통 .content에 실제 텍스트가 있습니다.
                content = getattr(chunk, 'content', str(chunk))
        
            full_answer += content
            data = json.dumps({"content": content})
            yield f"data: {data}\n\n"
        
        last_chat["question"] = user_question
        last_chat["answer"] = full_answer
        last_chat["embedding"] = EMBEDDINGS.embed_query(user_question)

    # content_type을 'text/event-stream'으로 설정하는 것이 핵심입니다.
    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')

def index(request):
    return render(request, 'makeQnS/index.html')
