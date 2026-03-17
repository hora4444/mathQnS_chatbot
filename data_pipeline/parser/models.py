from django.db import models

# Create your models here.
from django.db import models

class MockExamQuestion(models.Model):
    # 1. 문제 식별 정보
    question_id = models.CharField(max_length=100, unique=True, help_text="예: g1_2020_3_common_q1")
    grade = models.IntegerField(choices=[(1, '고1'), (2, '고2'), (3, '고3')])
    year = models.IntegerField()
    month = models.IntegerField()
    subject = models.CharField(max_length=50, null=True, blank=True, help_text="고3 선택과목(기하 등)")

    # 2. 텍스트 데이터
    raw_text = models.TextField(help_text="JSONL에서 가져온 원본(깨진) 텍스트")
    cleaned_latex = models.TextField(null=True, blank=True, help_text="Nougat 모델로 수선된 LaTeX 텍스트")
    
    # 3. 추가 정보
    score = models.IntegerField(default=0, help_text="배점 (2, 3, 4점)")
    answer = models.CharField(max_length=10, null=True, blank=True)

    def __str__(self):
        return f"[{self.question_id}] {self.year}년 {self.month}월"

class QuestionImage(models.Model):
    # 어떤 문제에 속한 이미지인지 연결 (1:N 관계)
    question = models.ForeignKey(MockExamQuestion, on_delete=models.CASCADE, related_name='images')
    image_path = models.CharField(max_length=500, help_text="C:\ai\source\... 절대 경로 저장")
    
    def __str__(self):
        return f"Image for {self.question.question_id}"