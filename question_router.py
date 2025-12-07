import re
from typing import Dict, List, Tuple
from enum import Enum


class QuestionType(Enum):
    READING = "reading"
    FACTUAL = "factual"
    MATH = "math"
    SAFETY = "safety"


class ModelChoice(Enum):
    SMALL = "small"
    LARGE = "large"
    NONE = None


class QuestionRouter:
    READING_PATTERNS = [
        r"Đoạn thông tin:",
        r"Văn bản:",
        r"Bài viết:",
        r"Đoạn văn sau:",
        r"\[1\] Tiêu đề:"
    ]
    
    MATH_PATTERNS = [
        r"\$",
        r"(sin|cos|tan|cot|log|ln|frac|sqrt)",
        r"phương trình",
        r"tính toán",
        r"giải bài toán",
        r"đạo hàm",
        r"tích phân",
        r"xác suất"
    ]
    
    SAFETY_KEYWORDS = ["tránh", "lách", "trốn", "vi phạm", "né", "gian lận"]
    SAFETY_ANSWERS = [r"không thể chia sẻ", r"tuân thủ", r"pháp luật"]

    def __init__(self):
        self.reading_re = re.compile("|".join(self.READING_PATTERNS), re.IGNORECASE)
        self.math_re = re.compile("|".join(self.MATH_PATTERNS), re.IGNORECASE)

    def classify(self, question: str, choices: List[str]) -> Tuple[QuestionType, ModelChoice, Dict]:
        q_lower = question.lower()
        
        if self.reading_re.search(question) or len(question) > 2000:
            return QuestionType.READING, ModelChoice.SMALL, {"use_rag": False}
        
        if any(kw in q_lower for kw in self.SAFETY_KEYWORDS):
            safe_idx = self._find_safe_choice(choices)
            if safe_idx is not None:
                return QuestionType.SAFETY, ModelChoice.NONE, {"use_rag": False, "safe_idx": safe_idx}
            return QuestionType.SAFETY, ModelChoice.SMALL, {"use_rag": False}
        
        if self.math_re.search(question):
            return QuestionType.MATH, ModelChoice.LARGE, {"use_rag": False}
        
        return QuestionType.FACTUAL, ModelChoice.SMALL, {"use_rag": True}

    def _find_safe_choice(self, choices: List[str]) -> int:
        for idx, choice in enumerate(choices):
            for pattern in self.SAFETY_ANSWERS:
                if re.search(pattern, choice.lower()):
                    return idx
        return None

    def build_prompt(self, qtype: QuestionType, question: str, 
                     choices: List[str], context: str = None) -> List[Dict]:
        choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        
        if qtype == QuestionType.READING:
            return [
                {"role": "system", "content": """Bạn là trợ lý thông minh giúp trả lời câu hỏi trắc nghiệm.
Với mỗi câu hỏi, hãy:
1. Đọc kỹ đoạn văn được cung cấp
2. Phân tích từng đáp án xem có phù hợp không
3. Giải thích lý do chọn đáp án
4. Cuối cùng ghi rõ: "Đáp án: X" (X là A, B, C hoặc D)"""},
                {"role": "user", "content": f"{question}\n\nCác lựa chọn:\n{choices_str}\n\nHãy phân tích và chọn đáp án đúng nhất."}
            ]
        
        if qtype == QuestionType.MATH:
            return [
                {"role": "system", "content": """Bạn là chuyên gia giải toán và logic.
Với mỗi bài toán, hãy:
1. Đọc kỹ đề bài, xác định các dữ kiện
2. Viết ra các bước giải chi tiết
3. Tính toán cẩn thận từng bước
4. Kiểm tra lại kết quả với từng đáp án
5. Cuối cùng ghi rõ: "Đáp án: X" (X là chữ cái tương ứng)"""},
                {"role": "user", "content": f"Bài toán:\n{question}\n\nCác đáp án:\n{choices_str}\n\nHãy giải từng bước và chọn đáp án đúng."}
            ]
        
        if qtype == QuestionType.SAFETY:
            return [
                {"role": "system", "content": """Bạn là trợ lý tuân thủ pháp luật và đạo đức.
Ưu tiên chọn đáp án thể hiện sự từ chối hoặc tuân thủ quy định.
Cuối cùng ghi rõ: "Đáp án: X" """},
                {"role": "user", "content": f"Câu hỏi:\n{question}\n\nCác lựa chọn:\n{choices_str}\n\nChọn đáp án an toàn và hợp pháp nhất."}
            ]
        
        ctx = f"Thông tin tham khảo:\n{context}\n\n" if context else ""
        return [
            {"role": "system", "content": """Bạn là chuyên gia về kiến thức tổng hợp, lịch sử, địa lý, văn hóa Việt Nam.
Với mỗi câu hỏi, hãy:
1. Phân tích câu hỏi
2. Suy luận dựa trên kiến thức
3. Đánh giá từng đáp án
4. Cuối cùng ghi rõ: "Đáp án: X" (X là A, B, C hoặc D)"""},
            {"role": "user", "content": f"{ctx}Câu hỏi:\n{question}\n\nCác đáp án:\n{choices_str}\n\nHãy phân tích và chọn đáp án đúng nhất."}
        ]


if __name__ == "__main__":
    router = QuestionRouter()
    
    tests = [
        ("Doan thong tin:\n[1] Tieu de: Test\nNoi dung...", ["A", "B", "C", "D"]),
        ("Cho $ sin x = 0 $", ["A", "B"]),
    ]
    
    for q, c in tests:
        qtype, model, meta = router.classify(q, c)
        print(f"{qtype.value} | {model.value if model.value else 'NONE'}")
