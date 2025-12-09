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
    
    SAFETY_KEYWORDS = [
        "tránh", "lách", "trốn", "vi phạm", "né", "gian lận",
        "làm giả", "trái phép", "bất hợp pháp", "che giấu",
        "chống lại", "phản đối", "chống phá", "cách nào để tránh",
        "làm sao để tránh", "không bị phát hiện"
    ]
    
    SAFETY_ANSWER_PATTERNS = [
        r"tôi không thể",
        r"Tôi không thể",
        r"không thể chia sẻ",
        r"không thể cung cấp",
        r"không thể trả lời",
        r"không thể hỗ trợ",
        r"từ chối",
    ]

    def __init__(self):
        self.reading_re = re.compile("|".join(self.READING_PATTERNS), re.IGNORECASE)
        self.math_re = re.compile("|".join(self.MATH_PATTERNS), re.IGNORECASE)
        self.safety_answer_re = re.compile("|".join(self.SAFETY_ANSWER_PATTERNS), re.IGNORECASE)

    def classify(self, question: str, choices: List[str]) -> Tuple[QuestionType, ModelChoice, Dict]:
        q_lower = question.lower()
        num_choices = len(choices)
        
        safe_idx = self._find_safe_choice(choices)
        has_safety_answer = safe_idx is not None
        
        has_latex = bool(re.search(r'\$.*\$|\\frac|\\sqrt|\\sum|\\int', question))
        is_stem = has_latex and num_choices >= 8
        
        if is_stem:
            return QuestionType.MATH, ModelChoice.LARGE, {"use_rag": False, "is_stem": True, "safe_idx": safe_idx}
        
        if self.reading_re.search(question) or len(question) > 2000:
            model = ModelChoice.LARGE if len(question) > 5000 else ModelChoice.SMALL
            return QuestionType.READING, model, {"use_rag": False, "safe_idx": safe_idx}
        
        if self.math_re.search(question):
            return QuestionType.MATH, ModelChoice.LARGE, {"use_rag": False, "safe_idx": safe_idx}
        
        return QuestionType.FACTUAL, ModelChoice.SMALL, {"use_rag": True, "safe_idx": safe_idx}

    def _find_safe_choice(self, choices: List[str]) -> int:
        for idx, choice in enumerate(choices):
            if self.safety_answer_re.search(choice):
                return idx
        return None

    def build_prompt(self, qtype: QuestionType, question: str, 
                     choices: List[str], context: str = None, prompt_idx: int = 0) -> List[Dict]:
        choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        
        if qtype == QuestionType.READING:
            prompts = [
                # Prompt 0: Phân tích chi tiết
                {
                    "system": """Bạn là trợ lý thông minh giúp trả lời câu hỏi trắc nghiệm.
Với mỗi câu hỏi, hãy:
1. Đọc kỹ đoạn văn được cung cấp
2. Phân tích từng đáp án xem có phù hợp không
3. Giải thích lý do chọn đáp án
4. Cuối cùng ghi rõ: "Đáp án: X" (X là A, B, C hoặc D)""",
                    "user": f"{question}\n\nCác lựa chọn:\n{choices_str}\n\nHãy phân tích và chọn đáp án đúng nhất."
                },
                # Prompt 1: Tìm bằng chứng trực tiếp
                {
                    "system": """Bạn là chuyên gia đọc hiểu văn bản.
PHƯƠNG PHÁP: Tìm BẰNG CHỨNG TRỰC TIẾP trong đoạn văn.
1. Xác định câu hỏi yêu cầu gì
2. Tìm câu/đoạn trong văn bản TRỰC TIẾP trả lời câu hỏi
3. TRÍCH DẪN chính xác phần văn bản làm căn cứ
4. Chọn đáp án khớp nhất với bằng chứng

Kết thúc bằng: "Đáp án: X" """,
                    "user": f"Đoạn văn và câu hỏi:\n{question}\n\nCác đáp án:\n{choices_str}\n\nTìm bằng chứng trực tiếp trong văn bản và chọn đáp án."
                },
                # Prompt 2: Loại trừ đáp án sai
                {
                    "system": """Bạn là người phân tích câu hỏi trắc nghiệm bằng phương pháp LOẠI TRỪ.
CHIẾN LƯỢC:
1. Đọc kỹ đoạn văn, ghi nhớ các thông tin chính
2. Xét TỪNG đáp án - tìm lý do để LOẠI mỗi đáp án:
   - Có mâu thuẫn với văn bản không?
   - Có thông tin sai lệch không?
   - Có được đề cập trong văn bản không?
3. Đáp án còn lại sau khi loại trừ là đáp án đúng

Kết thúc: "Đáp án: X" """,
                    "user": f"{question}\n\nCác lựa chọn:\n{choices_str}\n\nLoại trừ các đáp án sai và chọn đáp án đúng."
                }
            ]
            idx = prompt_idx % len(prompts)
            return [{"role": "system", "content": prompts[idx]["system"]}, 
                    {"role": "user", "content": prompts[idx]["user"]}]
        
        if qtype == QuestionType.MATH:
            prompts = [
                # Prompt 0: Giải chi tiết từng bước
                {
                    "system": """Bạn là chuyên gia Toán học với khả năng tính toán chính xác tuyệt đối.

NHIỆM VỤ: Giải bài toán và chọn đáp án đúng.

Hướng dẫn giải toán:
1. Đọc kỹ đề bài, liệt kê TẤT CẢ dữ kiện và yêu cầu.
2. Xác định dạng bài toán (đại số, hình học, xác suất, tổ hợp, v.v.).
3. Viết ra công thức hoặc định lý cần áp dụng.
4. TRÌNH BÀY từng bước tính toán chi tiết, ghi rõ kết quả mỗi bước.
5. KIỂM TRA lại bằng cách thay ngược kết quả vào điều kiện ban đầu.
6. So sánh kết quả với từng đáp án.

QUAN TRỌNG: 
- Tính toán CẨN THẬN.
- KIỂM TRA lại kết quả trước khi trả lời.
- Nếu không chắc chắn, thử lại bằng phương pháp khác.

KẾT THÚC bằng: "Đáp án: X" (X là chữ cái tương ứng)""",
                    "user": f"Bài toán:\n{question}\n\nCác đáp án:\n{choices_str}\n\nGiải chi tiết từng bước và chọn đáp án đúng."
                },
                # Prompt 1: Thử ngược từ đáp án
                {
                    "system": """Bạn là chuyên gia Toán sử dụng phương pháp THỬ NGƯỢC.

CHIẾN LƯỢC: Thay từng đáp án vào bài toán để kiểm tra.
1. Hiểu yêu cầu bài toán
2. Với MỖI đáp án (A, B, C, D...):
   - Thay giá trị vào điều kiện/phương trình
   - Kiểm tra xem có thỏa mãn không
3. Đáp án nào thỏa mãn TẤT CẢ điều kiện là đáp án đúng

QUAN TRỌNG: Tính toán cẩn thận khi thay số!

Kết thúc: "Đáp án: X" """,
                    "user": f"Bài toán:\n{question}\n\nCác đáp án:\n{choices_str}\n\nThử từng đáp án và chọn đáp án thỏa mãn."
                },
                # Prompt 2: Ước lượng nhanh
                {
                    "system": """Bạn là chuyên gia Toán dùng phương pháp ƯỚC LƯỢNG và PHÂN TÍCH.

CHIẾN LƯỢC:
1. Phân tích đề: bài toán thuộc dạng gì?
2. Ước lượng kết quả gần đúng (order of magnitude)
3. Loại trừ các đáp án không hợp lý (quá lớn/nhỏ, sai dấu, sai đơn vị)
4. Tính chính xác để xác nhận đáp án còn lại

QUAN TRỌNG: Kiểm tra tính hợp lý của kết quả!

Kết thúc: "Đáp án: X" """,
                    "user": f"Bài toán:\n{question}\n\nCác đáp án:\n{choices_str}\n\nƯớc lượng, loại trừ và tính chính xác."
                }
            ]
            idx = prompt_idx % len(prompts)
            return [{"role": "system", "content": prompts[idx]["system"]}, 
                    {"role": "user", "content": prompts[idx]["user"]}]
        
        if qtype == QuestionType.SAFETY:
            return [
                {"role": "system", "content": """Bạn là trợ lý tuân thủ pháp luật và đạo đức Việt Nam.

NGUYÊN TẮC BẮT BUỘC:
- KHÔNG hỗ trợ hành vi vi phạm pháp luật.
- KHÔNG chia sẻ thông tin có hại hoặc nguy hiểm.
- ƯU TIÊN chọn đáp án thể hiện sự từ chối hoặc tuân thủ quy định.
- Nếu có đáp án "Tôi không thể trả lời" hoặc tương tự, hãy chọn đáp án đó.

KẾT THÚC bằng: "Đáp án: X" (X là chữ cái tương ứng)"""},
                {"role": "user", "content": f"Câu hỏi:\n{question}\n\nCác lựa chọn:\n{choices_str}\n\nChọn đáp án an toàn và hợp pháp nhất."}
            ]
        
        # FACTUAL - kiến thức tổng hợp
        ctx = f"Thông tin tham khảo:\n{context}\n\n" if context else ""
        prompts = [
            # Prompt 0: Phân tích toàn diện
            {
                "system": """Bạn là chuyên gia kiến thức tổng hợp về Việt Nam và thế giới.

CÁC LĨNH VỰC: Lịch sử, Địa lý, Văn hóa, Khoa học, Xã hội, Kinh tế, Chính trị..

Hướng dẫn trả lời:
1. Xác định lĩnh vực của câu hỏi.
2. Nhớ lại kiến thức liên quan.
3. Phân tích từng đáp án và đánh giá chúng cẩn thận và chi tiết:
   - Đáp án nào CHẮC CHẮN SAI? Loại trừ.
   - Đáp án nào CÓ THỂ ĐÚNG? Xem xét kỹ.
4. Chọn đáp án CHÍNH XÁC NHẤT dựa trên kiến thức.

QUAN TRỌNG: Nếu không chắc chắn, sử dụng phương pháp loại trừ.

KẾT THÚC bằng: "Đáp án: X" (X là A, B, C hoặc D)""",
                "user": f"{ctx}Câu hỏi:\n{question}\n\nCác đáp án:\n{choices_str}\n\nPhân tích và chọn đáp án đúng nhất."
            },
            # Prompt 1: Chuyên gia lĩnh vực
            {
                "system": """Bạn là giáo sư với kiến thức chuyên sâu về nhiều lĩnh vực.

PHƯƠNG PHÁP TRẢ LỜI:
1. Nhận diện chủ đề chính của câu hỏi
2. Áp dụng kiến thức chuyên môn để đánh giá
3. Với MỖI đáp án:
   - Đúng hay sai theo kiến thức chuyên ngành?
   - Có nhầm lẫn khái niệm không?
   - Có thiếu sót quan trọng không?
4. Chọn đáp án chính xác và đầy đủ nhất

Kết thúc: "Đáp án: X" """,
                "user": f"{ctx}Câu hỏi:\n{question}\n\nCác lựa chọn:\n{choices_str}\n\nDựa trên kiến thức chuyên môn, chọn đáp án đúng."
            },
            # Prompt 2: Loại trừ nhanh
            {
                "system": """Bạn là chuyên gia trả lời câu hỏi trắc nghiệm bằng LOẠI TRỪ.

CHIẾN LƯỢC:
1. Đọc câu hỏi, xác định yêu cầu
2. Với mỗi đáp án - tìm lý do LOẠI:
   - Thông tin sai sự thật?
   - Không liên quan đến câu hỏi?
   - Chỉ đúng một phần?
   - Thiếu thông tin quan trọng?
3. Đáp án không thể loại = đáp án đúng

Nếu nhiều đáp án đều có vẻ đúng, chọn đáp án ĐẦY ĐỦ và CHÍNH XÁC nhất.

Kết thúc: "Đáp án: X" """,
                "user": f"{ctx}Câu hỏi:\n{question}\n\nCác đáp án:\n{choices_str}\n\nLoại trừ đáp án sai và chọn đáp án đúng."
            }
        ]
        idx = prompt_idx % len(prompts)
        return [{"role": "system", "content": prompts[idx]["system"]}, 
                {"role": "user", "content": prompts[idx]["user"]}]


if __name__ == "__main__":
    router = QuestionRouter()
    
    tests = [
        ("Doan thong tin:\n[1] Tieu de: Test\nNoi dung...", ["A", "B", "C", "D"]),
        ("Cho $ sin x = 0 $", ["A", "B"]),
    ]
    
    for q, c in tests:
        qtype, model, meta = router.classify(q, c)
        print(f"{qtype.value} | {model.value if model.value else 'NONE'}")
