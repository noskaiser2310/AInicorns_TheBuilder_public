import re
from typing import Dict, List, Tuple, Optional
from enum import Enum


class QuestionType(Enum):
    READING = "reading"
    MATH = "math"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    SOCIAL_HUMANITIES = "social_humanities"
    SAFETY = "safety"
    GENERAL = "general"
    FACTUAL = "factual" # Kept for backward compatibility if needed


class ModelChoice(Enum):
    LARGE = "large"
    SMALL = "small"
    NONE = None


class QuestionSubType(Enum):
    # Reading
    MAIN_IDEA = "main_idea"
    DETAIL = "detail"
    INFERENCE = "inference"
    VOCABULARY = "vocabulary"
    
    # STEM
    CALCULUS = "calculus"
    GEOMETRY = "geometry"
    ALGEBRA = "algebra"
    PROBABILITY = "probability"
    MECHANICS = "mechanics"
    ELECTROMAGNETISM = "electromagnetism"
    ORGANIC_CHEM = "organic_chem"
    GENETICS = "genetics"
    
    # Generic STEM
    BIOLOGY = "biology"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    MATH = "math"
    SCIENCE = "science"
    GENERAL = "general"

    # Social
    HISTORY = "history"
    GEOGRAPHY = "geography"
    LAW = "law"
    POLITICS = "politics"
    ECONOMICS = "economics"
    CULTURE = "culture"
    
    # Safety
    REFUSAL = "refusal"


class QuestionRouter:
    """
    Router nâng cao với cơ chế 'Context-Aware Priority'
    Ưu tiên độ chính xác (Precision) lên hàng đầu.
    """
    
    # 1. READING PATTERNS (High Priority)
    READING_PATTERNS = [
        r"Đoạn thông tin:", r"Văn bản:", r"Bài viết:", r"Đoạn văn sau:",
        r"\[1\] Tiêu đề:", r"Đọc đoạn văn", r"Dựa vào đoạn văn",
        r"Theo tác giả", r"Ý chính của đoạn", r"Thông tin nào sau đây không có",
    ]
    
    # 2. SAFETY PATTERNS (Critical Priority)
    SAFETY_KEYWORDS = [
        # Evasion/Fraud
        r"tránh thuế", r"lách luật", r"trốn thuế", r"vi phạm", r"gian lận",
        r"làm giả", r"trái phép", r"bất hợp pháp", r"che giấu",
        # Sabotage/Destruction  
        r"phá hoại", r"làm suy yếu", r"gây thiệt hại", r"phá vỡ",
        r"tấn công mạng", r"xâm phạm", r"bom", r"thuốc nổ",
        # Corruption
        r"tham nhũng", r"hối lộ", r"tham ô", r"lợi dụng chức vụ",
        r"lạm dụng quyền lực", r"biển thủ", r"rửa tiền",
        # Incitement/Division
        r"kích động", r"chia rẽ", r"xuyên tạc", r"gây mâu thuẫn",
        r"chống phá", r"lật đổ", r"phản động",
    ]
    
    SAFETY_ANSWER_PATTERNS = [
        r"tôi không thể", r"Tôi không thể", r"không thể chia sẻ",
        r"không thể cung cấp", r"không thể trả lời", r"không thể hỗ trợ",
        r"từ chối", r"vi phạm pháp luật", r"không hợp pháp",
    ]

    # 3. DOMAIN PATTERNS (Context-Aware)
    SUBTYPE_PATTERNS = {
        QuestionSubType.SCIENCE: [
            r"khoa học", r"tự nhiên", r"công nghệ", r"môi trường",
            r"biến đổi khí hậu", r"năng lượng tái tạo", r"vũ trụ",
        ],
        QuestionSubType.POLITICS: [
            r"Chủ Tịch Hồ Chí Minh", r"Hồ Chí Minh", r"Bác Hồ", r"Nguyễn Ái Quốc",
            r"Chủ nghĩa Mác", r"Lênin", r"Tư tưởng Hồ Chí Minh",
            r"Đảng Cộng sản", r"Nhà nước", r"Chính phủ", r"Quốc hội",
            r"Mặt trận Tổ quốc", r"Hệ thống chính trị", r"Tổng Bí thư",
            r"quyền lực chính trị", r"giai cấp", r"cách mạng",
            r"đại đoàn kết", r"dân chủ", r"xã hội chủ nghĩa",
            r"công chức", r"viên chức", r"bầu cử", r"hiến pháp",
        ],
        QuestionSubType.HISTORY: [
            r"lịch sử", r"thời kỳ", r"triều đại", r"năm nào", r"thế kỷ",
            r"chiến tranh", r"cách mạng", r"khởi nghĩa", r"vua", r"hoàng đế",
            r"Trần Nhân Tông", r"nhà Trần", r"nhà Nguyễn", r"nhà Lê",
            r"Đại Việt", r"Pháp thuộc", r"Mông Cổ", r"Điện Biên Phủ",
        ],
        QuestionSubType.LAW: [
            r"luật", r"pháp luật", r"nghị định", r"thông tư", r"điều \d+", r"nghị quyết",
            r"bộ luật", r"xử phạt", r"vi phạm", r"hành chính", r"hình sự",
            r"tố tụng", r"hiến pháp", r"hợp đồng", r"lao động", r"sở hữu trí tuệ",
            r"chế tài", r"quy định", r"lễ hội", r"xử lý kỷ luật", r"truy cứu",
            r"trách nhiệm", r"pháp lý", r"cơ quan nhà nước",
        ],
        QuestionSubType.ECONOMICS: [
            r"kinh tế", r"tài chính", r"thương mại", r"GDP", r"lạm phát",
            r"thị trường", r"cổ phiếu", r"ngân hàng", r"tiền tệ",
            r"cung cầu", r"lợi nhuận", r"doanh thu", r"chi phí cơ hội",
        ],
        
        # --- STEM (Science) ---
        QuestionSubType.BIOLOGY: [
            r"sinh học", r"tế bào", r"gen\b", r"dna", r"rna",
            r"quần thể", r"di truyền", r"đột biến", r"nhiễm sắc thể",
            r"Hardy-Weinberg", r"alen", r"kiểu gen", r"kiểu hình",
            r"enzyme", r"protein", r"axit amin", r"hô hấp tế bào", r"quang hợp",
        ],
        QuestionSubType.CHEMISTRY: [
            # More specific patterns to avoid false positives
            r"hóa học", r"phản ứng hóa học", r"chất hóa học", r"hợp chất", r"nguyên tố hóa học",
            r"axit\b", r"bazơ", r"muối hóa học", r"oxi hóa", r"khử",
            r"phân tử", r"nguyên tử", r"ion\b", r"liên kết hóa học",
            r"mol\b", r"nồng độ mol", r"dung dịch", r"kết tủa", r"\bpH\b",
            r"hữu cơ", r"đồng phân", r"hydrocacbon", r"phương trình hóa học",
        ],
        QuestionSubType.PHYSICS: [
            r"vật lý", r"cơ học", r"động lực học", r"tĩnh học",
            r"điện trở", r"điện áp", r"dòng điện", r"từ trường",
            r"sóng", r"dao động", r"con lắc", r"lò xo", r"tần số",
            r"quang học", r"khúc xạ", r"phản xạ", r"thấu kính",
            r"nhiệt độ", r"áp suất", r"thể tích", r"khí lý tưởng",
            r"gia tốc", r"vận tốc", r"lực", r"năng lượng",
            r"hạt nhân", r"phóng xạ", r"proton", r"electron",
        ],
        QuestionSubType.MATH: [
            r"\$.*\$", r"\\frac", r"\\sqrt", r"\\sum", r"\\int",
            r"phương trình", r"hệ phương trình", r"bất phương trình",
            r"đạo hàm", r"tích phân", r"xác suất", r"thống kê",
            r"tính giá trị", r"biểu thức", r"hàm số", r"đồ thị",
            r"vector", r"ma trận", r"định thức", r"logarit",
        ],
    }

    def __init__(self):
        # Pre-compile regex for performance
        self.reading_re = re.compile("|".join(self.READING_PATTERNS), re.IGNORECASE)
        self.safety_answer_re = re.compile("|".join(self.SAFETY_ANSWER_PATTERNS), re.IGNORECASE)
        self.safety_keywords_re = re.compile("|".join(self.SAFETY_KEYWORDS), re.IGNORECASE)
        
        self.subtype_res = {}
        for subtype, patterns in self.SUBTYPE_PATTERNS.items():
            self.subtype_res[subtype] = re.compile("|".join(patterns), re.IGNORECASE)

    def classify(self, question: str, choices: List[str]) -> Tuple[QuestionType, ModelChoice, Dict]:
        """
        Phân loại câu hỏi với độ chính xác cao nhất (Precision-Focused).
        """
        
        # 1. READING COMPREHENSION (Highest Priority)
        # Nếu có dấu hiệu đọc hiểu hoặc câu hỏi quá dài -> READING
        if self.reading_re.search(question) or len(question) > 1000:
            subtype = self._detect_reading_subtype(question)
            return QuestionType.READING, ModelChoice.LARGE, {
                "subtype": subtype.value
            }
        
        # 2. SAFETY CHECK (After Reading)
        # Chỉ SAFETY nếu: có đáp án từ chối VÀ câu hỏi chứa từ khóa nguy hiểm
        safe_idx = self._find_safe_choice(choices)
        if safe_idx is not None and self.safety_keywords_re.search(question):
            return QuestionType.SAFETY, ModelChoice.SMALL, {
                "subtype": QuestionSubType.REFUSAL.value,
                "safe_idx": safe_idx
            }

        # 3. SOCIAL & HUMANITIES CHECK (Priority over STEM to fix labels)
        for subtype in [QuestionSubType.POLITICS, QuestionSubType.HISTORY, QuestionSubType.LAW, QuestionSubType.ECONOMICS]:
            if self.subtype_res[subtype].search(question):
                return QuestionType.SOCIAL_HUMANITIES, ModelChoice.SMALL, { 
                    "subtype": subtype.value,
                    "use_rag": False 
                }

        # 4. STEM CHECK (Science & Math)
        # Check Biology & Chemistry trước
        for subtype in [QuestionSubType.BIOLOGY, QuestionSubType.CHEMISTRY]:
            if self.subtype_res[subtype].search(question):
                return getattr(QuestionType, subtype.name), ModelChoice.SMALL, {
                    "subtype": subtype.value,
                    "is_stem": True
                }
        
        # Check Physics & Math sau cùng
        has_latex = bool(re.search(r'\$.*\$|\\frac|\\sqrt|\\sum|\\int', question))
        
        if self.subtype_res[QuestionSubType.PHYSICS].search(question):
            return QuestionType.PHYSICS, ModelChoice.LARGE, {
                "subtype": QuestionSubType.PHYSICS.value,
                "is_stem": True,
                "has_latex": has_latex
            }
            
        if self.subtype_res[QuestionSubType.MATH].search(question) or has_latex:
             return QuestionType.MATH, ModelChoice.LARGE, {
                "subtype": QuestionSubType.ALGEBRA.value, # Default math subtype
                "is_stem": True,
                "has_latex": has_latex
            }

        # 5. FALLBACK (General Knowledge)
        return QuestionType.GENERAL, ModelChoice.SMALL, {
            "subtype": "general_knowledge"
        }

    def _find_safe_choice(self, choices: List[str]) -> Optional[int]:
        for idx, choice in enumerate(choices):
            if self.safety_answer_re.search(choice):
                return idx
        return None

    def _detect_reading_subtype(self, question: str) -> QuestionSubType:
        q_lower = question.lower()
        if any(w in q_lower for w in ["ý chính", "chủ đề", "nội dung chính"]): return QuestionSubType.MAIN_IDEA
        if any(w in q_lower for w in ["chi tiết", "theo đoạn"]): return QuestionSubType.DETAIL
        if any(w in q_lower for w in ["suy luận", "ngụ ý"]): return QuestionSubType.INFERENCE
        if any(w in q_lower for w in ["nghĩa của từ", "thay thế"]): return QuestionSubType.VOCABULARY
        return QuestionSubType.DETAIL

    def _detect_math_subtype(self, question: str) -> QuestionSubType:
        for subtype in [QuestionSubType.GEOMETRY, QuestionSubType.CALCULUS, 
                        QuestionSubType.PROBABILITY, QuestionSubType.ALGEBRA]:
            if subtype in self.subtype_res and self.subtype_res[subtype].search(question):
                return subtype
        return QuestionSubType.ARITHMETIC

    def _detect_factual_subtype(self, question: str) -> QuestionSubType:
        # Check STEM subtypes first (more specific)
        for subtype in [QuestionSubType.PHYSICS, QuestionSubType.CHEMISTRY, 
                        QuestionSubType.BIOLOGY]:
            if subtype in self.subtype_res and self.subtype_res[subtype].search(question):
                return subtype
        
        # Then check other factual subtypes
        for subtype in [QuestionSubType.LAW, QuestionSubType.HISTORY, 
                        QuestionSubType.GEOGRAPHY, QuestionSubType.SCIENCE,
                        QuestionSubType.CULTURE, QuestionSubType.ECONOMICS,
                        QuestionSubType.POLITICS]:
            if subtype in self.subtype_res and self.subtype_res[subtype].search(question):
                return subtype
        return QuestionSubType.GENERAL

    def build_prompt(self, qtype: QuestionType, question: str, 
                     choices: List[str], context: str = None, prompt_idx: int = 0, subtype: str = None) -> List[Dict]:
        """Build single consolidated prompt (no voting)"""
        choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        
        if qtype == QuestionType.READING:
            return self._build_reading_prompt(question, choices_str)
        
        if qtype == QuestionType.MATH:
            return self._build_math_prompt(question, choices_str)
        
        if qtype == QuestionType.SAFETY:
            return self._build_safety_prompt(question, choices_str)
        
        # STEM subjects - use provided subtype or default
        if qtype == QuestionType.PHYSICS:
            return self._build_factual_prompt(question, choices_str, context, subtype=subtype or "physics")
        
        if qtype == QuestionType.CHEMISTRY:
            return self._build_factual_prompt(question, choices_str, context, subtype=subtype or "chemistry")
        
        if qtype == QuestionType.BIOLOGY:
            return self._build_factual_prompt(question, choices_str, context, subtype=subtype or "biology")
        
        # SOCIAL_HUMANITIES - use provided subtype (law, history, politics, economics)
        if qtype == QuestionType.SOCIAL_HUMANITIES:
            return self._build_factual_prompt(question, choices_str, context, subtype=subtype or "general")
        
        # GENERAL and FACTUAL - use provided subtype or auto-detect
        if subtype:
            return self._build_factual_prompt(question, choices_str, context, subtype=subtype)
        return self._build_factual_prompt(question, choices_str, context)

    def _build_reading_prompt(self, question: str, choices_str: str) -> List[Dict]:
        """Advanced reading comprehension prompt with deep analysis"""
        return [
            {"role": "system", "content": """Bạn là chuyên gia đọc hiểu văn bản tiếng Việt cấp cao với 20 năm kinh nghiệm giảng dạy.

=== PHƯƠNG PHÁP PHÂN TÍCH CHUYÊN SÂU ===

BƯỚC 1 - ĐỌC VÀ HIỂU VĂN BẢN:
- Đọc TOÀN BỘ văn bản 
- Xác định: Chủ đề chính là gì? Tác giả muốn truyền tải điều gì?
- Ghi nhận các từ khóa, cụm từ quan trọng

BƯỚC 2 - PHÂN LOẠI CÂU HỎI:
- Ý CHÍNH/NỘI DUNG: Hỏi về thông điệp tổng thể → Tìm câu chủ đề hoặc tóm tắt ý
- CHI TIẾT CỤ THỂ: Hỏi về thông tin có trong văn bản → Tìm câu chứa thông tin đó
- SUY LUẬN: Hỏi "có thể suy ra" → Dựa vào thông tin để rút ra kết luận logic
- TỪ VỰNG/NGỮ NGHĨA: Hỏi nghĩa của từ → Xác định nghĩa dựa vào ngữ cảnh xung quanh
- THÁI ĐỘ/QUAN ĐIỂM: Hỏi về cảm xúc/ý kiến tác giả → Tìm từ ngữ biểu cảm

BƯỚC 3 - TRÍCH DẪN BẰNG CHỨNG:
- TÌM câu/đoạn trong văn bản TRỰC TIẾP liên quan đến câu hỏi
- GHI RÕ: "Bằng chứng: [trích dẫn nguyên văn từ văn bản]"
- Nếu không tìm thấy bằng chứng rõ ràng → Cẩn thận, có thể là câu hỏi suy luận

BƯỚC 4 - ĐÁNH GIÁ TỪNG ĐÁP ÁN:
Với MỖI đáp án A, B, C, D:
- ✓ ĐÚNG: Có bằng chứng trực tiếp trong văn bản
- ✗ SAI: Mâu thuẫn với nội dung văn bản  
- ✗ THÊM: Có thông tin KHÔNG có trong văn bản (bịa thêm)
- ✗ THIẾU: Chỉ đúng một phần, bỏ sót ý quan trọng
- ✗ SUY DIỄN QUÁ XA: Kết luận không được hỗ trợ đủ

BƯỚC 5 - QUYẾT ĐỊNH CUỐI CÙNG:
- Chọn đáp án được HỖ TRỢ TRỰC TIẾP và ĐẦY ĐỦ NHẤT
- Nếu 2+ đáp án có vẻ đúng → Chọn đáp án HOÀN CHỈNH và CHÍNH XÁC hơn
- Nếu không có đáp án hoàn hảo → Chọn đáp án ÍT SAI NHẤT

=== CẢNH BÁO SAI LẦM PHỔ BIẾN ===
Chọn đáp án vì "nghe hay" nhưng không có trong văn bản
Chọn đáp án có từ khóa giống văn bản nhưng nghĩa khác
Bỏ qua đáp án đúng vì nó quá "đơn giản"
Suy diễn thêm thông tin mà văn bản không đề cập

=== ĐỊNH DẠNG TRẢ LỜI ===
1. Loại câu hỏi: [Ý chính/Chi tiết/Suy luận/Từ vựng]
2. Bằng chứng: "[trích dẫn từ văn bản]"
3. Phân tích:
   - A: [đánh giá]
   - B: [đánh giá]
   - C: [đánh giá]
   - D: [đánh giá]
4. Đáp án cuối cùng: X"""},
            {"role": "user", "content": f"""{question}

Các lựa chọn:
{choices_str}

Hãy phân tích CHUYÊN SÂU theo đúng phương pháp 5 bước và chọn đáp án chính xác nhất."""}
        ]

    def _build_reading_prompt_v2(self, question: str, choices_str: str) -> List[Dict]:
        """Reading prompt v2 - 4-step data extraction method"""
        return [
            {"role": "system", "content": """Bạn là chuyên gia giải quyết bài tập đọc hiểu. Nhiệm vụ của bạn là chọn đáp án đúng nhất dựa trên văn bản.

QUY TẮC CỐT LÕI:
1. Chỉ dùng thông tin trong bài.
2. Với câu hỏi suy luận, phải chỉ rõ các bước logic dẫn đến kết luận.

QUY TRÌNH 4 BƯỚC (BẮT BUỘC):

BƯỚC 1: Phân tích & Định vị
- Xác định từ khóa.
- Tìm tất cả các đoạn văn có liên quan (có thể nằm rải rác ở nhiều nơi).

BƯỚC 2: Trích dẫn dữ liệu (Data Extraction)
- Trích dẫn nguyên văn các câu chứa thông tin (Evidence 1, Evidence 2...).
- KHÔNG được bỏ qua bước này.

BƯỚC 3: Xử lý Logic (Dành cho câu hỏi khó/suy luận)
- Nếu cần tính toán: Hãy viết phép tính ra (Ví dụ: 50k x 3 ngày = 150k).
- Nếu cần so sánh: Đặt thông tin của các đối tượng cạnh nhau để so.
- Nếu cần tìm nguyên nhân: Tìm mối liên hệ "Vì... nên..." giữa các trích dẫn.

BƯỚC 4: Kết luận
- ĐÁP ÁN CUỐI CÙNG: X """},
            {"role": "user", "content": f"""{question}

Các lựa chọn:
{choices_str}

Hãy thực hiện đúng quy trình 4 bước và chọn đáp án chính xác nhất."""}
        ]

    def _build_math_prompt(self, question: str, choices_str: str) -> List[Dict]:
        """Consolidated Math prompt with verification"""
        return [
            {"role": "system", "content": """Bạn là một Giáo sư Toán học và Chuyên gia Tính toán Hình thức (Formal Computation). Nhiệm vụ của bạn là giải quyết các bài toán với độ chính xác tuyệt đối, không chấp nhận sai số.

TƯ DUY THUẬT TOÁN (ALGORITHMIC REASONING):
Để giải quyết bài toán này, bạn BẮT BUỘC phải tuân thủ quy trình 5 bước sau đây như một chương trình máy tính:

1. [PARSE] PHÂN TÍCH DỮ LIỆU:
   - Input: Liệt kê tất cả biến số (x, y, n, P...) và giá trị của chúng.
   - Goal: Xác định rõ ràng đại lượng cần tìm.
   - Constraints: Lưu ý các điều kiện xác định (mẫu số khác 0, biểu thức trong căn >= 0, xác suất [0,1]).

2. [MODEL] MÔ HÌNH HÓA TOÁN HỌC:
   - Ánh xạ bài toán vào lĩnh vực cụ thể:
     + Giải tích: Đạo hàm, Tích phân, Laplace, Giới hạn,...
     + Đại số: Ma trận, Hệ phương trình, Số phức,...
     + Xác suất/Thống kê: Tổ hợp, Biến ngẫu nhiên, Phân phối chuẩn/Poisson,...  
   - VIẾT CÔNG THỨC GỐC: Viết công thức tổng quát trước khi thay số (Ví dụ: Định lý Bayes, Công thức nhân đôi, Biến đổi Laplace,...).

3. [EXECUTE] TÍNH TOÁN TỪNG BƯỚC (STEP-BY-STEP):
   - Thay số vào công thức.
   - Thực hiện biến đổi đại số trên từng dòng riêng biệt.
   - KHÔNG ĐƯỢC LÀM TẮT. Ví dụ: Nếu tính tích phân, hãy tìm nguyên hàm trước, sau đó thế cận.
   - Nếu là phương trình: Chuyển vế -> Đổi dấu -> Rút gọn.

4. [VERIFY] KIỂM TRA NGƯỢC:
   - Kiểm tra logic (Sanity Check): Kết quả có vi phạm miền xác định không? (VD: Xác suất > 1 là sai).
   - Kiểm tra đơn vị/thứ nguyên (nếu có).
   - Thử lại nghiệm vào phương trình gốc (nếu là bài giải phương trình).

5. [MATCH] ĐỐI CHIẾU & KẾT LUẬN:
   - So sánh kết quả tính được với danh sách lựa chọn (A, B, C, D...).
   - Nếu kết quả không khớp chính xác, hãy kiểm tra xem có cần làm tròn hoặc đổi dạng biểu diễn (ví dụ: 0.5 vs 1/2) không.
   - Chọn đáp án khớp nhất.

ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:
---
Phân tích: [Bước 1 & 2]
Giải chi tiết: [Bước 3 - Hiển thị rõ các bước biến đổi]
Kiểm tra: [Bước 4]
Đáp án cuối cùng: [Chỉ ghi 1 ký tự: A, B, C, D...]
---

VÍ DỤ MINH HỌA :
Input: Tìm nghiệm của x^2 - 5x + 6 = 0.
A. 1, 6
B. 2, 3
C. -2, -3
D. 0, 5

Output:
Phân tích: Phương trình bậc 2 dạng ax^2 + bx + c = 0 với a=1, b=-5, c=6.
Giải chi tiết:
Tính Delta = b^2 - 4ac = (-5)^2 - 4*1*6 = 25 - 24 = 1.
Vì Delta > 0, phương trình có 2 nghiệm phân biệt:
x1 = (-b + sqrt(Delta)) / 2a = (5 + 1) / 2 = 3.
x2 = (-b - sqrt(Delta)) / 2a = (5 - 1) / 2 = 2.
Kiểm tra: 2^2 - 5(2) + 6 = 4 - 10 + 6 = 0 (Đúng).
Đáp án cuối cùng: B """},
            {"role": "user", "content": f"""Giải bài toán sau:

{question}

Các đáp án:
{choices_str}

Giải chi tiết và chọn đáp án đúng."""}
        ]

    def _build_safety_prompt(self, question: str, choices_str: str) -> List[Dict]:
        """Safety prompt - balanced version that still reasons properly"""
        return [
            {"role": "system", "content": """Bạn là chuyên gia phân tích với kiến thức đa dạng về pháp luật, đạo đức, khoa học, lịch sử, kinh tế và các lĩnh vực chuyên môn.

NGUYÊN TẮC SUY LUẬN:
1. ĐỌC KỸ câu hỏi để hiểu đúng yêu cầu
2. PHÂN TÍCH từng đáp án dựa trên:
   - Tính chính xác về mặt kiến thức
   - Tính hợp pháp và đạo đức
   - Tính logic và thực tiễn

3. NẾU câu hỏi yêu cầu thông tin vi phạm pháp luật hoặc đạo đức (ví dụ: cách trốn thuế, phá hoại, tham nhũng...):
   → Chọn đáp án từ chối (ví dụ: "Tôi không thể trả lời")
   
4. NẾU câu hỏi là câu hỏi kiến thức bình thường:
   - Toán học: Tính toán chính xác, kiểm tra từng bước
   - Vật lý/Hóa học: Áp dụng công thức, đơn vị, định luật
   - Sinh học: Khái niệm khoa học, quy trình tự nhiên
   - Lịch sử: Sự kiện, năm tháng, nhân vật, ý nghĩa
   - Địa lý: Vị trí, đặc điểm, số liệu
   - Kinh tế: Nguyên lý, công thức, quy luật thị trường
   - Pháp luật: Điều khoản, thẩm quyền, quy định cụ thể
   → Chọn đáp án đúng nhất dựa trên suy luận logic và kiến thức chuyên môn

OUTPUT FORMAT:
- Phân tích ngắn gọn từng đáp án
- Kết luận: "Đáp án cuối cùng: X" """},
            {"role": "user", "content": f"""Phân tích và chọn đáp án phù hợp nhất:

{question}

Các đáp án:
{choices_str}

Suy luận và chọn đáp án."""}
        ]

    def _build_factual_prompt(self, question: str, choices_str: str, context: str = None, subtype: str = None) -> List[Dict]:
        """Specialized Factual prompts based on sub-type"""
        ctx = f"THÔNG TIN THAM KHẢO:\n{context}\n\n" if context else ""
        
        # Detect subtype from question if not provided
        if not subtype:
            detected = self._detect_factual_subtype(question)
            subtype = detected.value if detected else "general"
        
        # LAW - Legal questions
        if subtype == "law":
            return [
                {"role": "system", "content": """Bạn là chuyên gia PHÁP LUẬT Việt Nam.

KIẾN THỨC CHUYÊN MÔN:
- Hiến pháp, Bộ luật Hình sự, Bộ luật Dân sự, Luật Hành chính
- Nghị định, Thông tư, Quy định
- Thẩm quyền các cơ quan: Tòa án, Viện kiểm sát, Công an, Chính phủ

PHƯƠNG PHÁP PHÂN TÍCH:
1. Xác định: Câu hỏi yêu cầu gì? Constraints là gì?
2. Loại trừ: Đáp án nào vi phạm constraints hoặc sai rõ ràng?
3. Phân tích: Với mỗi đáp án còn lại, nêu evidence ủng hộ/bác bỏ
4. Quyết định: Chọn đáp án có evidence mạnh nhất

Kết thúc: "Đáp án cuối cùng: X" """},
                {"role": "user", "content": f"""{ctx}Câu hỏi pháp luật:
{question}

Các đáp án:
{choices_str}

Phân tích theo quy định pháp luật và chọn đáp án đúng."""}
            ]
        
        # HISTORY - Historical questions
        if subtype == "history":
            return [
                {"role": "system", "content": """Bạn là chuyên gia LỊCH SỬ Việt Nam và Thế giới.

KIẾN THỨC CHUYÊN MÔN:
- Lịch sử Việt Nam: Từ thời Hùng Vương đến hiện đại
- Các triều đại, cuộc kháng chiến, cách mạng
- Lịch sử thế giới: Cổ đại, Trung đại, Hiện đại
- Các sự kiện lớn, nhân vật lịch sử

PHƯƠNG PHÁP PHÂN TÍCH:
1. Xác định: Câu hỏi yêu cầu gì? Constraints là gì?
2. Loại trừ: Đáp án nào vi phạm constraints hoặc sai rõ ràng?
3. Phân tích: Với mỗi đáp án còn lại, nêu evidence ủng hộ/bác bỏ
4. Quyết định: Chọn đáp án có evidence mạnh nhất

Kết thúc: "Đáp án cuối cùng: X" """},
                {"role": "user", "content": f"""{ctx}Câu hỏi lịch sử:
{question}

Các đáp án:
{choices_str}

Phân tích theo kiến thức lịch sử và chọn đáp án đúng."""}
            ]
        
        # GEOGRAPHY - Geographical questions
        if subtype == "geography":
            return [
                {"role": "system", "content": """Bạn là chuyên gia ĐỊA LÝ Việt Nam và Thế giới.

KIẾN THỨC CHUYÊN MÔN:
- Địa lý Việt Nam: 63 tỉnh thành, vùng miền, đặc điểm địa hình
- Địa lý Thế giới: Châu lục, quốc gia, thủ đô, địa hình
- Số liệu: Diện tích, dân số, sông núi, biên giới

PHƯƠNG PHÁP PHÂN TÍCH:
1. Xác định: Câu hỏi yêu cầu gì? Constraints là gì?
2. Loại trừ: Đáp án nào vi phạm constraints hoặc sai rõ ràng?
3. Phân tích: Với mỗi đáp án còn lại, nêu evidence ủng hộ/bác bỏ
4. Quyết định: Chọn đáp án có evidence mạnh nhất

Kết thúc: "Đáp án cuối cùng: X" """},
                {"role": "user", "content": f"""{ctx}Câu hỏi địa lý:
{question}

Các đáp án:
{choices_str}

Phân tích theo kiến thức địa lý và chọn đáp án đúng."""}
            ]
        
        # SCIENCE - Scientific questions
        if subtype == "science":
            return [
                {"role": "system", "content": """Bạn là chuyên gia KHOA HỌC TỰ NHIÊN.

KIẾN THỨC CHUYÊN MÔN:
- Vật lý: Cơ học, Điện từ, Nhiệt động học
- Hóa học: Nguyên tử, Phân tử, Phản ứng hóa học
- Sinh học: Tế bào, Di truyền, Sinh thái
- Y học: Cơ thể người, Bệnh tật

PHƯƠNG PHÁP PHÂN TÍCH:
1. Xác định: Câu hỏi yêu cầu gì? Constraints là gì?
2. Loại trừ: Đáp án nào vi phạm constraints hoặc sai rõ ràng?
3. Phân tích: Với mỗi đáp án còn lại, nêu evidence ủng hộ/bác bỏ
4. Quyết định: Chọn đáp án có evidence mạnh nhất

Kết thúc: "Đáp án cuối cùng: X" """},
                {"role": "user", "content": f"""{ctx}Câu hỏi khoa học:
{question}

Các đáp án:
{choices_str}

Phân tích theo kiến thức khoa học và chọn đáp án đúng."""}
            ]
        
        # PHYSICS - Physics questions (separate from MATH)
        if subtype == "physics":
            return [
                {"role": "system", "content": """Bạn là một Giáo sư Vật lý lý thuyết và ứng dụng hàng đầu. Nhiệm vụ của bạn là giải quyết các bài toán Vật lý với độ chính xác tuyệt đối.

QUY TRÌNH SUY LUẬN BẮT BUỘC (CHAIN-OF-THOUGHT):
1. TRÍCH XUẤT DỮ LIỆU (VARIABLES):
   - Liệt kê mọi đại lượng đề bài cho.
   - BẮT BUỘC: Đổi ngay tất cả đơn vị về chuẩn SI (Mét, Kg, Giây, Joule, Coulomb...) trước khi tính.
   - Ví dụ: 5cm -> 0.05m; 600nm -> 6e-7m.

2. XÁC ĐỊNH NGUYÊN LÝ (PRINCIPLES):
   - Bài toán thuộc chuyên đề nào? (Cơ học Newton, Nhiệt động lực học, Mạch điện xoay chiều RLC, Lượng tử...).
   - Viết tên định luật hoặc nguyên lý bảo toàn sẽ sử dụng (Bảo toàn cơ năng, Định luật Ohm, Định luật Hess...).

3. THIẾT LẬP CÔNG THỨC (FORMULATION):
   - Viết công thức gốc dưới dạng ký hiệu (ví dụ: F = ma, U = I*R).
   - Biến đổi công thức để rút ra đại lượng cần tìm về một vế.

4. TÍNH TOÁN CHI TIẾT (CALCULATION):
   - Thay số vào biểu thức đã biến đổi.
   - Thực hiện phép tính từng bước một. KHÔNG ĐƯỢC TÍNH NHẨM TẮT.
   - Lưu ý các hằng số vật lý (c = 3e8, h = 6.626e-34, k_B...).

5. KIỂM TRA (VERIFICATION):
   - Kết quả có ý nghĩa vật lý không? (Ví dụ: Thời gian không thể âm, Động năng không thể âm).
   - So sánh với các lựa chọn (A, B, C, D).

ĐỊNH DẠNG ĐẦU RA:
Phân tích: [Tóm tắt dữ liệu và đổi đơn vị]
Công thức: [Công thức gốc và biến đổi]
Tính toán: [Các bước thay số và kết quả]
Kết thúc: "Đáp án cuối cùng: X" (X là A-L)"""},
                {"role": "user", "content": f"""{ctx}Bài toán VẬT LÝ:
{question}

Các đáp án:
{choices_str}

Giải chi tiết theo phương pháp trên và chọn đáp án đúng."""}
            ]
        
        # CHEMISTRY - Chemistry questions
        if subtype == "chemistry":
            return [
                {"role": "system", "content": """Bạn là chuyên gia HÓA HỌC với kiến thức toàn diện:

KIẾN THỨC CHUYÊN MÔN:
- Cấu tạo nguyên tử: Electron, Proton, Neutron, Lớp vỏ
- Bảng tuần hoàn: Chu kỳ, Nhóm, Xu hướng hóa học
- Liên kết hóa học: Ion, Cộng hóa trị, Kim loại
- Phản ứng hóa học: Oxi hóa-khử, Axit-bazơ, Kết tủa
- Hóa hữu cơ: Hydrocacbon, Nhóm chức, Đồng phân
- Nhiệt động học: Enthalpy, Entropy, Năng lượng tự do

QUY TRÌNH SUY LUẬN BẮT BUỘC:
1. PHƯƠNG TRÌNH PHẢN ỨNG (REACTION):
   - Viết phương trình hóa học cho quá trình được mô tả.
   - BẮT BUỘC: Cân bằng phương trình (kiểm tra bảo toàn nguyên tố và điện tích). Nếu không cân bằng, mọi tính toán sau đó sẽ sai.

2. CHUYỂN ĐỔI SỐ MOL (STOICHIOMETRY):
   - Chuyển tất cả dữ liệu (khối lượng, thể tích, nồng độ) về số Mol.
   - Xác định chất hết, chất dư (Limiting reagent) nếu cần.

3. TÍNH TOÁN THEO YÊU CẦU:
   - Áp dụng các định luật: Bảo toàn khối lượng, Hess (Nhiệt hóa học), Henderson-Hasselbalch (pH đệm), Nernst (Điện hóa).
   - Chú ý đến hệ số tỉ lượng trong phương trình đã cân bằng.

4. XEM XÉT ĐIỀU KIỆN:
   - Kiểm tra điều kiện tiêu chuẩn (STP) hay điều kiện thường.
   - Lưu ý đơn vị năng lượng (kJ vs J) và nhiệt độ (Kelvin vs Celsius).

QUAN TRỌNG:
- Chú ý hệ số cân bằng
- Kiểm tra bảo toàn khối lượng
- Xét đúng điều kiện phản ứng

Kết thúc: "Đáp án cuối cùng: X" (X là A-J)"""},
                {"role": "user", "content": f"""{ctx}Bài toán HÓA HỌC:
{question}

Các đáp án:
{choices_str}

Giải chi tiết và chọn đáp án đúng."""}
            ]
        
        # BIOLOGY - Biology questions  
        if subtype == "biology":
            return [
                {"role": "system", "content": """Bạn là một Chuyên gia Sinh học hiện đại (Di truyền học & Sinh học phân tử).

QUY TRÌNH SUY LUẬN BẮT BUỘC:
1. PHÂN TÍCH TỪ KHÓA:
   - Tìm các từ khóa cốt lõi: "DNA", "RNA", "Hardy-Weinberg", "Alen lặn", "Trội hoàn toàn".
   - Phân biệt rõ các cơ chế: Nguyên phân vs Giảm phân, Phiên mã vs Dịch mã.

2. ÁP DỤNG CÔNG THỨC (NẾU CÓ):
   - Với Di truyền quần thể: Sử dụng p^2 + 2pq + q^2 = 1. Xác định rõ đề bài cho p (tần số alen) hay q^2 (tỷ lệ kiểu hình lặn).
   - Với Di truyền phân tử: A=T, G=X; Số liên kết Hydro, chiều dài gen.

3. SUY LUẬN LOGIC (VỚI LÝ THUYẾT):
   - Dựa trên Học thuyết Tiến hóa hiện đại và Sinh học tế bào.
   - Loại trừ các phương án sai dựa trên mâu thuẫn lý thuyết cơ bản.

4. KẾT LUẬN:
   - Đảm bảo câu trả lời phù hợp với bối cảnh sinh học (ví dụ: không có xác suất > 1, không có số cá thể lẻ).

ĐỊNH DẠNG ĐẦU RA:
Phân tích: [Cơ sở lý thuyết/Công thức]
Suy luận: [Các bước giải]
Đáp án cuối cùng: [Ký tự A/B/C/D]"""},
                {"role": "user", "content": f"""{ctx}Câu hỏi SINH HỌC:
{question}

Các đáp án:
{choices_str}

Phân tích và giải đáp."""}
            ]
        
        # CULTURE - Cultural/social questions
        if subtype == "culture":
            return [
                {"role": "system", "content": """Bạn là chuyên gia VĂN HÓA - XÃ HỘI Việt Nam và Thế giới.

KIẾN THỨC CHUYÊN MÔN:
- Văn hóa Việt Nam: Phong tục, truyền thống, lễ hội
- Văn học: Tác giả, tác phẩm, thời kỳ văn học
- Nghệ thuật: Âm nhạc, hội họa, điêu khắc
- Tôn giáo, tín ngưỡng

PHƯƠNG PHÁP PHÂN TÍCH:
1. Xác định: Câu hỏi yêu cầu gì? Constraints là gì?
2. Loại trừ: Đáp án nào vi phạm constraints hoặc sai rõ ràng?
3. Phân tích: Với mỗi đáp án còn lại, nêu evidence ủng hộ/bác bỏ
4. Quyết định: Chọn đáp án có evidence mạnh nhất

Kết thúc: "Đáp án cuối cùng: X" """},
                {"role": "user", "content": f"""{ctx}Câu hỏi văn hóa:
{question}

Các đáp án:
{choices_str}

Phân tích và chọn đáp án đúng nhất."""}
            ]
        
        # ECONOMICS - Economic questions
        if subtype == "economics":
            return [
                {"role": "system", "content": """Bạn là chuyên gia KINH TẾ.

KIẾN THỨC CHUYÊN MÔN:
- Kinh tế vĩ mô: GDP, lạm phát, thất nghiệp
- Kinh tế vi mô: Cung cầu, thị trường
- Tài chính: Ngân hàng, chứng khoán, tiền tệ
- Thương mại quốc tế

PHƯƠNG PHÁP PHÂN TÍCH:
1. Xác định: Câu hỏi yêu cầu gì? Constraints là gì?
2. Loại trừ: Đáp án nào vi phạm constraints hoặc sai rõ ràng?
3. Phân tích: Với mỗi đáp án còn lại, nêu evidence ủng hộ/bác bỏ
4. Quyết định: Chọn đáp án có evidence mạnh nhất

Kết thúc: "Đáp án cuối cùng: X" """},
                {"role": "user", "content": f"""{ctx}Câu hỏi kinh tế:
{question}

Các đáp án:
{choices_str}

Phân tích và chọn đáp án đúng nhất."""}
            ]
        
        # POLITICS - Political questions
        if subtype == "politics":
            return [
                {"role": "system", "content": """Bạn là chuyên gia CHÍNH TRỊ - NHÀ NƯỚC và bạn đứng ở góc độ người dân Việt Nam và mang cái nhìn tích cực.

KIẾN THỨC CHUYÊN MÔN:
- Hệ thống chính trị Việt Nam: Đảng, Nhà nước, Quốc hội, Chính phủ
- Chính trị quốc tế: Các tổ chức, quan hệ ngoại giao
- Lãnh đạo: Chủ tịch nước, Thủ tướng, Tổng Bí thư

PHƯƠNG PHÁP PHÂN TÍCH:
1. Xác định: Câu hỏi yêu cầu gì? Constraints là gì?
2. Loại trừ: Đáp án nào vi phạm constraints hoặc sai rõ ràng?
3. Phân tích: Với mỗi đáp án còn lại, nêu evidence ủng hộ/bác bỏ
4. Quyết định: Chọn đáp án có evidence mạnh nhất

Luôn kết thúc bằng : "Đáp án cuối cùng: X" """},
                {"role": "user", "content": f"""{ctx}Câu hỏi chính trị:
{question}

Các đáp án:
{choices_str}

Phân tích và chọn đáp án đúng nhất."""}
            ]
        
        # GENERAL - Super powerful prompt for unclassified questions
        return [
            {"role": "system", "content": """Bạn là CHUYÊN GIA ĐA LĨNH VỰC và còn được gọi là bách khoa toàn thư có kiến thức sâu rộng về:
- Khoa học tự nhiên: Vật lý, Hóa học, Sinh học, Toán học
- Khoa học xã hội: Lịch sử, Địa lý, Kinh tế, Chính trị, Pháp luật
- Văn hóa nghệ thuật: Văn học, Âm nhạc, Hội họa, Tôn giáo
- Kiến thức thực tiễn: Đời sống, Xã hội, Công nghệ

=== PHƯƠNG PHÁP PHÂN TÍCH CHUYÊN SÂU ===

BƯỚC 1 - HIỂU CÂU HỎI:
- Đọc kỹ và xác định: CÂU HỎI yêu cầu gì?
- Lĩnh vực: Thuộc về khoa học, xã hội, văn hóa hay đời sống?
- Từ khóa quan trọng: Có từ nào cần chú ý đặc biệt?

BƯỚC 2 - PHÂN TÍCH TỪNG ĐÁP ÁN:
Với MỖI đáp án A, B, C, D... hãy đánh giá:
- ✓ HỢP LÝ: Có logic và phù hợp với câu hỏi
- ✗ MÂU THUẪN: Trái ngược với thông tin đã biết
- ✗ KHÔNG LIÊN QUAN: Không trả lời đúng câu hỏi
- ✗ THIẾU LOGIC: Kết luận không được hỗ trợ đủ
- ? KHÔNG CHẮC: Cần xem xét thêm

BƯỚC 3 - SO SÁNH VÀ LOẠI TRỪ:
- Loại bỏ các đáp án có lỗi rõ ràng
- So sánh các đáp án còn lại
- Tìm đáp án HOÀN CHỈNH và CHÍNH XÁC nhất

BƯỚC 4 - KIỂM TRA CUỐI:
- Đọc lại câu hỏi: Đáp án đã chọn có trả lời ĐÚNG yêu cầu?
- Đáp án đã chọn có logic không?

=== NGUYÊN TẮC QUAN TRỌNG ===
✓ Nếu có THÔNG TIN THAM KHẢO → DỰA VÀO đó để trả lời
✓ Ưu tiên đáp án có BẰNG CHỨNG rõ ràng
✓ Khi không chắc → Chọn đáp án HỢP LÝ NHẤT
✓ Tránh suy diễn quá xa hoặc thêm thông tin không có

=== ĐỊNH DẠNG TRẢ LỜI ===
1. Lĩnh vực: [xác định lĩnh vực câu hỏi]
2. Phân tích từng đáp án:
   - A: [đánh giá]
   - B: [đánh giá]
   - C: [đánh giá]
   - D: [đánh giá]
3. Kết luận: [giải thích nguyên nhân]
4. Đáp án cuối cùng: X"""},
            {"role": "user", "content": f"""{ctx}Câu hỏi:
{question}

Các đáp án:
{choices_str}

Hãy áp dụng phương pháp phân tích chuyên sâu 4 bước và chọn đáp án CHÍNH XÁC NHẤT."""}
        ]


if __name__ == "__main__":
    router = QuestionRouter()
    
    tests = [
        ("Đoạn thông tin:\n[1] Tiêu đề: Test\nNội dung...", ["A", "B", "C", "D"]),
        ("Cho $ sin x = 0 $", ["A", "B"]),
        ("Theo điều 5 luật hình sự, việc xử phạt...", ["A", "B", "C", "D"]),
        ("Thủ đô của Việt Nam là gì?", ["A", "B", "C", "D"]),
    ]
    
    for q, c in tests:
        qtype, model, meta = router.classify(q, c)
        print(f"{qtype.value} | {model.value if model.value else 'NONE'} | subtype: {meta.get('subtype')}")
