import re
from typing import Dict, List, Tuple
from enum import Enum


class QuestionType(Enum):
    READING = "reading"
    FACTUAL = "factual"
    MATH = "math"
    SAFETY = "safety"


class QuestionSubType(Enum):
    # Reading sub-types
    MAIN_IDEA = "main_idea"          # Ý chính
    DETAIL = "detail"                 # Chi tiết
    INFERENCE = "inference"           # Suy luận
    VOCABULARY = "vocabulary"         # Từ vựng
    
    # Factual sub-types
    HISTORY = "history"               # Lịch sử
    GEOGRAPHY = "geography"           # Địa lý
    LAW = "law"                       # Luật, pháp luật
    SCIENCE = "science"               # Khoa học
    CULTURE = "culture"               # Văn hóa, xã hội
    ECONOMICS = "economics"           # Kinh tế
    POLITICS = "politics"             # Chính trị
    GENERAL = "general"               # Kiến thức chung
    
    # STEM sub-types
    # Math
    ALGEBRA = "algebra"               # Đại số
    GEOMETRY = "geometry"             # Hình học
    CALCULUS = "calculus"             # Giải tích
    PROBABILITY = "probability"       # Xác suất, tổ hợp
    ARITHMETIC = "arithmetic"         # Tính toán số học
    
    # Physics
    PHYSICS = "physics"               # Vật lý
    MECHANICS = "mechanics"           # Cơ học
    ELECTROMAGNETISM = "electromagnetism"  # Điện từ
    THERMODYNAMICS = "thermodynamics" # Nhiệt động lực học
    OPTICS = "optics"                 # Quang học
    
    # Chemistry
    CHEMISTRY = "chemistry"           # Hóa học
    
    # Biology
    BIOLOGY = "biology"               # Sinh học
    
    # Safety
    REFUSAL = "refusal"


class ModelChoice(Enum):
    SMALL = "small"
    LARGE = "large"
    NONE = None


class QuestionRouter:
    # Reading patterns
    READING_PATTERNS = [
        r"Đoạn thông tin:",
        r"Văn bản:",
        r"Bài viết:",
        r"Đoạn văn sau:",
        r"\[1\] Tiêu đề:",
        r"Đọc đoạn văn",
        r"Dựa vào đoạn văn",
    ]
    
    # Math patterns
    MATH_PATTERNS = [
        r"\$",
        r"(sin|cos|tan|cot|log|ln|frac|sqrt)",
        r"phương trình",
        r"tính toán",
        r"giải bài toán",
        r"đạo hàm",
        r"tích phân",
        r"xác suất",
        r"tính giá trị",
        r"biểu thức",
    ]
    
    # Sub-type detection patterns
    SUBTYPE_PATTERNS = {
        # Factual sub-types
        QuestionSubType.HISTORY: [
            r"lịch sử", r"thời kỳ", r"triều đại", r"năm nào", r"thế kỷ",
            r"chiến tranh", r"cách mạng", r"cuộc khởi nghĩa", r"vua", r"hoàng đế",
            r"năm \d{3,4}", r"thời đại", r"trận đánh",
        ],
        QuestionSubType.GEOGRAPHY: [
            r"địa lý", r"quốc gia", r"thành phố", r"thủ đô", r"châu lục",
            r"sông", r"núi", r"biển", r"đại dương", r"diện tích", r"dân số",
            r"biên giới", r"lãnh thổ", r"tỉnh", r"vùng",
        ],
        QuestionSubType.LAW: [
            r"luật", r"pháp luật", r"nghị định", r"thông tư", r"điều \d+",
            r"khoản", r"quy định", r"hình sự", r"dân sự", r"hành chính",
            r"tố tụng", r"hiến pháp", r"bộ luật", r"xử phạt", r"vi phạm",
            r"thẩm quyền", r"trách nhiệm", r"nghĩa vụ", r"quyền",
            r"hợp đồng", r"thừa kế", r"sở hữu", r"tài sản", r"lao động",
        ],
        QuestionSubType.SCIENCE: [
            r"khoa học", r"y học",
        ],
        # Physics patterns
        QuestionSubType.PHYSICS: [
            r"vật lý", r"cơ học", r"động lực học", r"tĩnh học",
            r"điện trở", r"điện áp", r"dòng điện", r"từ trường", r"điện trường",
            r"sóng", r"dao động", r"con lắc", r"lò xo", r"tần số", r"bước sóng",
            r"quang học", r"khúc xạ", r"phản xạ", r"thấu kính", r"gương",
            r"nhiệt độ", r"áp suất", r"thể tích", r"mol",
            r"gia tốc", r"vận tốc", r"lực", r"khối lượng", r"năng lượng",
            r"điện tích", r"tụ điện", r"cuộn cảm", r"mạch điện",
            r"hạt nhân", r"phóng xạ", r"neutron", r"proton", r"electron",
            r"quỹ đạo", r"hấp dẫn", r"vệ tinh", r"hành tinh",
            r"\\Omega", r"\\mu", r"\\epsilon", r"\\lambda",
        ],
        # Chemistry patterns
        QuestionSubType.CHEMISTRY: [
            r"hóa học", r"phản ứng", r"chất", r"hợp chất", r"nguyên tố",
            r"axit", r"bazơ", r"muối", r"oxi hóa", r"khử",
            r"phân tử", r"nguyên tử", r"ion", r"liên kết",
            r"mol", r"nồng độ", r"dung dịch", r"kết tủa",
            r"chiral", r"đồng phân", r"xúc tác", r"aldehyt", r"cinnamic",
        ],
        # Biology patterns
        QuestionSubType.BIOLOGY: [
            r"sinh học", r"tế bào", r"gen", r"dna", r"rna",
            r"quần thể", r"di truyền", r"đột biến", r"nhiễm sắc thể",
            r"Hardy-Weinberg", r"alen", r"kiểu gen", r"kiểu hình",
            r"enzyme", r"protein", r"axit amin", r"lipid",
            r"trao đổi chất", r"hô hấp", r"quang hợp",
        ],
        QuestionSubType.CULTURE: [
            r"văn hóa", r"phong tục", r"truyền thống", r"lễ hội", r"tín ngưỡng",
            r"tôn giáo", r"nghệ thuật", r"âm nhạc", r"hội họa", r"văn học",
            r"tác giả", r"tác phẩm", r"nhà văn", r"nhà thơ",
        ],
        QuestionSubType.ECONOMICS: [
            r"kinh tế", r"tài chính", r"thương mại", r"gdp", r"lạm phát",
            r"thị trường", r"cổ phiếu", r"ngân hàng", r"tiền tệ",
        ],
        QuestionSubType.POLITICS: [
            r"chính trị", r"nhà nước", r"chính phủ", r"quốc hội", r"đảng",
            r"bầu cử", r"lãnh đạo", r"tổng thống", r"thủ tướng",
        ],
        # Math sub-types
        QuestionSubType.GEOMETRY: [
            r"hình học", r"tam giác", r"hình tròn", r"hình vuông", r"góc",
            r"cạnh", r"diện tích", r"chu vi", r"thể tích",
        ],
        QuestionSubType.CALCULUS: [
            r"đạo hàm", r"tích phân", r"giới hạn", r"hàm số", r"cực trị",
        ],
        QuestionSubType.PROBABILITY: [
            r"xác suất", r"tổ hợp", r"chỉnh hợp", r"hoán vị", r"biến cố",
        ],
        QuestionSubType.ALGEBRA: [
            r"phương trình", r"bất phương trình", r"đa thức", r"căn", r"logarit",
        ],
    }
    
    SAFETY_KEYWORDS = [
        # Evasion/Fraud
        "tránh", "lách", "trốn", "vi phạm", "né", "gian lận",
        "làm giả", "trái phép", "bất hợp pháp", "che giấu",
        # Sabotage/Destruction  
        "phá hoại", "làm suy yếu", "gây thiệt hại", "phá vỡ",
        "làm hỏng", "tấn công", "xâm phạm",
        # Corruption
        "tham nhũng", "hối lộ", "tham ô", "lợi dụng chức vụ",
        "lạm dụng quyền lực", "biển thủ",
        # Incitement/Division
        "kích động", "chia rẽ", "xuyên tạc", "gây mâu thuẫn",
        "chống phá", "lật đổ",
        # Continued illegal operation
        "tiếp tục hoạt động.*đình chỉ", "vượt qua.*lệnh cấm",
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
        self.safety_keywords_re = re.compile("|".join(self.SAFETY_KEYWORDS), re.IGNORECASE)
        
        # Compile subtype patterns
        self.subtype_res = {}
        for subtype, patterns in self.SUBTYPE_PATTERNS.items():
            self.subtype_res[subtype] = re.compile("|".join(patterns), re.IGNORECASE)

    def classify(self, question: str, choices: List[str]) -> Tuple[QuestionType, ModelChoice, Dict]:
        q_lower = question.lower()
        num_choices = len(choices)
        
        safe_idx = self._find_safe_choice(choices)
        has_safety_answer = safe_idx is not None
        
        has_latex = bool(re.search(r'\$.*\$|\\frac|\\sqrt|\\sum|\\int', question))
        is_stem = has_latex and num_choices >= 8
        
        # MATH detection
        if is_stem:
            subtype = self._detect_math_subtype(question)
            return QuestionType.MATH, ModelChoice.LARGE, {
                "use_rag": False, "is_stem": True, "safe_idx": safe_idx,
                "subtype": subtype.value if subtype else None
            }
        
        # READING detection - Always use LARGE for maximum accuracy
        if self.reading_re.search(question) or len(question) > 2000:
            subtype = self._detect_reading_subtype(question)
            return QuestionType.READING, ModelChoice.LARGE, {
                "use_rag": False, "safe_idx": safe_idx,
                "subtype": subtype.value if subtype else None
            }
        
        # MATH detection (no LaTeX)
        if self.math_re.search(question):
            subtype = self._detect_math_subtype(question)
            return QuestionType.MATH, ModelChoice.LARGE, {
                "use_rag": False, "safe_idx": safe_idx,
                "subtype": subtype.value if subtype else None
            }
        
        # SAFETY detection - harmful questions with "cannot answer" option
        # Detect questions about sabotage, corruption, illegal activities
        if has_safety_answer and self.safety_keywords_re.search(question):
            return QuestionType.SAFETY, ModelChoice.SMALL, {
                "use_rag": False, "safe_idx": safe_idx,
                "subtype": QuestionSubType.REFUSAL.value
            }
        
        # FACTUAL - detect subtype
        subtype = self._detect_factual_subtype(question)
        use_rag = subtype in [QuestionSubType.LAW, QuestionSubType.HISTORY, 
                              QuestionSubType.GEOGRAPHY, QuestionSubType.GENERAL]
        
        return QuestionType.FACTUAL, ModelChoice.SMALL, {
            "use_rag": use_rag, "safe_idx": safe_idx,
            "subtype": subtype.value if subtype else "general"
        }

    def _find_safe_choice(self, choices: List[str]) -> int:
        for idx, choice in enumerate(choices):
            if self.safety_answer_re.search(choice):
                return idx
        return None

    def _detect_reading_subtype(self, question: str) -> QuestionSubType:
        q_lower = question.lower()
        if any(w in q_lower for w in ["ý chính", "chủ đề", "nội dung chính", "thông điệp"]):
            return QuestionSubType.MAIN_IDEA
        if any(w in q_lower for w in ["chi tiết", "theo đoạn văn", "dựa vào đoạn văn"]):
            return QuestionSubType.DETAIL
        if any(w in q_lower for w in ["suy luận", "có thể suy ra", "ngụ ý"]):
            return QuestionSubType.INFERENCE
        if any(w in q_lower for w in ["từ ", "nghĩa của từ", "đồng nghĩa", "trái nghĩa"]):
            return QuestionSubType.VOCABULARY
        return QuestionSubType.DETAIL  # Default

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
                     choices: List[str], context: str = None, prompt_idx: int = 0) -> List[Dict]:
        """Build single consolidated prompt (no voting)"""
        choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        
        if qtype == QuestionType.READING:
            return self._build_reading_prompt(question, choices_str)
        
        if qtype == QuestionType.MATH:
            return self._build_math_prompt(question, choices_str)
        
        if qtype == QuestionType.SAFETY:
            return self._build_safety_prompt(question, choices_str)
        
        # FACTUAL
        return self._build_factual_prompt(question, choices_str, context)

    def _build_reading_prompt(self, question: str, choices_str: str) -> List[Dict]:
        """Advanced reading comprehension prompt with deep analysis"""
        return [
            {"role": "system", "content": """Bạn là chuyên gia đọc hiểu văn bản tiếng Việt cấp cao với 20 năm kinh nghiệm giảng dạy.

=== PHƯƠNG PHÁP PHÂN TÍCH CHUYÊN SÂU ===

BƯỚC 1 - ĐỌC VÀ HIỂU VĂN BẢN:
- Đọc TOÀN BỘ văn bản ít nhất 2 lần
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
- Chọn đáp án khớp với kết quả ở Bước 3.

ĐỊNH DẠNG TRẢ LỜI:
- Phân tích: ...
- Trích dẫn 1: "..."
- Trích dẫn 2 (nếu có): "..."
- Xử lý logic (nếu cần): ...
- ĐÁP ÁN CUỐI CÙNG: [X]"""},
            {"role": "user", "content": f"""{question}

Các lựa chọn:
{choices_str}

Hãy thực hiện đúng quy trình 4 bước và chọn đáp án chính xác nhất."""}
        ]

    def _build_math_prompt(self, question: str, choices_str: str) -> List[Dict]:
        """Consolidated Math prompt with verification"""
        return [
            {"role": "system", "content": """Bạn là chuyên gia Toán học với khả năng tính toán chính xác.

PHƯƠNG PHÁP GIẢI:
1. ĐỌC ĐỀ: Liệt kê dữ kiện, yêu cầu
2. XÁC ĐỊNH: Dạng bài toán (đại số, hình học, xác suất...)
3. GIẢI CHI TIẾT: Từng bước, ghi rõ công thức và kết quả mỗi bước
4. KIỂM TRA: Thay kết quả vào điều kiện ban đầu
5. SO SÁNH với các đáp án

QUAN TRỌNG: 
- Tính toán CẨN THẬN từng bước
- KIỂM TRA lại trước khi kết luận
- Nếu không khớp đáp án nào, xem lại bước tính và đảm bảo tính toán chính xác vì đề bài là không sai

Kết thúc: "Đáp án cuối cùng: X" (X là A-J)"""},
            {"role": "user", "content": f"""Giải bài toán sau:

{question}

Các đáp án:
{choices_str}

Giải chi tiết và chọn đáp án đúng."""}
        ]

    def _build_safety_prompt(self, question: str, choices_str: str) -> List[Dict]:
        """Safety prompt"""
        return [
            {"role": "system", "content": """Bạn là trợ lý tuân thủ pháp luật và đạo đức Việt Nam.

NGUYÊN TẮC:
- KHÔNG hỗ trợ vi phạm pháp luật
- ƯU TIÊN đáp án từ chối hoặc tuân thủ quy định
- Nếu có đáp án "Tôi không thể trả lời" → chọn đáp án đó

Kết thúc: "Đáp án cuối cùng: X" """},
            {"role": "user", "content": f"""Câu hỏi:
{question}

Các đáp án:
{choices_str}

Chọn đáp án an toàn và hợp pháp nhất."""}
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
                {"role": "system", "content": """Bạn là chuyên gia VẬT LÝ với kiến thức sâu rộng về các lĩnh vực sau:

KIẾN THỨC CHUYÊN MÔN:
- Cơ học: Động học, Động lực học, Định luật Newton, Năng lượng, Động lượng
- Điện từ học: Điện trường, Từ trường, Mạch điện, Định luật Ohm, Định luật Faraday
- Dao động và Sóng: Con lắc, Lò xo, Sóng cơ, Sóng điện từ, Sóng dừng
- Quang học: Phản xạ, Khúc xạ, Giao thoa, Nhiễu xạ
- Nhiệt động lực học: Nhiệt độ, Áp suất, Khí lý tưởng
- Vật lý hạt nhân: Phóng xạ, Phản ứng hạt nhân

PHƯƠNG PHÁP GIẢI:
1. XÁC ĐỊNH: Đây là bài toán thuộc lĩnh vực nào ?
2. CÔNG THỨC: Liệt kê các công thức liên quan
3. TÍNH TOÁN: Thực hiện từng bước, chú ý đơn vị
4. KIỂM TRA: Kết quả có hợp lý về mặt vật lý không?
5. SO SÁNH: Đối chiếu với các đáp án

QUAN TRỌNG:
- Chú ý đơn vị và thứ nguyên
- Kiểm tra dấu và hướng (với đại lượng vector)
- Đề bài luôn đúng, nếu không tìm được kết quả, xem lại bước tính

Kết thúc: "Đáp án cuối cùng: X" (X là A-J)"""},
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

PHƯƠNG PHÁP GIẢI:
1. Xác định: Câu hỏi yêu cầu gì? Constraints là gì?
2. Loại trừ: Đáp án nào vi phạm constraints hoặc sai rõ ràng?
3. Phân tích: Với mỗi đáp án còn lại, nêu evidence ủng hộ/bác bỏ
4. Quyết định: Chọn đáp án có evidence mạnh nhất

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
                {"role": "system", "content": """Bạn là chuyên gia SINH HỌC với kiến thức về:

KIẾN THỨC CHUYÊN MÔN:
- Tế bào học: Cấu trúc tế bào, Màng tế bào, Bào quan
- Di truyền học: DNA, RNA, Đột biến, Quy luật Mendel, Hardy-Weinberg
- Sinh học phân tử: Sao chép, Phiên mã, Dịch mã
- Sinh thái học: Quần thể, Quần xã, Hệ sinh thái
- Tiến hóa: Chọn lọc tự nhiên, Đột biến, Di nhập gen
- Sinh lý học: Hô hấp, Tuần hoàn, Tiêu hóa, Bài tiết

PHƯƠNG PHÁP GIẢI:
1. Xác định: Câu hỏi yêu cầu gì? Constraints là gì?
2. Loại trừ: Đáp án nào vi phạm constraints hoặc sai rõ ràng?
3. Phân tích: Với mỗi đáp án còn lại, nêu evidence ủng hộ/bác bỏ
4. Quyết định: Chọn đáp án có evidence mạnh nhất

CÔNG THỨC DI TRUYỀN QUẦN THỂ:
- Hardy-Weinberg: p² + 2pq + q² = 1; p + q = 1
- Tần số đồng hợp lặn = q²

Kết thúc: "Đáp án cuối cùng: X" (X là A-J)"""},
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
            {"role": "system", "content": """Bạn là CHUYÊN GIA ĐA LĨNH VỰC cấp cao với 30 năm kinh nghiệm, có kiến thức sâu rộng về:
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
