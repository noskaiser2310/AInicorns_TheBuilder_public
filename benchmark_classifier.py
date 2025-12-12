import json
import re
import pandas as pd
from collections import Counter
from enum import Enum


class BenchmarkCategory(Enum):
    RAG = "RAG"           # Reading comprehension
    STEM = "STEM"         # Math, Physics, Chemistry, Biology
    COMPULSORY = "Compulsory"  # Vietnam-related: Politics, History, Law, HCM thought
    MULTI_DOMAIN = "Multi-Domain"  # Other multi-domain questions


class BenchmarkClassifier:
    """Classifier for benchmark categories"""
    
    # RAG patterns - Reading comprehension with context
    RAG_PATTERNS = [
        r"Đoạn thông tin:",
        r"Văn bản:",
        r"Bài viết:",
        r"Đoạn văn sau:",
        r"\[1\] Tiêu đề:",
        r"-- Đoạn văn \d+ --",
        r"-- Document \d+ --",
        r"Title:",
        r"Content:",
        r"Câu hỏi:",
    ]
    
    # STEM patterns - Math, Physics, Chemistry, Biology
    STEM_PATTERNS = {
        "math": [
            r"\$.*\$",
            r"\\frac", r"\\sqrt", r"\\sum", r"\\int",
            r"phương trình", r"tính toán", r"giải bài toán",
            r"đạo hàm", r"tích phân", r"xác suất",
            r"tính giá trị", r"biểu thức", r"hàm số",
            r"vector", r"ma trận", r"định thức",
            r"logarit", r"căn bậc",
        ],
        "physics": [
            r"vật lý", r"cơ học", r"động lực học",
            r"điện trở", r"điện áp", r"dòng điện", r"từ trường",
            r"sóng", r"dao động", r"con lắc", r"lò xo", r"tần số",
            r"quang học", r"khúc xạ", r"phản xạ", r"thấu kính",
            r"nhiệt độ", r"áp suất", r"thể tích",
            r"gia tốc", r"vận tốc", r"lực", r"năng lượng",
            r"\\Omega", r"\\mu", r"\\epsilon", r"\\lambda",
            r"hạt nhân", r"phóng xạ", r"electron", r"proton",
            r"trường hấp dẫn", r"quỹ đạo",
        ],
        "chemistry": [
            r"hóa học", r"phản ứng", r"hợp chất", r"nguyên tố",
            r"axit", r"bazơ", r"muối", r"oxi hóa", r"khử",
            r"phân tử", r"nguyên tử", r"ion", r"liên kết",
            r"mol", r"nồng độ", r"dung dịch", r"kết tủa",
            r"chiral", r"đồng phân", r"aldehyt", r"cinnamic",
        ],
        "biology": [
            r"sinh học", r"tế bào", r"gen", r"dna", r"rna",
            r"quần thể", r"di truyền", r"đột biến", r"nhiễm sắc thể",
            r"Hardy-Weinberg", r"alen", r"kiểu gen", r"kiểu hình",
            r"enzyme", r"protein", r"axit amin",
            r"hô hấp", r"quang hợp",
        ],
    }
    
    # Compulsory patterns - Vietnam-related questions
    COMPULSORY_PATTERNS = {
        "politics": [
            r"Chủ Tịch Hồ Chí Minh", r"Hồ Chí Minh", r"Chủ nghĩa Mác",
            r"chính trị", r"nhà nước", r"chính phủ", r"quốc hội",
            r"đảng", r"bầu cử", r"lãnh đạo", r"Tổng Bí thư",
            r"Thủ tướng", r"xã hội chủ nghĩa", r"cách mạng",
            r"giành quyền lực", r"chuyển giao quyền lực",
            r"tư tưởng Hồ Chí Minh", r"Nguyễn Ái Quốc",
        ],
        "history_vn": [
            r"lịch sử Việt Nam", r"triều đại", r"kháng chiến",
            r"Trần Nhân Tông", r"nhà Trần", r"nhà Nguyễn", r"nhà Lê",
            r"Đại Việt", r"Bắc thuộc", r"Pháp thuộc",
            r"cuộc khởi nghĩa", r"quân Nguyên", r"Mông Cổ xâm lược",
        ],
        "law_vn": [
            r"Luật.*Việt Nam", r"luật pháp Việt Nam",
            r"Bộ luật Hình sự", r"Bộ luật Dân sự",
            r"Nghị định", r"Thông tư", r"điều \d+",
            r"pháp luật", r"vi phạm hành chính",
            r"Luật Bảo vệ môi trường", r"Luật.*\d{4}",
            r"thẩm quyền", r"xử phạt",
        ],
        "culture_vn": [
            r"văn hóa Việt Nam", r"truyền thống Việt Nam",
            r"lễ hội Việt Nam", r"phong tục Việt Nam",
            r"tiếng Việt", r"tục ngữ", r"ca dao",
        ],
        "geography_vn": [
            r"Việt Nam", r"Hà Nội", r"TP\.? HCM", r"Thành phố Hồ Chí Minh",
            r"tỉnh.*Việt Nam", r"miền Bắc|miền Trung|miền Nam",
        ],
    }
    
    # Safety patterns
    SAFETY_PATTERNS = [
        r"tránh", r"lách", r"trốn", r"vi phạm",
        r"Tôi không thể chia sẻ", r"Tôi không thể trả lời",
    ]
    
    def __init__(self):
        self.rag_re = re.compile("|".join(self.RAG_PATTERNS), re.IGNORECASE)
        self.safety_re = re.compile("|".join(self.SAFETY_PATTERNS), re.IGNORECASE)
        
        # Compile STEM patterns
        self.stem_res = {}
        for category, patterns in self.STEM_PATTERNS.items():
            self.stem_res[category] = re.compile("|".join(patterns), re.IGNORECASE)
        
        # Compile Compulsory patterns
        self.compulsory_res = {}
        for category, patterns in self.COMPULSORY_PATTERNS.items():
            self.compulsory_res[category] = re.compile("|".join(patterns), re.IGNORECASE)
    
    def classify(self, question: str, choices: list) -> tuple:
        """
        Classify question into benchmark category
        Returns: (BenchmarkCategory, sub_category, details)
        """
        q_lower = question.lower()
        num_choices = len(choices)
        
        # Check for RAG (Reading comprehension with context)
        if self.rag_re.search(question) and len(question) > 500:
            sub_category = self._detect_rag_subcategory(question)
            return BenchmarkCategory.RAG, sub_category, {"has_context": True}
        
        # Check for STEM with LaTeX (high confidence)
        has_latex = bool(re.search(r'\$.*\$|\\frac|\\sqrt|\\sum|\\int', question))
        if has_latex and num_choices >= 8:
            sub_category = self._detect_stem_subcategory(question)
            return BenchmarkCategory.STEM, sub_category, {"has_latex": True, "num_choices": num_choices}
        
        # Check for STEM without LaTeX
        for category, regex in self.stem_res.items():
            if regex.search(question):
                return BenchmarkCategory.STEM, category, {"has_latex": has_latex}
        
        # Check for Compulsory (Vietnam-related)
        for category, regex in self.compulsory_res.items():
            if regex.search(question):
                return BenchmarkCategory.COMPULSORY, category, {}
        
        # Default to Multi-Domain
        sub_category = self._detect_multidomain_subcategory(question)
        return BenchmarkCategory.MULTI_DOMAIN, sub_category, {}
    
    def _detect_rag_subcategory(self, question: str) -> str:
        """Detect RAG subcategory"""
        q_lower = question.lower()
        if any(w in q_lower for w in ["ý chính", "chủ đề", "nội dung chính"]):
            return "main_idea"
        if any(w in q_lower for w in ["chi tiết", "theo đoạn văn", "dựa vào"]):
            return "detail"
        if any(w in q_lower for w in ["suy luận", "có thể suy ra", "ngụ ý"]):
            return "inference"
        return "reading"
    
    def _detect_stem_subcategory(self, question: str) -> str:
        """Detect STEM subcategory"""
        for category, regex in self.stem_res.items():
            if regex.search(question):
                return category
        return "math"  # Default
    
    def _detect_multidomain_subcategory(self, question: str) -> str:
        """Detect Multi-Domain subcategory"""
        q_lower = question.lower()
        if any(w in q_lower for w in ["kinh tế", "tài chính", "thương mại", "gdp"]):
            return "economics"
        if any(w in q_lower for w in ["tâm lý", "xã hội", "hành vi"]):
            return "psychology"
        if any(w in q_lower for w in ["lịch sử", "thế kỷ", "triều đại"]):
            return "history"
        if any(w in q_lower for w in ["địa lý", "quốc gia", "thành phố"]):
            return "geography"
        if any(w in q_lower for w in ["văn hóa", "nghệ thuật", "âm nhạc"]):
            return "culture"
        return "general"


def analyze_questions(test_file: str, output_dir: str = "D:/VNPT_Hackathon/AInicorns_TheBuilder_public/sub"):
    """Analyze and classify all questions"""
    
    # Load test data
    with open(test_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    classifier = BenchmarkClassifier()
    
    results = []
    for q in questions:
        qid = q['qid']
        question = q['question']
        choices = q['choices']
        
        category, sub_category, details = classifier.classify(question, choices)
        
        results.append({
            'qid': qid,
            'category': category.value,
            'sub_category': sub_category,
            'num_choices': len(choices),
            'question_length': len(question),
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print("=" * 60)
    print("BENCHMARK CATEGORY ANALYSIS")
    print("=" * 60)
    print(f"\nTotal questions: {len(df)}")
    print()
    
    print("=== Category Distribution ===")
    category_counts = df['category'].value_counts()
    for cat, count in category_counts.items():
        pct = count / len(df) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")
    
    print()
    print("=== Sub-category Distribution ===")
    for category in df['category'].unique():
        print(f"\n{category}:")
        sub_counts = df[df['category'] == category]['sub_category'].value_counts()
        for sub, count in sub_counts.items():
            print(f"  - {sub}: {count}")
    
    # Save results
    output_file = f"{output_dir}/benchmark_classification.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Create summary file
    # Convert sub_categories with proper key formatting
    sub_cat_dict = {}
    for (cat, sub), count in df.groupby('category')['sub_category'].value_counts().items():
        if cat not in sub_cat_dict:
            sub_cat_dict[cat] = {}
        sub_cat_dict[cat][sub] = count
    
    summary = {
        'total_questions': len(df),
        'categories': category_counts.to_dict(),
        'sub_categories': sub_cat_dict
    }
    
    summary_file = f"{output_dir}/benchmark_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to: {summary_file}")
    
    return df


if __name__ == "__main__":
    test_file = "D:/VNPT_Hackathon/AInicorns_TheBuilder_public/data/test.json"
    df = analyze_questions(test_file)
    
    # Show sample questions from each category
    print("\n" + "=" * 60)
    print("SAMPLE QUESTIONS BY CATEGORY")
    print("=" * 60)
    
    with open(test_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    questions_dict = {q['qid']: q['question'][:200] + "..." if len(q['question']) > 200 else q['question'] for q in questions}
    
    for category in df['category'].unique():
        print(f"\n=== {category} ===")
        sample = df[df['category'] == category].head(3)
        for _, row in sample.iterrows():
            print(f"  [{row['qid']}] ({row['sub_category']})")
            print(f"    {questions_dict[row['qid']][:100]}...")
