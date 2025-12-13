"""
PDF to TXT Converter
Chuyển đổi các file PDF sang TXT sử dụng pdfplumber
Sử dụng: conda activate rag && python convert_pdf_to_txt.py
"""

import os
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("Installing pdfplumber...")
    os.system("pip install pdfplumber")
    import pdfplumber


def convert_pdf_to_txt(pdf_path: Path, output_dir: Path):
    """Convert a single PDF to TXT"""
    try:
        txt_filename = pdf_path.stem + ".txt"
        txt_path = output_dir / txt_filename
        
        # Skip if already converted
        if txt_path.exists():
            print(f"[SKIP] {pdf_path.name} - already converted")
            return True
        
        print(f"[CONVERTING] {pdf_path.name}...")
        
        text_content = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(f"--- Page {page_num} ---\n{page_text}")
        
        if text_content:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(text_content))
            print(f"[OK] {txt_path.name} ({len(text_content)} pages)")
            return True
        else:
            print(f"[WARN] {pdf_path.name} - no text extracted")
            return False
            
    except Exception as e:
        print(f"[ERROR] {pdf_path.name}: {e}")
        return False


def process_directory(pdf_dir: Path, output_dir: Path):
    """Process all PDFs in a directory"""
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing {len(pdf_files)} PDFs from {pdf_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    success = 0
    for pdf_file in pdf_files:
        if convert_pdf_to_txt(pdf_file, output_dir):
            success += 1
    
    print(f"\nCompleted: {success}/{len(pdf_files)} files converted")


if __name__ == "__main__":
    base_dir = Path("data/crawl")
    
    # Directory 1: data_congbao
    congbao_pdf = base_dir / "data_congbao" / "pdf"
    congbao_txt = base_dir / "data_congbao" / "txt"
    
    # Directory 2: data_nghi_quyet  
    nghi_quyet_pdf = base_dir / "data_nghi_quyet" / "pdf"
    nghi_quyet_txt = base_dir / "data_nghi_quyet" / "txt"
    
    # Process both directories
    if congbao_pdf.exists():
        process_directory(congbao_pdf, congbao_txt)
    else:
        print(f"Directory not found: {congbao_pdf}")
    
    if nghi_quyet_pdf.exists():
        process_directory(nghi_quyet_pdf, nghi_quyet_txt)
    else:
        print(f"Directory not found: {nghi_quyet_pdf}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
