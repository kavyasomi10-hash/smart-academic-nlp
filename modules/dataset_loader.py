import fitz  # This is PyMuPDF

def extract_text_from_pdf(pdf_path):
    print(f"Opening {pdf_path}...")
    
    # Open the PDF document
    doc = fitz.open(pdf_path)
    full_text = ""
    
    # Loop through every page and extract the text
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        full_text += text + "\n"
        
    print(f"Successfully extracted {len(doc)} pages of text.")
    return full_text

# You can test it by adding this at the bottom:
if __name__ == "__main__":
    # Make sure to change this to the actual name of your PDF file
    sample_pdf = "../data/Class8_Science_Chapter1.pdf" 
    
    try:
        extracted = extract_text_from_pdf(sample_pdf)
        print("\n--- Preview of First 500 Characters ---")
        print(extracted[:500])
    except Exception as e:
        print(f"Error loading PDF: {e}")