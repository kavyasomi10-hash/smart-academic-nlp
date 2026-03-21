# modules/dataset_loader.py
# YOUR OWN CODE — Dataset Loader for NCERT Class 8 Science
# Extracts all 13 chapters from the textbook PDF
# and saves them as a structured JSON dataset

import fitz       # PyMuPDF — for reading PDF
import re         # Regular expressions — for text cleaning
import json       # For saving structured dataset
import os         # For file path handling

class NCERTDatasetLoader:
    """
    Loads NCERT Class 8 Science textbook PDF and extracts
    each chapter as clean, structured text ready for
    simplification and summarization pipeline.
    """

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = None
        self.chapters = {}

        # Known chapter structure of this specific textbook
        # Detected from PDF analysis
        self.chapter_map = {
            1:  "Exploring the Investigative World of Science",
            2:  "The Invisible Living World: Beyond Our Naked Eye",
            3:  "Health: The Ultimate Treasure",
            4:  "Electricity: Magnetic and Heating Effects",
            5:  "Exploring Forces",
            6:  "Pressure, Winds, Storms, and Cyclones",
            7:  "Particulate Nature of Matter",
            8:  "Nature of Matter: Elements, Compounds, and Mixtures",
            9:  "The Amazing World of Solutes, Solvents, and Solutions",
            10: "Light: Mirrors and Lenses",
            11: "Keeping Time with the Skies",
            12: "How Nature Works in Harmony",
            13: "Our Home: Earth, a Unique Life Sustaining Planet",
        }

    # ── Step 1: Open PDF ──────────────────────────────────────────────────────
    def load_pdf(self):
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        self.doc = fitz.open(self.pdf_path)
        print(f"PDF loaded: {len(self.doc)} pages")
        return self

    # ── Step 2: Find chapter start pages ─────────────────────────────────────
    def find_chapter_pages(self):
        """
        YOUR OWN LOGIC:
        Scans every page looking for 'Chapter N — Title' pattern.
        Records the first occurrence of each chapter number.
        """
        chapter_starts = {}

        for page_num in range(len(self.doc)):
            text = self.doc[page_num].get_text()
            lines = [l.strip() for l in text.split('\n') if l.strip()]

            for line in lines[:5]:  # Check first 5 lines of each page
                match = re.match(r'Chapter\s+(\d+)\s*[—\-]\s*(.+)', line)
                if match:
                    ch_num = int(match.group(1))
                    if ch_num not in chapter_starts:
                        chapter_starts[ch_num] = page_num
                    break

        print(f"Found {len(chapter_starts)} chapters in PDF")
        return chapter_starts

    # ── Step 3: Extract raw text for each chapter ─────────────────────────────
    def extract_chapter_text(self, start_page, end_page):
        """
        YOUR OWN LOGIC:
        Extracts and joins text from all pages of a chapter.
        """
        raw_text = ""
        for page_num in range(start_page, end_page):
            raw_text += self.doc[page_num].get_text()
        return raw_text

    # ── Step 4: Clean extracted text ──────────────────────────────────────────
    def clean_text(self, text):
        """
        YOUR OWN PREPROCESSING LOGIC:
        Cleans NCERT-specific formatting issues.
        """
        # Fix words split across lines (single word on its own line)
        lines = text.split('\n')
        fixed_lines = []
        buffer = ""

        for line in lines:
            line = line.strip()
            if not line:
                if buffer:
                    fixed_lines.append(buffer)
                    buffer = ""
                fixed_lines.append("")
                continue

            # If line is a single word (likely broken from previous line)
            if len(line.split()) == 1 and buffer:
                buffer += " " + line
            else:
                if buffer:
                    fixed_lines.append(buffer)
                buffer = line

        if buffer:
            fixed_lines.append(buffer)

        text = ' '.join(fixed_lines)

        # Remove page numbers (standalone numbers)
        text = re.sub(r'\s+\d{1,3}\s+', ' ', text)

        # Remove chapter header repetitions
        text = re.sub(r'Chapter\s+\d+\s*[—\-]\s*[A-Za-z ,:\-]+', ' ', text)

        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E\n]', ' ', text)

        # Remove activity/exercise markers
        text = re.sub(r'Activity\s*\d*', '', text)
        text = re.sub(r'Exercise\s*\d*', '', text)
        text = re.sub(r'Fig\.\s*\d+\.\d+', '', text)
        text = re.sub(r'Table\s*\d+\.\d+', '', text)

        # Fix spacing after punctuation
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        text = re.sub(r',([A-Za-z])', r', \1', text)

        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        text = text.strip()

        return text

    # ── Step 5: Split into paragraphs ─────────────────────────────────────────
    def extract_paragraphs(self, text, min_words=40):
        """
        YOUR OWN LOGIC:
        Splits chapter text into meaningful paragraphs.
        Filters out short fragments, captions, and headers.
        """
        # Split on double space or sentence boundaries
        raw_paragraphs = re.split(r'\s{2,}|(?<=[.!?])\s+(?=[A-Z])', text)

        clean_paragraphs = []
        for para in raw_paragraphs:
            para = para.strip()
            word_count = len(para.split())

            # Filter: must have enough words
            if word_count < min_words:
                continue

            # Filter: must contain at least one verb (basic sentence check)
            if not re.search(r'\b(is|are|was|were|has|have|can|will|do|does|'
                            r'shows|helps|causes|makes|allows|enables|forms|'
                            r'contains|produces|provides|occurs|results)\b',
                            para, re.IGNORECASE):
                continue

            # Filter: remove if too many numbers (likely a table/data)
            numbers = len(re.findall(r'\d+', para))
            if numbers > word_count * 0.3:
                continue

            clean_paragraphs.append(para)

        return clean_paragraphs

    # ── Step 6: Full pipeline ─────────────────────────────────────────────────
    def build_dataset(self, output_path="data/ncert_dataset.json"):
        """
        YOUR OWN LOGIC:
        Runs the full extraction pipeline for all chapters.
        Saves structured JSON dataset.
        """
        self.load_pdf()
        chapter_pages = self.find_chapter_pages()

        dataset = {
            "source": "NCERT Class 8 Science Textbook",
            "total_chapters": len(chapter_pages),
            "chapters": []
        }

        chapter_nums = sorted(chapter_pages.keys())

        for i, ch_num in enumerate(chapter_nums):
            start_page = chapter_pages[ch_num]

            # End page = start of next chapter or end of doc
            if i + 1 < len(chapter_nums):
                end_page = chapter_pages[chapter_nums[i + 1]]
            else:
                end_page = len(self.doc)

            print(f"\nProcessing Chapter {ch_num}: "
                  f"'{self.chapter_map.get(ch_num, 'Unknown')}' "
                  f"(pages {start_page+1}-{end_page})")

            # Extract and clean text
            raw_text    = self.extract_chapter_text(start_page, end_page)
            clean       = self.clean_text(raw_text)
            paragraphs  = self.extract_paragraphs(clean)

            print(f"  Extracted {len(paragraphs)} paragraphs "
                  f"({len(clean.split())} total words)")

            chapter_data = {
                "chapter_number": ch_num,
                "chapter_title":  self.chapter_map.get(ch_num, f"Chapter {ch_num}"),
                "start_page":     start_page + 1,
                "end_page":       end_page,
                "total_words":    len(clean.split()),
                "total_paragraphs": len(paragraphs),
                "full_text":      clean,
                "paragraphs":     paragraphs
            }

            dataset["chapters"].append(chapter_data)

        # Add dataset statistics
        total_words = sum(c["total_words"] for c in dataset["chapters"])
        total_paras = sum(c["total_paragraphs"] for c in dataset["chapters"])
        dataset["total_words"]      = total_words
        dataset["total_paragraphs"] = total_paras

        # Save to JSON
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*55}")
        print(f"DATASET BUILT SUCCESSFULLY")
        print(f"{'='*55}")
        print(f"Total chapters  : {dataset['total_chapters']}")
        print(f"Total paragraphs: {total_paras}")
        print(f"Total words     : {total_words}")
        print(f"Saved to        : {output_path}")
        print(f"{'='*55}")

        self.doc.close()
        return dataset


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = NCERTDatasetLoader("data/ncert_class8_science.pdf")
    dataset = loader.build_dataset("data/ncert_dataset.json")

    # Show sample from Chapter 1
    ch1 = dataset["chapters"][0]
    print(f"\nSAMPLE FROM CHAPTER 1:")
    print(f"Title     : {ch1['chapter_title']}")
    print(f"Paragraphs: {ch1['total_paragraphs']}")
    print(f"Words     : {ch1['total_words']}")
    if ch1["paragraphs"]:
        print(f"\nFirst paragraph:")
        print(ch1["paragraphs"][0][:300] + "...")
