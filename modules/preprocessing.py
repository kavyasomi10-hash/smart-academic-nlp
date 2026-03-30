# modules/preprocessing.py
# YOUR OWN CODE — Text Preprocessing Pipeline
# Cleans raw academic text before passing to models

import re
import unicodedata


class TextPreprocessor:
    """
    Preprocessing pipeline for academic text.
    Handles NCERT-specific formatting issues from PDF extraction.
    All logic written from scratch — no NLP library used.
    """

    def __init__(self):
        print("Text Preprocessor ready.")

    # ── YOUR OWN METHOD 1 ────────────────────────────────────────────────────
    def remove_non_printable(self, text):
        """
        Removes non-printable characters like U+00A0 (non-breaking space)
        that appear when extracting text from PDFs.
        Uses Unicode normalisation to standardise characters.
        """
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
        return text

    # ── YOUR OWN METHOD 2 ────────────────────────────────────────────────────
    def remove_page_numbers(self, text):
        """
        Removes standalone page numbers that appear in NCERT PDF extraction.
        Pattern: a number alone on a line (e.g. "42" between paragraphs).
        """
        text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)
        text = re.sub(r'^\d{1,3}\s*$', '', text, flags=re.MULTILINE)
        return text

    # ── YOUR OWN METHOD 3 ────────────────────────────────────────────────────
    def fix_broken_words(self, text):
        """
        Fixes words broken across lines in PDF extraction.
        Example: "photo-\nsynthesis" becomes "photosynthesis"
        Also joins single words that appear alone on a line.
        """
        text = re.sub(r'-\n(\w)', r'\1', text)
        lines = text.split('\n')
        fixed = []
        buffer = ""
        for line in lines:
            line = line.strip()
            if not line:
                if buffer:
                    fixed.append(buffer)
                    buffer = ""
                fixed.append("")
                continue
            if len(line.split()) == 1 and buffer:
                buffer += " " + line
            else:
                if buffer:
                    fixed.append(buffer)
                buffer = line
        if buffer:
            fixed.append(buffer)
        return ' '.join(fixed)

    # ── YOUR OWN METHOD 4 ────────────────────────────────────────────────────
    def remove_ncert_artifacts(self, text):
        """
        Removes NCERT textbook specific content that is not useful
        for simplification or summarization:
        - Activity labels
        - Figure captions
        - Exercise headings
        - Table labels
        """
        text = re.sub(r'Activity\s*\d*', '', text)
        text = re.sub(r'Exercise\s*\d*', '', text)
        text = re.sub(r'Fig\.\s*\d+[\.\d]*', '', text)
        text = re.sub(r'Table\s*\d+[\.\d]*', '', text)
        text = re.sub(r'Box\s*\d+[\.\d]*', '', text)
        text = re.sub(r'Do\s+You\s+Know\??', '', text)
        text = re.sub(r'Think\s+and\s+Discuss', '', text)
        return text

    # ── YOUR OWN METHOD 5 ────────────────────────────────────────────────────
    def fix_punctuation(self, text):
        """
        Fixes punctuation spacing issues common in PDF text extraction.
        - Adds space after period if missing before capital letter
        - Fixes multiple consecutive periods
        - Fixes space before punctuation
        """
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        text = re.sub(r',([A-Za-z])', r', \1', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text

    # ── YOUR OWN METHOD 6 ────────────────────────────────────────────────────
    def remove_extra_whitespace(self, text):
        """
        Removes all extra whitespace and normalises to single spaces.
        """
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        return text.strip()

    # ── YOUR OWN METHOD 7 ────────────────────────────────────────────────────
    def validate_input(self, text):
        """
        Validates that input text meets minimum requirements
        for meaningful simplification and summarization.
        Returns (is_valid, message).
        """
        word_count = len(text.split())
        if word_count < 30:
            return False, f"Too short: {word_count} words. Need at least 30."
        if word_count > 2000:
            return False, f"Too long: {word_count} words. Max 2000."
        if not re.search(r'[.!?]', text):
            return False, "No sentences detected. Add proper punctuation."
        return True, f"Valid input: {word_count} words."

    # ── FULL PIPELINE ────────────────────────────────────────────────────────
    def preprocess(self, text):
        """
        Runs all preprocessing steps in correct order.
        Returns clean text ready for simplification/summarization.
        """
        print("  Preprocessing: Removing non-printable characters...")
        text = self.remove_non_printable(text)

        print("  Preprocessing: Removing page numbers...")
        text = self.remove_page_numbers(text)

        print("  Preprocessing: Fixing broken words...")
        text = self.fix_broken_words(text)

        print("  Preprocessing: Removing NCERT artifacts...")
        text = self.remove_ncert_artifacts(text)

        print("  Preprocessing: Fixing punctuation...")
        text = self.fix_punctuation(text)

        print("  Preprocessing: Cleaning whitespace...")
        text = self.remove_extra_whitespace(text)

        valid, message = self.validate_input(text)
        print(f"  Validation: {message}")

        return text


if __name__ == "__main__":
    preprocessor = TextPreprocessor()

    test_text = """
        Combustion  is a chemical process  in which a substance
        reacts with oxygen to give off heat and  light.
        42
        The substance that undergoes combustion is said to be
        combustible.The lowest temperature at which a substance
        catches fire is called its ignition temperature.
        Activity 6.1
        Fig. 6.2
        Students
        are
        encouraged
        to gather information.
    """

    print("ORIGINAL TEXT:")
    print(test_text)
    print("\nPREPROCESSED TEXT:")
    result = preprocessor.preprocess(test_text)
    print(result)
    print(f"\nOriginal words : {len(test_text.split())}")
    print(f"Cleaned words  : {len(result.split())}")