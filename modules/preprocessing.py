# modules/preprocessing.py
# YOUR OWN CODE - Text Preprocessing Pipeline
import re
import unicodedata


class TextPreprocessor:
    def __init__(self):
        print("Text Preprocessor ready.")

    def remove_non_printable(self, text):
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
        return text

    def remove_page_numbers(self, text):
        text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)
        text = re.sub(r'^\d{1,3}\s*$', '', text,
                      flags=re.MULTILINE)
        return text

    def fix_broken_words(self, text):
        text = re.sub(r'-\n(\w)', r'\1', text)
        lines = text.split('\n')
        fixed, buffer = [], ""
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

    def remove_ncert_artifacts(self, text):
        text = re.sub(r'Activity\s*\d*', '', text)
        text = re.sub(r'Exercise\s*\d*', '', text)
        text = re.sub(r'Fig\.\s*\d+[\.\d]*', '', text)
        text = re.sub(r'Table\s*\d+[\.\d]*', '', text)
        text = re.sub(r'Box\s*\d+[\.\d]*', '', text)
        text = re.sub(r'Do\s+You\s+Know\??', '', text)
        text = re.sub(r'Think\s+and\s+Discuss', '', text)
        return text

    def fix_punctuation(self, text):
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        text = re.sub(r',([A-Za-z])', r', \1', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text

    def remove_extra_whitespace(self, text):
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        return text.strip()

    def validate_input(self, text):
        word_count = len(text.split())
        if word_count < 30:
            return False, f"Too short: {word_count} words."
        if word_count > 2000:
            return False, f"Too long: {word_count} words."
        return True, f"Valid: {word_count} words."

    def preprocess(self, text):
        print("  Preprocessing: Removing non-printable chars...")
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