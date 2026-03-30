# modules/summarization.py
# Abstractive Summarization using BART-large-CNN
# Direct model loading (no pipeline API)

import re
import torch
from transformers import BartForConditionalGeneration, BartTokenizer


class TextSummarizer:
    def __init__(self):
        print("Loading BART summarization model...")
        self.tokenizer = BartTokenizer.from_pretrained(
            "facebook/bart-large-cnn")
        self.model = BartForConditionalGeneration.from_pretrained(
            "facebook/bart-large-cnn")
        self.model.eval()
        print("Model loaded.")

    def _clean(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def _chunk(self, text, size):
        words = text.split()
        return [" ".join(words[i:i+size])
                for i in range(0, len(words), size)]

    def _generate(self, text, max_len, min_len):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )
        with torch.no_grad():
            output = self.model.generate(
                inputs["input_ids"],
                max_length=max_len,
                min_length=min_len,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
        return self.tokenizer.decode(
            output[0], skip_special_tokens=True)

    def summarize(self, text):
        text = self._clean(text)
        word_count = len(text.split())

        if word_count < 30:
            return "Text too short. Please provide at least 30 words."

        max_len = max(60,  int(word_count * 0.35))
        min_len = max(25,  int(word_count * 0.20))

        if word_count > 600:
            chunks = self._chunk(text, 550)
            parts = []
            for chunk in chunks:
                c_words = len(chunk.split())
                part = self._generate(
                    chunk,
                    max(60, int(c_words * 0.35)),
                    max(25, int(c_words * 0.20))
                )
                parts.append(part)
            combined = " ".join(parts)
            return self._generate(combined, max_len, min_len)

        return self._generate(text, max_len, min_len)


if __name__ == "__main__":
    summarizer = TextSummarizer()
    text = input("Enter academic text: ")
    result = summarizer.summarize(text)
    print("\nSummary:\n")
    print(result)
    orig = len(text.split())
    summ = len(result.split())
    print(f"\nOriginal words: {orig}")
    print(f"Summary words : {summ}")
    print(f"Compression   : {round((1 - summ/orig)*100, 1)}%")
