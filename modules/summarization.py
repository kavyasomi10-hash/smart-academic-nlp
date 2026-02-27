# modules/summarization.py
from transformers import pipeline

class TextSummarizer:
    def __init__(self):
        print("Loading DistilBART summarization model...")
        self.summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
        )
        print("Model loaded.")

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        result = self.summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        return result[0]["summary_text"]


# Quick test
if __name__ == "__main__":
    summarizer = TextSummarizer()
    text = input("Enter academic text: ")
    result = summarizer.summarize(text)
    print("\nSummary:\n")
    print(result)
 