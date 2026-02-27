# modules/simplification.py
# Hybrid Approach: T5-base (sentence shortening) + Vocabulary Simplification Layer

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re

SIMPLIFY_VOCAB = {
    "adenosine triphosphate": "ATP (the energy currency of the cell)",
    "oxidative phosphorylation": "a process that makes energy",
    "cellular respiration": "how cells produce energy",
    "mitochondrial membrane": "the outer wall of the mitochondria",
    "eukaryotic": "complex (having a nucleus)",
    "organelle": "a small part inside a cell",
    "metabolic processes": "body functions that use energy",
    "metabolism": "how the body uses energy",
    "synthesize": "make",
    "synthesis": "creation",
    "glucose": "sugar",
    "self-attention mechanism": "a method that lets the AI focus on important words",
    "recurrent neural networks": "older AI models that read text one word at a time",
    "vanishing gradient": "a problem where older AI models stop learning",
    "unsupervised learning": "learning without being given correct answers",
    "transfer learning": "using a trained model on a new task",
    "pre-trained": "already trained on a lot of data",
    "fine-tuned": "adjusted for a specific task",
    "fine-tuning": "adjusting a model for a specific task",
    "downstream tasks": "tasks the AI is applied to after training",
    "corpus": "a large collection of text",
    "tokenization": "splitting text into small word pieces",
    "embedding": "turning words into numbers the AI understands",
    "abstractive summarization": "writing a new short version in your own words",
    "sentiment analysis": "finding if text is positive or negative",
    "machine translation": "translating text from one language to another",
    "fundamentally revolutionized": "completely changed",
    "paradigmatic shift": "a major change in approach",
    "precipitated": "caused",
    "demonstrate": "show",
    "demonstrates": "shows",
    "utilize": "use",
    "utilizes": "uses",
    "leverage": "use",
    "leverages": "uses",
    "facilitate": "help",
    "necessitates": "requires",
    "subsequently": "after that",
    "consequently": "as a result",
    "furthermore": "also",
    "nevertheless": "still",
    "aforementioned": "mentioned earlier",
    "significant": "important",
    "significantly": "greatly",
    "substantially": "by a large amount",
    "methodology": "method or approach",
    "framework": "a system or structure",
    "paradigm": "a way of thinking",
    "novel": "new",
    "robust": "strong and reliable",
    "scalable": "can handle more data easily",
    "efficacy": "how well it works",
    "augmented": "improved",
    "inherent": "built-in",
    "salient": "important",
    "comprises": "is made of",
}


class TextSimplifier:
    def __init__(self):
        print("Loading T5-base simplification model...")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.model.eval()
        print("Model loaded.")

    def _replace_hard_words(self, text: str) -> str:
        result = text
        for hard, simple in sorted(SIMPLIFY_VOCAB.items(), key=lambda x: -len(x[0])):
            pattern = re.compile(re.escape(hard), re.IGNORECASE)
            result = pattern.sub(simple, result)
        return result

    def _t5_shorten(self, text: str) -> str:
        prompt = f"simplify: {text}"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_length=200,
                min_length=40,
                num_beams=4,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=2.5,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def simplify(self, text: str, level: str = "Grade 8") -> str:
        print(f"\n  Stage 1: Replacing difficult vocabulary...")
        stage1 = self._replace_hard_words(text)
        print(f"  Stage 2: T5 sentence restructuring...")
        stage2 = self._t5_shorten(stage1)
        final = self._replace_hard_words(stage2)
        return final


if __name__ == "__main__":
    simplifier = TextSimplifier()
    test_texts = [
        "The mitochondria is an organelle found in eukaryotic cells that is responsible for the production of adenosine triphosphate through the process of cellular respiration. This organelle utilizes oxygen and glucose to synthesize ATP via oxidative phosphorylation, which occurs across the inner mitochondrial membrane.",
        "Transformer models have fundamentally revolutionized the field of natural language processing by introducing a self-attention mechanism that allows the model to weigh the relevance of different words in a sentence.",
    ]
    print("\n" + "="*60)
    print("SIMPLIFICATION TEST — HYBRID (Vocab + T5)")
    print("="*60)
    for i, text in enumerate(test_texts, 1):
        print(f"\nOriginal {i}:\n{text}")
        result = simplifier.simplify(text)
        print(f"\nSimplified {i}:\n{result}")
        print("-"*60)
    print("\n--- Try your own text ---")
    text = input("\nEnter academic text: ")
    result = simplifier.simplify(text)
    print("\nSimplified Text:\n")
    print(result)