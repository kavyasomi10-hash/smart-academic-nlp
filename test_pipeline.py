# test_pipeline.py — Tests both modules together as a pipeline
import sys
sys.path.append("modules")

from simplification import TextSimplifier
from summarization import TextSummarizer

text = """Transformer models have fundamentally revolutionized the field of
natural language processing by introducing a self-attention mechanism that
allows the model to weigh the relevance of different words in a sentence
regardless of their positional distance from one another. Unlike recurrent
neural networks, which process tokens sequentially and suffer from vanishing
gradient problems over long sequences, transformers process all the tokens
in parallel, significantly improving computational efficiency. Pre-trained
models such as BERT, GPT, and T5 leverage large-scale unsupervised learning
on massive text corpora, enabling them to be fine-tuned on downstream tasks
with minimal labeled data."""

print("="*60)
print("ORIGINAL TEXT:")
print(text)
print(f"\nWord count: {len(text.split())}")

print("\n" + "="*60)
print("STEP 1 — SIMPLIFICATION (FLAN-T5 hybrid):")
simplifier = TextSimplifier()
simplified = simplifier.simplify(text)
print(simplified)
print(f"\nWord count: {len(simplified.split())}")

print("\n" + "="*60)
print("STEP 2 — SUMMARIZATION (DistilBART):")
summarizer = TextSummarizer()
summary = summarizer.summarize(simplified)
print(summary)
print(f"\nWord count: {len(summary.split())}")

print("\n" + "="*60)
original_words = len(text.split())
summary_words  = len(summary.split())
compression    = round((1 - summary_words / original_words) * 100, 1)
print(f"📊 COMPRESSION RATIO: {original_words} → {summary_words} words ({compression}% reduction)")


