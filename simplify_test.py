from transformers import pipeline
print("Loading model...")
simplifier = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

print("Model loaded successfully.")

text = "Photosynthesis is a biochemical process by which green plants synthesize food using sunlight."

result = simplifier(
    "Simplify this text: " + text,
    max_length=100
)

print("\nSimplified Output:")
print(result[0]["generated_text"])