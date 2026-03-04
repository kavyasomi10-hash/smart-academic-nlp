# cbse_test.py
# Tests the pipeline on CBSE Class 8 Science textbook content
# Source: NCERT Textbook (ncert.nic.in) - Free and Open Access

import sys
sys.path.append("modules")

from simplification import TextSimplifier
from summarization import TextSummarizer

passages = {
    "Chapter 1 - Crop Production": """
        Agriculture is the science and art of cultivating plants and livestock.
        The practice of agriculture is also known as farming. Scientists who
        practice agriculture are called agricultural scientists. The history of
        agriculture dates back thousands of years. People gathered wild fruits
        and hunted wild animals for food in earlier times. Gradually they learnt
        to grow plants and rear animals for food. Thus, the practice of agriculture
        began. Different crops are grown in different seasons. Food crops like wheat
        and rice are grown in different parts of the country depending upon the
        climatic conditions, availability of water and suitability of soil.
        Kharif crops are sown in the rainy season from June to September.
        Paddy, maize, soyabean, groundnut and cotton are kharif crops.
        Rabi crops are grown in the winter season from October to March.
        Wheat, gram, pea, mustard and linseed are rabi crops.
    """,

    "Chapter 6 - Combustion and Flame": """
        Combustion is a chemical process in which a substance reacts with oxygen
        to give off heat and light. The substance that undergoes combustion is said
        to be combustible. It is also called fuel. The fuel may be solid, liquid or
        gas. Sometimes light is given off during combustion in the form of a flame.
        Candle, kerosene, LPG are examples of fuels which produce flame on burning.
        Charcoal does not produce a flame on burning. For combustion to take place,
        presence of a combustible substance, air which contains oxygen, and heat or
        ignition temperature is required. The lowest temperature at which a substance
        catches fire is called its ignition temperature. The substances which have
        very low ignition temperature and can easily catch fire with a flame are
        called inflammable substances. Petrol, alcohol and liquefied petroleum gas
        are examples of inflammable substances.
    """,

    "Chapter 11 - Force and Pressure": """
        A force is essentially a push or a pull. A bullock pulling a cart, a
        footballer kicking a ball, a carpenter using a saw on a plank of wood,
        a blacksmith hammering a hot iron piece are some examples of forces.
        Forces applied on an object in the same direction add to one another.
        If the two forces act in the opposite directions on an object, the net
        force acting on it is the difference between the two forces. The strength
        of a force is usually expressed by its magnitude. We also need to specify
        the direction in which a force acts. The pressure exerted by air around us
        is known as atmospheric pressure. The atmospheric pressure at sea level is
        approximately equal to the weight of a column of air of 10 metres high
        over each square centimetre of our body. Pressure is defined as force per
        unit area. Liquids also exert pressure on the walls of the container.
    """,
}

print("Loading models...")
simplifier = TextSimplifier()
summarizer = TextSummarizer()

print("\n" + "="*65)
print("  CBSE CLASS 8 SCIENCE - SIMPLIFICATION AND SUMMARIZATION")
print("  Source: NCERT Textbook (ncert.nic.in) - Free and Open Access")
print("="*65)

results = []

for chapter, text in passages.items():
    text = text.strip()
    original_words = len(text.split())

    print("\n" + "="*65)
    print("  " + chapter)
    print("="*65)

    # Show original
    print(f"\nORIGINAL TEXT ({original_words} words):")
    print(text[:250] + "..." if len(text) > 250 else text)

    # Stage 1 — Simplify original text
    print("\nSIMPLIFIED TEXT (T5-base):")
    simplified = simplifier.simplify(text, level="Grade 8")
    simplified_words = len(simplified.split())
    print(simplified)

    # Stage 2 — Summarize ORIGINAL text (not simplified)
    # This gives BART enough words to work with properly
    print("\nSUMMARY (BART-large-CNN):")
    summary = summarizer.summarize(text)
    summary_words = len(summary.split())
    print(summary)

    # Metrics
    simp_compression = round((1 - simplified_words / original_words) * 100, 1)
    summ_compression = round((1 - summary_words / original_words) * 100, 1)

    print(f"\nMETRICS:")
    print(f"  Original words       : {original_words}")
    print(f"  Simplified words     : {simplified_words}  ({simp_compression}% simpler)")
    print(f"  Summary words        : {summary_words}  ({summ_compression}% shorter)")

    results.append({
        "chapter": chapter,
        "original": original_words,
        "simplified": simplified_words,
        "summary": summary_words,
        "simp_comp": simp_compression,
        "summ_comp": summ_compression,
    })

# Final results table
print("\n" + "="*65)
print("  OVERALL RESULTS SUMMARY")
print("="*65)
print(f"{'Chapter':<32} {'Orig':>5} {'Simp':>5} {'Summ':>5} {'Simp%':>6} {'Summ%':>6}")
print("-"*65)
for r in results:
    print(f"{r['chapter']:<32} {r['original']:>5} {r['simplified']:>5} "
          f"{r['summary']:>5} {r['simp_comp']:>5}% {r['summ_comp']:>5}%")

avg_simp = round(sum(r['simp_comp'] for r in results) / len(results), 1)
avg_summ = round(sum(r['summ_comp'] for r in results) / len(results), 1)
print("-"*65)
print(f"{'Average':<32} {'':>5} {'':>5} {'':>5} {avg_simp:>5}% {avg_summ:>5}%")
print("="*65)
print(f"\nSimplification reduces complexity by avg {avg_simp}%")
print(f"Summarization reduces length by avg     {avg_summ}%")