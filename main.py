# main.py
# Entry point for Smart Academic NLP project
# Streamlit UI will be connected here in Week 5

import sys
sys.path.append("modules")

from simplification import TextSimplifier
from summarization import TextSummarizer

def main():
    print("Smart Academic Text Processing System")
    print("======================================")
    print("1. Simplify Text")
    print("2. Summarize Text")
    print("3. Full Pipeline (Simplify then Summarize)")
    print("4. Exit")

    choice = input("\nEnter choice (1-4): ")

    if choice == "1":
        text = input("Enter text: ")
        s = TextSimplifier()
        print("\nSimplified:\n", s.simplify(text))

    elif choice == "2":
        text = input("Enter text: ")
        s = TextSummarizer()
        print("\nSummary:\n", s.summarize(text))

    elif choice == "3":
        text = input("Enter text: ")
        simp = TextSimplifier()
        summ = TextSummarizer()
        simplified = simp.simplify(text)
        summary = summ.summarize(simplified)
        print("\nSimplified:\n", simplified)
        print("\nSummary:\n", summary)

    elif choice == "4":
        print("Exiting...")

    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()