from modules.simplification import TextSimplifier

if __name__ == "__main__":
    print("Program started")

    simplifier = TextSimplifier()
    print("Model loaded")

    text = input("Enter academic text: ")

    simplified = simplifier.simplify(text)

    print("\nSimplified Text:\n")
    print(simplified)

