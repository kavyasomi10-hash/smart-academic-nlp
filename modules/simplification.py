# modules/simplification.py
# Hybrid: Vocabulary Replacement + T5-base
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re

SIMPLIFY_VOCAB = {
    "adenosine triphosphate": "ATP (the energy currency of the cell)",
    "oxidative phosphorylation": "a process that makes energy",
    "cellular respiration": "how cells produce energy",
    "eukaryotic": "complex (having a nucleus)",
    "organelle": "a small part inside a cell",
    "microorganism": "a tiny living thing too small to see",
    "photosynthesis": "how plants make food using sunlight",
    "chlorophyll": "the green substance in leaves",
    "decomposition": "the breaking down of dead matter",
    "germination": "when a seed starts to grow",
    "fertilization": "adding nutrients to help plants grow",
    "respiration": "the process of breathing to get energy",
    "combustion": "burning",
    "combustible": "able to catch fire and burn",
    "ignition temperature": "the temperature at which something catches fire",
    "inflammable": "catches fire very easily",
    "atmospheric pressure": "the weight of air pressing down on us",
    "gravitational force": "the pulling force of the Earth",
    "friction": "the force that slows things down",
    "velocity": "speed in a particular direction",
    "acceleration": "the rate at which speed increases",
    "evaporation": "when liquid turns into vapor",
    "condensation": "when vapor cools and turns into liquid",
    "precipitation": "rain or snow falling from clouds",
    "biodiversity": "the variety of living things in an area",
    "ecosystem": "a community of living things and their environment",
    "solute": "the substance that gets dissolved",
    "solvent": "the liquid that does the dissolving",
    "solution": "a mixture where one substance dissolves in another",
    "element": "a pure substance made of one type of atom",
    "compound": "a substance made of two or more elements joined",
    "mixture": "two or more substances combined but not joined",
    "reflection": "when light bounces off a surface",
    "refraction": "when light bends passing through a substance",
    "concave": "curved inward like a bowl",
    "convex": "curved outward like a ball",
    "fundamental rights": "basic rights given to every citizen",
    "sovereignty": "having full power over a country",
    "democracy": "a system where people vote to choose government",
    "constitution": "the basic law of a country",
    "legislature": "the part of government that makes laws",
    "judiciary": "the part of government that applies laws",
    "self-attention mechanism": "a method that lets AI focus on important words",
    "unsupervised learning": "learning without being given correct answers",
    "transfer learning": "using a trained model on a new task",
    "pre-trained": "already trained on a lot of data",
    "fine-tuned": "adjusted for a specific task",
    "synthesize": "make",
    "utilize": "use",
    "leverage": "use",
    "facilitate": "help",
    "subsequently": "after that",
    "consequently": "as a result",
    "significant": "important",
    "significantly": "greatly",
    "methodology": "method or approach",
    "novel": "new",
    "robust": "strong and reliable",
    "efficacy": "how well something works",
    "antibiotics": "medicines that kill harmful bacteria",
    "bacteria": "tiny living things that can cause disease",
    "fungi": "organisms like mushrooms that break down matter",
    "protozoa": "tiny single-celled living things",
    "cholera": "a serious disease caused by bacteria in water",
    "tuberculosis": "a serious lung disease caused by bacteria",
    "typhoid": "a disease caused by bacteria in food or water",
    "nitrogen": "a gas in air that helps plants grow",
    "paradigm": "a way of thinking or doing things",
    "computational": "related to computer processing",
    "democratizing": "making available to everyone",
    "constraints": "limitations or restrictions",
    "distillation": "a process of extracting the most important parts",
    "dependencies": "relationships between different parts",
    "corpora": "large collections of text data",
    "deployment": "putting something into practical use",
}


class TextSimplifier:
    def __init__(self):
        print("Loading T5-base simplification model...")
        self.tokenizer = T5Tokenizer.from_pretrained(
            "t5-base", legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(
            "t5-base")
        self.model.eval()
        print("Model loaded.")

    def _replace_hard_words(self, text):
        result = text
        for hard, simple in sorted(
                SIMPLIFY_VOCAB.items(),
                key=lambda x: -len(x[0])):
            pattern = re.compile(
                re.escape(hard), re.IGNORECASE)
            result = pattern.sub(simple, result)
        return result

    def _t5_shorten(self, text):
        prompt = "simplify: " + text
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_length=200,
                min_length=60,
                num_beams=4,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=2.5,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        return self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True)

    def _clean_output(self, text):
        # Remove T5 hallucinated prefixes like "John sutter:"
        text = re.sub(r'^[A-Za-z\s]{0,25}:\s*', '', text)
        # Remove anything before first capital letter
        match = re.search(r'[A-Z]', text)
        if match and match.start() < 20:
            text = text[match.start():]
        return text.strip()

    def simplify(self, text, level="Grade 8"):
        print("  Stage 1: Replacing difficult vocabulary...")
        stage1 = self._replace_hard_words(text)
        print("  Stage 2: T5 sentence restructuring...")
        stage2 = self._t5_shorten(stage1)
        stage2 = self._clean_output(stage2)
        final = self._replace_hard_words(stage2)
        result = final.strip()
        if result and not result[0].isupper():
            result = result[0].upper() + result[1:]
        return result


if __name__ == "__main__":
    simplifier = TextSimplifier()
    text = input("Enter academic text: ")
    result = simplifier.simplify(text)
    print("\nSimplified Text:\n")
    print(result)
    orig = len(text.split())
    simp = len(result.split())
    print(f"\nOriginal : {orig} words")
    print(f"Simplified: {simp} words")
    print(f"Reduction : {round((1-simp/orig)*100,1)}%")