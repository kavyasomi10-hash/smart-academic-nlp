
# modules/simplification.py
# Hybrid Approach: T5-base + Vocabulary Simplification Layer

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
    "combustion": "burning",
    "combustible": "able to catch fire and burn",
    "ignition temperature": "the temperature at which something catches fire",
    "inflammable": "catches fire very easily",
    "atmospheric pressure": "the weight of air pressing down on us",
    "photosynthesis": "the process by which plants make food using sunlight",
    "chlorophyll": "the green substance in leaves that captures sunlight",
    "evaporation": "when a liquid turns into vapor or gas",
    "condensation": "when water vapor cools and turns back into liquid",
    "precipitation": "rain, snow or hail falling from clouds",
    "biodiversity": "the variety of living things in an area",
    "ecosystem": "a community of living things and their environment",
    "microorganism": "a living thing too small to see without a microscope",
    "decomposition": "the process of breaking down dead matter",
    "fertilization": "adding nutrients to soil to help plants grow",
    "germination": "when a seed starts to grow into a plant",
    "photosynthesis": "how plants make food using sunlight",
    "respiration": "the process of breathing to get energy",
    "digestion": "breaking down food in the body",
    "excretion": "removing waste products from the body",
    "reproduction": "the process of producing offspring",
    "hereditary": "passed down from parents to children",
    "gravitational force": "the pulling force of the Earth",
    "friction": "the force that slows things down when they rub together",
    "velocity": "speed in a particular direction",
    "acceleration": "the rate at which speed increases",
    "electromagnetic": "related to electricity and magnetism",
    "legislature": "the part of government that makes laws",
    "judiciary": "the part of government that applies laws in courts",
    "executive": "the part of government that runs the country",
    "constitution": "the basic law of a country",
    "fundamental rights": "basic rights given to every citizen",
    "sovereignty": "having full power and control over a country",
    "secularism": "treating all religions equally",
    "democracy": "a system where people vote to choose their government",
    "marginalisation": "when a group of people is pushed to the edges of society",
    "exploitation": "using someone unfairly for your own benefit",
}


class TextSimplifier:
    def __init__(self):
        print("Loading T5-base simplification model...")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.model.eval()
        print("Model loaded.")

    def _replace_hard_words(self, text):
        result = text
        for hard, simple in sorted(SIMPLIFY_VOCAB.items(), key=lambda x: -len(x[0])):
            pattern = re.compile(re.escape(hard), re.IGNORECASE)
            result = pattern.sub(simple, result)
        return result

    def _t5_shorten(self, text):
        prompt = f"simplify: {text}"
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
                min_length=80,
                num_beams=4,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=2.5,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def simplify(self, text, level="Grade 8"):
        print(f"  Stage 1: Replacing difficult vocabulary...")
        stage1 = self._replace_hard_words(text)
        print(f"  Stage 2: T5 sentence restructuring...")
        stage2 = self._t5_shorten(stage1)
        final = self._replace_hard_words(stage2)
        result = final.strip()
            # Ensure it starts with capital letter
        if result and not result[0].isupper():
            result = result[0].upper() + result[1:]
        return result


if __name__ == "__main__":
    simplifier = TextSimplifier()
    text = input("Enter academic text: ")
    result = simplifier.simplify(text, level="Grade 8")
    print("\nSimplified Text:\n")
    print(result)