def __init__(self):
    print("Loading BART summarization model...")
    from transformers import BartForConditionalGeneration, BartTokenizer
    self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    self.model.eval()
    print("Model loaded.")