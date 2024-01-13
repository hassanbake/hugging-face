from transformers import pipeline

class Classification:
    def classify(texts):
        classifier = pipeline("sentiment-analysis")
        return classifier(texts)

    def classify(text, candidate_labels):
        classifier = pipeline("zero-shot-classification")
        return classifier(text, candidate_labels=candidate_labels)
    
class Generator:
    def generate(text):
        generator = pipeline("text-generation")
        return generator(text)