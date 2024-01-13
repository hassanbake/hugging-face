from transformers import pipeline
# from transformers import AutoTokenizer, TFAutoModelForSequenceClassification # for tensorflow
from transformers import AutoTokenizer, AutoModelForSequenceClassification # for pytorch

def main():
    classifier = pipeline("sentiment-analysis")
    res = classifier("I have been waiting for Hugging Face course my whole life.")
    print(res)

    generator = pipeline("text-generation", model="distilgpt2")
    res = generator(
        "In this course we will teach you how to",
        max_length = 30,
        num_return_sequences = 2
    )
    print(res)

    classifier = pipeline("zero-shot-classification")
    res = classifier(
        "This is a course about python list comprehension",
        candidate_labels = ["education", "politics", "business"]
    )
    print(res)
  
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    res = classifier("I have been waiting for Hugging Face course my whole life.")
    print(res)







if __name__ == "__main__":
    main()