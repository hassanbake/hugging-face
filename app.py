from transformers import pipeline

def main():
    classifier = pipeline("sentiment-analysis")
    res = classifier("I have been waiting for Hugging Face course my whole life.")
    print(res)

if __name__ == "__main__":
    main()