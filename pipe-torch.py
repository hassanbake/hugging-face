from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

x_train = ["I have been waiting for Hugging Face course my whole life.",
           "python is great!"]

res = classifier(x_train)
print(res)

batch = tokenizer(x_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
print(batch)

with torch.no_grad():
    outputs = model(**batch)
    print(outputs)
    predictions = F.softmax(outputs.logits,dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)
