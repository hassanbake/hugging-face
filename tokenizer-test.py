from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

sequense = "Using a Transformer network is simple."
res = tokenizer(sequense)
print(res)

tokens = tokenizer.tokenize(sequense)
print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

decoded_string = tokenizer.decode(ids)
print(decoded_string)

