import json


import numpy as np
import pandas as pd
from keras.src.metrics import F1Score
from keras.src.saving import load_model
from keras.src.utils import pad_sequences
from keras.src.legacy.preprocessing.text import tokenizer_from_json

test_text = []
test_id = []

with open('test.json', 'r', encoding='utf-8') as test_json:
    data = json.load(test_json)
    for row in data:
        if 'text' in row and 'id' in row:
            test_text.append(row['text'])
            test_id.append(row['id'])

with open('tokenizer_sentiment.json', 'r', encoding='utf-8') as f:
    tokenizer_json = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_json)

vocab_size = 20000
max_len = 400
sequences = tokenizer.texts_to_sequences(test_text)
pads = pad_sequences(sequences, maxlen=max_len)


model = load_model("model.keras", custom_objects={'F1Score': F1Score})

predictions = model.predict(pads)
predicted_labels = np.argmax(predictions, axis=1)

label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}
predicted_sentiments = [label_map[label] for label in predicted_labels]

df = pd.DataFrame({
    'id': test_id,
    'sentiment': predicted_sentiments
})
df.to_csv('submission.csv', index=False)