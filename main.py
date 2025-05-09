import json

import numpy as np
from keras import Sequential
from keras.src.layers import Embedding, Bidirectional, Dense, Dropout, GRU, \
    LayerNormalization, GlobalMaxPooling1D
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.losses import CategoricalCrossentropy
from keras.src.metrics import F1Score
from keras.src.optimizers import Adam
from keras.src.utils import pad_sequences, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

dataset_text = []
dataset_sentiment = []

with open('train.json', 'r', encoding='utf-8') as train_file:
    train_json = json.load(train_file)

    for record in train_json:
        sentiment = -1

        if record['sentiment'] == "negative":
            sentiment = 0
        elif record['sentiment'] == "neutral":
            sentiment = 1
        elif record['sentiment'] == "positive":
            sentiment = 2

        if sentiment == -1:
            continue

        dataset_text.append(record['text'])
        dataset_sentiment.append(sentiment)

vocab_size = 20000
max_len = 500
tokenizer = Tokenizer(num_words=vocab_size, lower=True)
tokenizer.fit_on_texts(dataset_text)
with open('tokenizer_sentiment.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))

sequences = tokenizer.texts_to_sequences(dataset_text)
pads = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

y = to_categorical(dataset_sentiment, num_classes=3)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(dataset_sentiment),
    y=dataset_sentiment
)
class_weight_dict = dict(enumerate(class_weights))

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=1024))
model.add(LayerNormalization())
model.add(Bidirectional(GRU(512, return_sequences=True)))
model.add(Dropout(0.4))
model.add(GlobalMaxPooling1D())
model.add(Dense(32, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax'))

model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(learning_rate=1e-4), metrics=[F1Score()])

X_train, X_val, y_train, y_val = train_test_split(pads, y, test_size=0.2, random_state=42, stratify=dataset_sentiment)

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=8, class_weight=class_weight_dict, batch_size=128, verbose=1)
model.save("model.keras")