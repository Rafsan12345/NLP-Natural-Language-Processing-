import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import os

# 1️⃣ ডেটাসেট তৈরি
data = {
    'sentence': [
        'আমি আজ খুব খুশি',
        'আজ খুব মন খারাপ',
        'তুমি অনেক ভালো বন্ধু',
        'সব কিছু বাজে লাগছে',
        'আজকের দিনটা সুন্দর',
        'তুমি ভালো করেছো',
        'আমার মনটা খারাপ',
        'আজ দারুণ লাগছে',
        'তোমার ব্যবহার খারাপ',
        'আজ আমি অনেক আনন্দিত'
    ],
    'label': [
        'positive',
        'negative',
        'positive',
        'negative',
        'positive',
        'positive',
        'negative',
        'positive',
        'negative',
        'positive'
    ]
}

df = pd.DataFrame(data)

# 2️⃣ টোকেনাইজেশন
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(df['sentence'])
sequences = tokenizer.texts_to_sequences(df['sentence'])
padded = pad_sequences(sequences, padding='post')

# 3️⃣ লেবেল এনকোডিং
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['label'])
labels = to_categorical(labels)

# 4️⃣ মডেল তৈরি
vocab_size = len(tokenizer.word_index) + 1
model = Sequential([
    Embedding(vocab_size, 16, input_length=padded.shape[1]),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')  # 2 classes: positive, negative
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5️⃣ মডেল ট্রেইনিং
model.fit(padded, labels, epochs=50, verbose=0)

# 6️⃣ মডেল সেভ করা
model.save("bangla_sentiment_model.h5")

# 7️⃣ টেস্ট করা
def classify(sentence):
    model = load_model("bangla_sentiment_model.h5")
    seq = tokenizer.texts_to_sequences([sentence])
    pad = pad_sequences(seq, maxlen=padded.shape[1], padding='post')
    pred = model.predict(pad)
    label = label_encoder.inverse_transform([np.argmax(pred)])
    return label[0]

# ✅ উদাহরণ টেস্ট
test_sentence = "আজকে মন খারাপ"
print("🔍 বাক্য:", test_sentence)
print("➡️ ক্লাসিফিকেশন রেজাল্ট:", classify(test_sentence))
