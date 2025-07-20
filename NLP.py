import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# 🎯 Step 1: বাংলা ডেটাসেট (10টি বাক্য)
sentences = [
    "আমি আজ খুব খুশি",
    "তুমি আজ স্কুলে যাবে",
    "আজ আকাশে মেঘ",
    "বৃষ্টি পড়ছে সারাদিন",
    "আমার বই হারিয়ে গেছে",
    "তোমার নাম কী",
    "তুমি খুব ভালো",
    "আজকে অনেক গরম",
    "তোমার বাড়ি কোথায়",
    "আমি বাংলা ভাষা ভালোবাসি"
]

# 🎯 Step 2: টোকেনাইজেশন
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1

# 🎯 Step 3: n-gram তৈরি
input_sequences = []
for line in sentences:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        input_sequences.append(n_gram_seq)

# 🎯 Step 4: Padding এবং Features/Labels ভাগ করা
max_seq_len = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')
X = input_sequences[:, :-1]
y = tf.keras.utils.to_categorical(input_sequences[:, -1], num_classes=total_words)

# 🎯 Step 5: মডেল তৈরি
model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_seq_len - 1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

# 🎯 Step 6: কম্পাইল ও ট্রেইন
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=0)

# 🎯 Step 7: মডেল সংরক্ষণ
model.save("bangla_text_gen.h5")

# 🎯 Step 8: টেস্ট - আংশিক বাক্য দিয়ে পূর্ণ বাক্য বানানো
def generate_text(seed_text, next_words=3):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# 🎯 উদাহরণ
print(generate_text("আমি"))
print(generate_text("তুমি আজ"))

