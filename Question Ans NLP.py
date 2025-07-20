import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# 🔹 1. ডেটাসেট তৈরি (প্রশ্ন ও উত্তর)
questions = [
    "ট্রান্সফরমার কি",
    "রেজিস্টর কি",
    "ক্যাপাসিটর কি",
    "ইনডাক্টর কি",
    "ওহমস আইন কি",
    "ভোল্টেজ কি",
    "কারেন্ট কি",
    "রেজিস্ট্যান্স কি",
    "এসি কি",
    "ডিসি কি"
]

answers = [
    "ট্রান্সফরমার একটি বৈদ্যুতিক যন্ত্র যা ভোল্টেজ পরিবর্তন করে",
    "রেজিস্টর একটি প্যাসিভ কম্পোনেন্ট যা বিদ্যুৎ প্রবাহ সীমাবদ্ধ করে",
    "ক্যাপাসিটর চার্জ সংরক্ষণ করে এবং সময়ে সময়ে ছাড়ে",
    "ইনডাক্টর একটি প্যাসিভ ডিভাইস যা চৌম্বকীয় ক্ষেত্র তৈরি করে",
    "ওহমস আইন বলে ভোল্টেজ সমান কারেন্ট গুণ রেজিস্ট্যান্স",
    "ভোল্টেজ হলো বৈদ্যুতিক পটেনশিয়াল পার্থক্য",
    "কারেন্ট হলো ইলেকট্রনের প্রবাহ",
    "রেজিস্ট্যান্স হলো বৈদ্যুতিক প্রতিরোধ",
    "এসি হলো পরিবর্তনশীল কারেন্ট",
    "ডিসি হলো ধ্রুব কারেন্ট"
]

# 🔹 2. টোকেনাইজার ও প্যাডিং
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)

vocab_size = len(tokenizer.word_index) + 1

# প্রশ্ন ও উত্তর সিকোয়েন্স
X = tokenizer.texts_to_sequences(questions)
y = tokenizer.texts_to_sequences(answers)

# প্যাড করা
maxlen = max(len(seq) for seq in X + y)
X = pad_sequences(X, maxlen=maxlen, padding='post')
y = pad_sequences(y, maxlen=maxlen, padding='post')

# y কে One-hot vector বানানো
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# 🔹 3. মডেল তৈরি
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=maxlen),
    LSTM(128, return_sequences=True),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 🔹 4. ট্রেইনিং
model.fit(X, y, epochs=300, verbose=0)

# 🔹 5. মডেল সংরক্ষণ
model.save("qa_model.h5")

# 🔹 6. টেস্ট ফাংশন
def answer_question(input_text):
    seq = tokenizer.texts_to_sequences([input_text])
    seq = pad_sequences(seq, maxlen=maxlen, padding='post')
    pred = model.predict(seq)
    predicted_seq = np.argmax(pred, axis=-1)[0]
    
    # টোকেনকে শব্দে রূপান্তর
    reverse_word_index = dict((i, word) for word, i in tokenizer.word_index.items())
    output_text = ' '.join([reverse_word_index.get(i, '') for i in predicted_seq if i != 0])
    return output_text.strip()

# 🔹 7. উদাহরণ টেস্ট
print("🔹 প্রশ্ন: প্রতিরোধ কি")
print("✅ উত্তর:", answer_question("ক্যাপাসিটর কি"))

print("🔹 প্রশ্ন: ট্রান্সফরমার কি")
print("✅ উত্তর:", answer_question("ট্রান্সফরমার কি"))
