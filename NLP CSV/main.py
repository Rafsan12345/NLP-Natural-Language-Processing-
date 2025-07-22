import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# CSV লোড করা
df = pd.read_csv("math_dataset.csv")

# ইনপুট ও আউটপুট আলাদা করা
questions = df['question'].astype(str).values
answers = df['answer'].astype(str).values

# টোকেনাইজ করা
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
X_seq = tokenizer.texts_to_sequences(questions)
X_pad = pad_sequences(X_seq, padding='post')

# আউটপুট নাম্বারে কনভার্ট (যেহেতু অ্যানসার সংখ্যা)
y = np.array(answers, dtype=float)

# ট্রেন-টেস্ট ভাগ
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# মডেল তৈরি
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=16, input_length=X_pad.shape[1]),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(1)  # সংখ্যা আউটপুট
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# ট্রেইন
model.fit(X_train, y_train, epochs=100, verbose=1)

# সেভ .h5 ফাইলে
model.save("math_solver_model.h5")

# 🔍 টেস্ট: একটি নতুন প্রশ্ন দিলে উত্তর বলে
def solve_math(question):
    seq = tokenizer.texts_to_sequences([question])
    pad = pad_sequences(seq, maxlen=X_pad.shape[1], padding='post')
    prediction = model.predict(pad)
    return round(prediction[0][0], 2)

# উদাহরণ
print("2 + 3 =", solve_math("18 / 6"))
print("10 + 9 =", solve_math("10 + 9"))
print("6 - 1 =", solve_math("6 - 1"))
