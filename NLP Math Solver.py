import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# 🌟 ১. Training Data তৈরি
questions = [
    "2+3", "10-4", "5+7", "9-2", "3+6", "8+1", "7-5", "4+4", "6+3", "12-5"
]
answers = [
    "5", "6", "12", "7", "9", "9", "2", "8", "9", "7"
]

# 🌟 ২. Tokenization
tokenizer = Tokenizer(char_level=True)  # character-level tokenizer
tokenizer.fit_on_texts(questions + answers)

X_seq = tokenizer.texts_to_sequences(questions)
y_seq = tokenizer.texts_to_sequences(answers)

# Padding
max_len = max([len(seq) for seq in X_seq])
X = pad_sequences(X_seq, maxlen=max_len, padding='post')
y = np.array([seq[0] for seq in y_seq])  # single-digit answer

# 🌟 ৩. Model বানানো
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=10, input_length=max_len))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(tokenizer.word_index)+1, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 🌟 ৪. Model Train
model.fit(X, y, epochs=300, verbose=0)

# 🌟 ৫. Save Model
model.save("math_solver.h5")

# 🌟 ৬. Prediction Function
def solve_math(expr):
    seq = tokenizer.texts_to_sequences([expr])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded, verbose=0)
    predicted_token = np.argmax(pred)
    
    for char, index in tokenizer.word_index.items():
        if index == predicted_token:
            return char
    return "?"

# 🌟 ৭. Test
test_expr = "2+3"
print(f"{test_expr} = {solve_math(test_expr)}")
