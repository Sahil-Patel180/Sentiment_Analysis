import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# 1. Load JSON Data
def load_json_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    headlines = [entry['headline'] for entry in data]
    sarcasm_labels = [entry['is_sarcastic'] for entry in data]
    return headlines, sarcasm_labels

# 2. Preprocessing the text
def preprocess_data(headlines, sarcasm_labels, max_len=100):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(headlines)
    sequences = tokenizer.texts_to_sequences(headlines)
    X = pad_sequences(sequences, maxlen=max_len)
    
    y = np.array(sarcasm_labels)
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    
    return X, y, tokenizer

# 3. Create the Deep Learning Model
def create_model(input_dim, embedding_dim=128, lstm_units=128, output_dim=1):
    model = Sequential([
        Embedding(input_dim, embedding_dim, input_length=100),
        LSTM(lstm_units, dropout=0.3, recurrent_dropout=0.3),
        Dropout(0.5),
        Dense(output_dim, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 4. Train the Model
def train_and_evaluate_model(json_file):
    headlines, sarcasm_labels = load_json_data(json_file)
    X, y, tokenizer = preprocess_data(headlines, sarcasm_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = create_model(5000)
    model.fit(X_train, y_train, epochs=20, batch_size=256, validation_data=(X_test, y_test))
    
    accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy[1] * 100:.2f}%")
    
    model.save('stma.keras')
    return model, tokenizer

# Example usage
if __name__ == "__main__":
    model, tokenizer = train_and_evaluate_model('sarcasm.json')