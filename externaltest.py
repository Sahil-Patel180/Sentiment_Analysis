from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

#Load the trained model
model = load_model('stma.keras')

#Sample input text for sentiment analysis
input_text = ""

#Define parameters based on how the model was trained (e.g., max sequence length)
max_sequence_length = 100  #Change this to the max length used during training
vocab_size = 10000  #Adjust this to the vocabulary size used during training

#Initialize the tokenizer (you would have used the same one during training)
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts([input_text])  #Fit tokenizer on your corpus or load a saved tokenizer

#Convert the input text to a sequence
input_sequence = tokenizer.texts_to_sequences([input_text])

#Pad the sequence to the same length used during training
input_padded = pad_sequences(input_sequence, maxlen=max_sequence_length)

#Make a prediction (output is a probability)
predictions = model.predict(input_padded)

#Output prediction result (assuming binary classification, e.g., positive/negative)
predicted_class = np.argmax(predictions, axis=1)
print(f"Predicted sentiment class: {predicted_class}")
