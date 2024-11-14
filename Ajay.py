import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import string
import re
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
import nltk
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

nltk.download('wordnet')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
spell = SpellChecker()
lemmatizer = WordNetLemmatizer()

class TextVectorizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocabulary = set()
        
    def build_vocabulary(self, texts):
        # Add special tokens
        self.word2idx['<PAD>'] = 0
        self.word2idx['<UNK>'] = 1
        
        # Build vocabulary from texts
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)
      
        # Add words to vocabulary
        for idx, (word, count) in enumerate(word_counts.most_common(), start=2):
            self.word2idx[word] = idx
            
            # Create reverse mapping
            self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
    def text_to_vector(self, text):
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in text.split()]

# Read and process text
df = pd.read_csv("trainer.text", engine="python", sep="\r\n", names=["text"])

# Text preprocessing pipeline
df['lowerCase'] = df['text'].apply(lambda x: x.lower())
df['remove_punctuation'] = df['lowerCase'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
df['remove_numbers'] = df['remove_punctuation'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
df['remove_stopwords'] = df['remove_numbers'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
df['spell_corrected'] = df['remove_stopwords'].apply(lambda x: ' '.join([spell.correction(word) if spell.correction(word) else word for word in x.split()]))
df['final_text'] = df['spell_corrected'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Create vectorizer and build vocabulary
vectorizer = TextVectorizer()
vectorizer.build_vocabulary(df['final_text'])
d_model = 512

embedding_layer = nn.Embedding(num_embeddings=len(vectorizer.word2idx), embedding_dim=d_model)

# Convert texts to vectors
# vectors = [torch.tensor(vectorizer.text_to_vector(text)) for text in df['final_text']]
vectors = [embedding_layer(torch.tensor(vectorizer.text_to_vector(text))) for text in df['final_text']]

# Pad the sequences to ensure they are of the same length
# padded_vectors = pad_sequence(vectors, batch_first=True, padding_value=vectorizer.word2idx['<PAD>'])
padded_vectors = pad_sequence(vectors, batch_first=True, padding_value=0)  # Use 0 for <PAD> in embeddings

# Print results
print("\nVocabulary:")
for word, idx in sorted(vectorizer.word2idx.items(), key=lambda x: x[1]):
    print(f"{word}: {idx}")

print("\nText to Vector conversion:")
# for text, vector in zip(df['final_text'], vectors):
#     print(f"\nOriginal text: {text}")
#     print(f"Vector shape: {vector.shape}, Vector: {vector.tolist()}")   # Convert tensor to list for printing

# print(df['final_text'].head())

# Now you can pass `padded_vectors` to your transformer model in run.py
# Example:
# model = ...  # Load your transformer model
# output = model(padded_vectors)  # Pass the padded vectors to the model



