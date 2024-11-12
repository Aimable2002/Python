# import torch
# import torch.nn as nn

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_sequence_length):
#         super().__init__()
#         self.max_sequence_length = max_sequence_length
#         self.d_model = d_model
        
#         # Pre-compute position encoding matrix
#         self.register_buffer('PE', self._create_positional_encoding())
    
#     def _create_positional_encoding(self):
#         even_i = torch.arange(0, self.d_model, 2).float()
#         denominator = torch.pow(10000, even_i/self.d_model)
#         position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        
#         even_PE = torch.sin(position / denominator)
#         odd_PE = torch.cos(position / denominator)
#         stacked = torch.stack([even_PE, odd_PE], dim=2)
#         PE = torch.flatten(stacked, start_dim=1, end_dim=2)
#         return PE
    
#     def forward(self, x):
#         # x expected shape: [batch_size, seq_len, d_model]
#         return x + self.PE[:x.size(1), :]
    
    

# import numpy as np
# import pandas as pd
# from nltk.tokenize import word_tokenize
# import string
# import re
# from nltk.corpus import stopwords
# from collections import Counter
# from nltk.stem import WordNetLemmatizer
# import nltk
# from spellchecker import SpellChecker

# nltk.download('wordnet')
# nltk.download('punkt')

# stop_words = set(stopwords.words('english'))

# spell = SpellChecker()

# spell = SpellChecker()
# lemmatizer = WordNetLemmatizer()

# def correct_spelling(text):
#     words = text.split()
#     corrected_words = []
#     for word in words:
#         corrected = spell.correction(word)
#         corrected_words.append(corrected if corrected else word)
#     return ' '.join(corrected_words)


# df = pd.read_csv("trainer.text", engine="python", sep="\r\n", names=["text"])



# df['lowerCase'] = df['text'].apply(lambda x: x.lower())

# df['remove_punctuation'] = df['lowerCase'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# df['remove_numbers'] = df['remove_punctuation'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# df['remove_stopwords'] = df['remove_numbers'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# # df['spell_corrected'] = df['remove_stopwords'].apply(correct_spelling)

# word_counts = Counter()
# for text in df['remove_stopwords'].values:
#     words = text.split()
#     word_counts.update(words)
# print(word_counts)

# # lemmatizer = WordNetLemmatizer()
# df['lemmatized'] = df['remove_stopwords'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# df['tokenized'] = df['lemmatized'].apply(lambda x: word_tokenize(x))


# print(df['lemmatized'].head())


















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

# Convert texts to vectors
vectors = [vectorizer.text_to_vector(text) for text in df['final_text']]

# Print results
print("\nVocabulary:")
for word, idx in sorted(vectorizer.word2idx.items(), key=lambda x: x[1]):
    print(f"{word}: {idx}")

print("\nText to Vector conversion:")
for text, vector in zip(df['final_text'], vectors):
    print(f"\nOriginal text: {text}")
    print(f"Vector: {vector}")

print(df['final_text'].head())



