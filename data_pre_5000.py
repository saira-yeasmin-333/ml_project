import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,roc_auc_score
from sklearn.preprocessing import StandardScaler
import pickle

# Load your data
dat = pd.read_csv("balanced_reviews.csv")

# Preprocess your DataFrame
dat = dat.drop(columns=['book_id', 'ratings_count', 'review_likes', 'like_share'])
dat["rating_diff"] = dat["user_rating"] - dat["avg_rating"]
dat = dat.drop(columns=['avg_rating'])
dat["quote"] = dat["review_text"].str.contains("\"")
dat["review_length"] = dat["review_text"].str.len()
# Drop rows with missing 'review_text'
dat = dat.dropna(subset=['review_text']).reset_index(drop=True)


# Initialize the tokenizer and model from the pre-trained BERT base uncased model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=False)
model.eval()  # Set the model to evaluation mode

# Tokenize, encode, and pad the reviews
max_sequence_length = 256  # Maximum sequence length
tokenized_reviews = [tokenizer.encode(review, add_special_tokens=True, max_length=max_sequence_length, truncation=True, padding='max_length') for review in dat['review_text']]

# Convert the tokenized reviews into tensors
input_ids = torch.tensor(tokenized_reviews)
attention_masks = torch.tensor([[float(i > 0) for i in seq] for seq in input_ids])

# Create a DataLoader
dataset = TensorDataset(input_ids, attention_masks)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# Define a function to get sentiment predictions
def get_sentiment_predictions(model, dataloader):
    model.eval()  # Make sure the model is in evaluation mode
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_masks = batch
            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs[0]
            probabilities = torch.softmax(logits, dim=1)
            predictions.extend(probabilities[:, 1].tolist())  # Assuming index 1 corresponds to positive sentiment
    return predictions

# Get sentiment predictions
sentiment_predictions = get_sentiment_predictions(model, dataloader)

# Add the predictions to the DataFrame
dat['sentiment_probabilities'] = sentiment_predictions

# Print the first few rows to verify
# print(dat.head())

# dat.to_csv("filtered_csv_with_sentiment.csv", index=False)

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure you've downloaded the necessary NLTK data
nltk.download('punkt')

# Function to lemmatize text
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in word_tokens]
    return ' '.join(lemmatized_words)


# Apply lemmatization to the review_text column
dat['lemmatized_text'] = dat['review_text'].apply(lemmatize_text)

# Initialize CountVectorizer
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(dat['lemmatized_text'])

# Convert the CountVectorizer output to a list of dictionaries
feature_names = vectorizer.get_feature_names_out()
X_dense = X.todense()
count_list = [dict(zip(feature_names, X_dense[i].tolist()[0])) for i in range(X.shape[0])]

# Add the list of dictionaries as a new column in the DataFrame
dat['countvectorized_data'] = count_list

# Print the first few rows to verify
# print(dat.head())

# # Save the modified DataFrame
# dat.to_csv("filtered_csv_with_countvectorized_data.csv", index=False)

# for col in dat.columns:
#     print(f'{col} : {type(dat[col])}')

from scipy.sparse import csr_matrix

def transform_countvectorized_data(data, vocab_size=100):
    transformed_data = np.zeros((len(data), vocab_size))
    for i, token_counts in enumerate(data):
        for token, count in token_counts.items():
            # Simple hash function to convert token to an index
            # Note: This can cause collisions
            index = hash(token) % vocab_size
            transformed_data[i, index] += count
    return transformed_data

dat['total_token_count'] = dat['countvectorized_data'].apply(lambda x: sum(x.values()))

# Prepare the feature matrix excluding raw text and complex dictionary data
X = dat[['user_reviews', 'user_rating', 'days_since_review', 'rating_diff', 'quote', 'review_length', 'total_token_count']].copy()
# Ensure all data is numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Fill any NaNs that resulted from conversion errors
X.fillna(0, inplace=True)

# Target variable
y = dat['popular']

print('data preprocess finish')

data_with_target = pd.concat([X, y], axis=1)

# Write the DataFrame to a CSV file
data_with_target.to_csv('data_with_target_10000.csv', index=False)