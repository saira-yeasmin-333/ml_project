import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,roc_auc_score
from sklearn.preprocessing import StandardScaler
import pickle

# Load your data
dat = pd.read_csv("first_10_reviews.csv")

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
model = BertModel.from_pretrained('bert-base-uncased')  # Use BertModel
model.eval()  # Set the model to evaluation mode

# Function to get BERT embeddings using BertModel
def get_bert_embeddings(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the last hidden state as embeddings (alternative to pooler_output)
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze()  # Get the embeddings of the [CLS] token
    return embeddings.numpy()  # Convert the tensor to a NumPy array

# Apply the function to the 'review_text' column to get embeddings
dat['bert_embeddings'] = dat['review_text'].apply(get_bert_embeddings)

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



# Apply the function to the 'lemmatized_text' column to get embeddings
dat['bert_embeddings'] = dat['review_text'].apply(get_bert_embeddings)

# Convert the list of embeddings into a DataFrame where each column represents one dimension of the embeddings
embeddings_df = pd.DataFrame(dat['bert_embeddings'].tolist())

# Prepare the feature matrix with other features
X = dat[['user_reviews', 'user_rating', 'days_since_review', 'rating_diff', 'quote', 'review_length','sentiment_probabilities']].copy()
X = X.apply(pd.to_numeric, errors='coerce')  # Ensure all data is numeric
X.fillna(0, inplace=True)

# Concatenate the embeddings DataFrame with the other features
X = pd.concat([X, embeddings_df], axis=1)

# Assuming 'popular' is your target variable
y = dat['popular']

# Concatenate features and target into a single DataFrame
data_with_target = pd.concat([X, y], axis=1)

def calculate_features(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    num_words = len(tokens)  # Number of words
    word_lengths = [len(word) for word in tokens]
    avg_word_len = np.mean(word_lengths) if word_lengths else 0  # Average word length

    # POS tagging
    pos_tags = pos_tag(tokens)
    num_verbs = len([word for word, tag in pos_tags if tag.startswith('VB')])  # Count verbs
    num_nouns = len([word for word, tag in pos_tags if tag.startswith('NN')])  # Count nouns
    num_adj = len([word for word, tag in pos_tags if tag.startswith('JJ')])  # Count adjectives

    pct_verbs = num_verbs / num_words if num_words > 0 else 0  # Percentage of verbs
    pct_nouns = num_nouns / num_words if num_words > 0 else 0  # Percentage of nouns
    pct_adj = num_adj / num_words if num_words > 0 else 0  # Percentage of adjectives

    sentences = nltk.sent_tokenize(text)
    avg_sent_len = np.mean([len(word_tokenize(sentence)) for sentence in sentences]) if sentences else 0  # Average sentence length

    return num_words, avg_word_len, avg_sent_len, pct_verbs, pct_nouns, pct_adj

# Apply the function to calculate features for each review
features_df = dat['review_text'].apply(lambda x: calculate_features(x))
features_df = pd.DataFrame(features_df.tolist(), columns=["num_words", "avg_word_len", "avg_sent_len", "pct_verbs", "pct_nouns", "pct_adj"])

# Concatenate the new features DataFrame with the original features
X = pd.concat([X, features_df], axis=1)

# Write the DataFrame to a CSV file
data_with_target.to_csv('embed_25.csv', index=False)