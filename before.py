import pandas as pd

# read review data
dat = pd.read_csv("filtered_reviews.csv")
dat = dat.drop(columns=['book_id','ratings_count','review_likes','like_share'])

# difference between user rating and average book rating
dat["rating_diff"] = dat["user_rating"]-dat["avg_rating"]
dat = dat.drop(columns=['avg_rating'])

# flag if review contains a quotation
dat["quote"] = dat["review_text"].str.contains("\"")

print(dat.columns)

# Check for null values in a specific column (e.g., 'review_text')
print(dat['review_text'].isnull().sum())

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.eval()

# Drop rows with missing 'review_text'
print("Length before dropping null values:", len(dat))
dat = dat.dropna(subset=['review_text']).reset_index(drop=True)
print("Length after dropping null values:", len(dat))

# Reindex the DataFrame
dat.reset_index(drop=True, inplace=True)

# Tokenize and encode the reviews
max_sequence_length = 256  # Adjust as needed
def generate_tokenized_reviews(data):
    for review_text in data['review_text']:
        yield tokenizer.encode(review_text, add_special_tokens=True, max_length=max_sequence_length, truncation=True)

tokenized_reviews = generate_tokenized_reviews(dat)

# Pad sequences to the same length
max_len = max(map(len, tokenized_reviews))
padded_reviews = [i + [0] * (max_len - len(i)) for i in tokenized_reviews]

# Convert to PyTorch tensors
input_ids = torch.tensor(padded_reviews)
attention_masks = torch.where(input_ids != 0, torch.tensor(1), torch.tensor(0))

# Create DataLoader
dataset = TensorDataset(input_ids, attention_masks)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# Function to get sentiment predictions
def get_sentiment_predictions(model, dataloader):
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_masks = batch
            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predictions.extend(probabilities[:, 1].tolist())  # Assuming index 1 corresponds to positive sentiment
    return predictions

# Get sentiment predictions for the dataset
sentiment_predictions = get_sentiment_predictions(model, dataloader)

# Add sentiment predictions to the DataFrame
# Add sentiment predictions to the DataFrame as a new column
sentiment_series = pd.Series(sentiment_predictions, name='new_sentiment_probabilities')
dat = pd.concat([dat, sentiment_series], axis=1)

# Add sentiment predictions to the DataFrame as a new column
sentiment_series = pd.Series(sentiment_predictions, name='new_sentiment_probabilities')
dat = pd.concat([dat, sentiment_series], axis=1)

# Print the first few rows of the DataFrame with the new column
print(dat.head())

# Print information about the DataFrame
print(dat.info())

# Check for NaN values in the DataFrame
print("NaN values in the DataFrame:")
print(dat.isnull().sum())

# Print the first few rows of the DataFrame with the new column
print(dat.head(10))

