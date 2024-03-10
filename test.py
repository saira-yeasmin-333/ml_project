import pandas as pd

df = pd.read_csv('filtered_reviews.csv')

# Filter the DataFrame for 'popular' is 0 and 1
df_popular_0 = df[df['popular'] == 0]
df_popular_1 = df[df['popular'] == 1]

# Calculate the number of rows for each category
num_rows_total = 5000
num_rows_positive = int(num_rows_total * 0.5)  # 45% of 10000
num_rows_negative = num_rows_total - num_rows_positive  # 55% of 10000

# Sample the rows randomly
sampled_positive = df_popular_1.sample(n=num_rows_positive, random_state=42)
sampled_negative = df_popular_0.sample(n=num_rows_negative, random_state=42)

# Concatenate the sampled DataFrames
balanced_df = pd.concat([sampled_positive, sampled_negative]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total rows in the balanced DataFrame: {len(balanced_df)}")
print(f"Number of positive rows: {len(balanced_df[balanced_df['popular'] == 1])}")
print(f"Number of negative rows: {len(balanced_df[balanced_df['popular'] == 0])}")
balanced_df.to_csv('balanced_5000_reviews_50%.csv', index=False)

# # Extract the first 10 rows
# first_10_rows = df.head(25)

# # Save the first 10 rows into a new CSV file
# first_10_rows.to_csv('first_25_reviews.csv', index=False)
