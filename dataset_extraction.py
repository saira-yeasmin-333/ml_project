import gzip
import shutil

# Set the path to your gzipped file
gz_file_path = "goodreads_books.json.gz"

# Set the destination path for the uncompressed file
uncompressed_file_path = "goodreads_books.json"

# Open the gzipped file and write the uncompressed content
with gzip.open(gz_file_path, 'rb') as gz_file:
    with open(uncompressed_file_path, 'wb') as uncompressed_file:
        shutil.copyfileobj(gz_file, uncompressed_file)

print(f"Unzipped {gz_file_path} to {uncompressed_file_path}")