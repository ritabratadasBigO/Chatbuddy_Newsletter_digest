import os
import requests
import pickle
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Config
BASE_URL = "https://niteowl1986.github.io/Daily-News-Feed/"
OUTPUT_DIR = "newsbot_data"
INDEX_OUTPUT = os.path.join(OUTPUT_DIR, "newsbot_faiss.index")
DOCS_OUTPUT = os.path.join(OUTPUT_DIR, "newsbot_docs.pkl")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prepare date range
start_date = datetime(2024, 10, 1)
end_date = datetime.today()

# Load embedding model
model = SentenceTransformer(EMBEDDING_MODEL)

# Load existing documents if available
existing_docs = []
if os.path.exists(DOCS_OUTPUT):
    with open(DOCS_OUTPUT, "rb") as f:
        existing_docs = pickle.load(f)

existing_dates = set(doc[0] for doc in existing_docs)
documents = list(existing_docs)
dates_processed = 0

while start_date <= end_date:
    formatted_date = start_date.strftime('%d%b%Y')
    if formatted_date in existing_dates:
        start_date += timedelta(days=1)
        continue

    html_filename = f"daily_news_feed_{formatted_date}.html"
    file_url = f"{BASE_URL}{html_filename}"

    try:
        response = requests.get(file_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            entries = soup.find_all(["li", "p", "div"])

            for entry in entries:
                text = entry.get_text(strip=True)
                if len(text.split()) > 10:
                    documents.append((formatted_date, text))
            print(f"✅ Processed: {html_filename}")
        else:
            print(f"⚠️ Skipped (not found): {html_filename}")
    except Exception as e:
        print(f"❌ Error for {html_filename}: {e}")

    start_date += timedelta(days=1)
    dates_processed += 1

print(f"\n🔎 Total documents collected: {len(documents)} from {dates_processed} new days")

# Generate embeddings only for new documents
texts_for_embedding = [doc[1] for doc in documents]
embeddings = model.encode(texts_for_embedding, show_progress_bar=True)

# Save FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))
faiss.write_index(index, INDEX_OUTPUT)

# Save document texts with date
with open(DOCS_OUTPUT, "wb") as f:
    pickle.dump(documents, f)

print("\n✅ FAISS index and document texts saved with date tracking.")