import os
import requests
import pickle
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re
import subprocess

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

# Config
BASE_URL = "https://niteowl1986.github.io/Daily-News-Feed/"
OUTPUT_DIR = "newsbot_data"
INDEX_OUTPUT = os.path.join(OUTPUT_DIR, "newsbot_faiss.index")
DOCS_OUTPUT = "newsbot_data/newsbot_docs.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
github_pat = os.getenv("GITHUB_PAT")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Start date discovery from GitHub filenames
try:
    response = requests.get("https://niteowl1986.github.io/Daily-News-Feed/")
    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all("a", href=True)
    date_strings = []

    for link in links:
        match = re.search(r"daily_news_feed_(\d{2}[A-Za-z]{3}\d{4})\.html", link["href"])
        if match:
            date_strings.append(match.group(1))

    if date_strings:
        latest_date_str = max(date_strings)
        start_date = datetime.strptime(latest_date_str, '%d%b%Y') + timedelta(days=1)
    else:
        start_date = datetime(2024, 10, 1)
except Exception as e:
    print(f"⚠️ Could not fetch or parse GitHub page: {e}")
    start_date = datetime(2024, 10, 1)

# Load existing docs
if os.path.exists(DOCS_OUTPUT):
    with open(DOCS_OUTPUT, "rb") as f:
        existing_docs = pickle.load(f)
else:
    existing_docs = []

end_date = datetime.today()

documents = list(existing_docs)
existing_dates = set(doc[0] for doc in existing_docs)
dates_processed = 0

# Load embedding model
model = SentenceTransformer(EMBEDDING_MODEL)

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
new_documents = [doc for doc in documents if doc[0] not in existing_dates]
texts_for_embedding = [doc[1] for doc in new_documents]
embeddings = model.encode(texts_for_embedding, show_progress_bar=True)

if embeddings.size > 0:
    dimension = embeddings.shape[1]
    if os.path.exists(INDEX_OUTPUT):
        index = faiss.read_index(INDEX_OUTPUT)
    else:
        index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings).astype('float32'))
    faiss.write_index(index, INDEX_OUTPUT)

    # Append only new documents to existing_docs and save
    documents = existing_docs + new_documents

    with open(DOCS_OUTPUT, "wb") as f:
        pickle.dump(documents, f)

    print("\n✅ FAISS index and document texts saved with date tracking.")
else:
    print("\n⚠️ No new documents to embed or index.")

# REPO_URL = f"https://{GITHUB_PAT}@github.com/niteowl1986/Chatbuddy_Newsletter_digest.git" 

def git_commit_and_push():
    if not github_pat:
        print("❌ GITHUB_PAT not set in environment.")
        return

    try:
        # Ensure repo has a remote named origin; if not, add it
        result = subprocess.run(["git", "remote"], capture_output=True, text=True)
        remotes = result.stdout.strip().split("\n")
        if "origin" not in remotes:
            subprocess.run([
                "git", "remote", "add", "origin",
                f"https://{github_pat}@github.com/niteowl1986/Chatbuddy_Newsletter_digest.git"
            ], check=True)
        else:
            subprocess.run([
                "git", "remote", "set-url", "origin",
                f"https://{github_pat}@github.com/niteowl1986/Chatbuddy_Newsletter_digest.git"
            ], check=True)

        subprocess.run(["git", "add", "newsbot_data/newsbot_docs.pkl", "newsbot_data/newsbot_faiss.index"], check=True)

        # Only commit if there are staged changes
        result = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if result.returncode == 0:
            subprocess.run(["git", "status"])
            print("ℹ️ Nothing new to commit.")
            return

        subprocess.run(["git", "commit", "-m", "Automated daily update of FAISS index"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        subprocess.run(["git", "status"])
        print("✅ Changes pushed to GitHub.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Git operation failed: {e}")
git_commit_and_push()