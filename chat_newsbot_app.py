import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
from io import BytesIO
import os
from datetime import datetime, timedelta
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the environment variables from the api_keys.txt file
with open('api_keys.txt', 'r') as file:
    for line in file:
        if 'groq_key' in line:
            groq_api_key = line.split('=')[1].strip()
            # Remove single quotes from the groq_api_key
            groq_api_key = groq_api_key.replace("'", '')
        
# --- Configuration ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "newsbot_data/newsbot_faiss.index"
DOCS_PATH = "newsbot_data/newsbot_docs.pkl"
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

LLM_MODEL = "llama3-8b-8192"

# --- Load resources ---
st.set_page_config(page_title="NewsBot Chat", layout="wide")
st.title("🧠 Chat with Dhurin News Summaries")

@st.cache_resource
def load_faiss():
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)
    return index, docs

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

index, documents = load_faiss()
embedder = load_embedder()
groq_client = Groq(api_key=GROQ_API_KEY)

# --- Helper to search similar documents ---
def get_top_k_docs(question, k=5):
    question_vec = embedder.encode([question])
    D, I = index.search(np.array(question_vec).astype("float32"), k)
    return [(documents[i][0], documents[i][1]) for i in I[0]]

import re
from calendar import month_abbr

def extract_target_month(query):
    match = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s?(\d{4})?", query.lower())
    if match:
        mon = match.group(1).capitalize()
        year = match.group(2) if match.group(2) else None
        return f"{mon}{year}" if year else mon
    return None

def extract_relative_date_range(query):
    query = query.lower()
    today = datetime.today()

    if "last week" in query:
        return today - timedelta(days=7), today
    elif "last 2 weeks" in query or "past 2 weeks" in query:
        return today - timedelta(days=14), today
    elif "last month" in query or "past month" in query:
        return today - timedelta(days=30), today
    elif "last 3 months" in query or "past 3 months" in query:
        return today - timedelta(days=90), today
    return None, None

# --- Streamlit interface ---
user_question = st.text_input("💬 Ask a question about recent news:")

# --- Add voice input ---
st.markdown("🎙️ Or record your voice below:")
audio_bytes = audio_recorder()
voice_question = ""

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    recognizer = sr.Recognizer()
    try:
        audio_stream = BytesIO(audio_bytes)
        with sr.AudioFile(audio_stream) as source:
            audio_data = recognizer.record(source)
            voice_question = recognizer.recognize_google(audio_data)
            st.success(f"You said: {voice_question}")
    except sr.UnknownValueError:
        st.warning("Could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"Speech recognition failed: {e}")
    except Exception as e:
        st.error(f"Audio processing failed: {e}")

# --- Choose between text or voice input
final_question = user_question if user_question else voice_question

target_month = extract_target_month(final_question)

# --- Chatbot logic ---
if final_question:
    with st.spinner("Thinking..."):
        top_docs = get_top_k_docs(final_question)
        filtered_docs = top_docs

        # Time range filter (relative dates)
        start_date, end_date = extract_relative_date_range(final_question)
        if start_date and end_date:
            def parse_doc_date(date_str):
                try:
                    return datetime.strptime(date_str.strip(), "%d%b%Y")
                except Exception:
                    return None

            filtered_docs = [
                doc for doc in filtered_docs
                if (parsed := parse_doc_date(doc[0])) and start_date <= parsed <= end_date
            ]

            if not filtered_docs:
                st.warning(f"No news summaries found in the requested time range.")
                st.stop()

        # Month name filter
        if target_month:
            filtered_docs = [doc for doc in filtered_docs if target_month in doc[0]]

            if not filtered_docs:
                st.warning(f"No relevant news found for {target_month}. Try another question.")
                st.stop()

        context = "\n\n".join([f"Date: {doc[0]}\nSummary: {doc[1]}" for doc in filtered_docs])

        summarization_prompt = f"""
        Based on the following news summaries, provide a 2-3 line summary that captures the main themes, developments, or notable events:

        {context}

        Question: {final_question}
        """

        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the following news summaries to respond."},
                {"role": "user", "content": summarization_prompt}
            ],
            temperature=0.7
        )
        answer = response.choices[0].message.content

    st.markdown("### 🧠 Summary Answer:")
    st.markdown(answer)

    with st.expander("🔍 Context used from articles"):
        for i, doc in enumerate(filtered_docs):
            st.markdown(f"**Doc {i+1} ({doc[0]}):** {doc[1]}")