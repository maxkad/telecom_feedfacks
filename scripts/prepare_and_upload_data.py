import os
import sys
import uuid
import sqlite3
import logging
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import tiktoken


# ============================================================
# CONFIGURE LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(funcName)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_GPT_4O_API_KEY")
openai_mini_api_key = os.getenv("OPENAI_GPT_4O_MINI_DEPLOYMENT_NAME")

# Set the option to display all columns
pd.options.display.max_columns = None

# ============================================================
# CONFIGURATION CONSTANTS
# ============================================================
INDEX_NAME = "telecom-feedback-texts-idx"
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_DIM = 384
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
MODEL_NAME_FOR_TOKENIZER = "gpt-4o-mini"
DB_FILE = "telecom_db.db"

# PATH_TO_DATA_SOURCE = "/workspaces/telecom_feedfacks/data/sample.csv"
PATH_TO_DATA_SOURCE = "/workspaces/telecom_feedfacks/data/Feedback.csv"

pc = Pinecone(api_key=pinecone_api_key)


# ============================================================
# DATABASE SETUP
# ============================================================

def create_db_and_tables():
    """Creates the SQLite database and feedback/topics tables if not exist."""
    logging.info("Starting: create_db_and_tables()")
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # Create main feedback table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS feedbacks_tbl (
            id TEXT,
            service_name TEXT,
            level INTEGER,
            text TEXT,
            reference_number TEXT,
            request_id TEXT,
            process_id TEXT,
            creation_date TEXT,
            chunk_id TEXT,
            chunk_text TEXT,
            topic_id INTEGER
        )
    ''')
    conn.execute("CREATE INDEX IF NOT EXISTS idx_level ON feedbacks_tbl(level)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_service ON feedbacks_tbl(service_name)")

    # Create topics summary table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS topics_tbl (
            topic_id INTEGER PRIMARY KEY,
            keywords TEXT,
            num_feedbacks INTEGER,
            avg_level REAL,
            summary TEXT
        )
    ''')

    conn.commit()
    conn.close()
    logging.info("Completed: create_db_and_tables()")


def delete_db_and_tables():
    """Deletes the SQLite tables if they already exist."""
    logging.info("Starting: delete_db_and_tables()")
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS feedbacks_tbl;")
    cur.execute("DROP TABLE IF EXISTS topics_tbl;")
    conn.commit()
    conn.close()
    logging.info("Completed: delete_db_and_tables()")


# ============================================================
# DATA LOADING
# ============================================================

def load_and_process_data() -> pd.DataFrame:
    """Loads the feedback CSV file and returns a cleaned DataFrame."""
    logging.info("Starting: load_and_process_data()")
    # df = pd.read_csv(PATH_TO_DATA_SOURCE)
    # df = pd.read_csv(PATH_TO_DATA_SOURCE, encoding='"utf-8"')
    df = pd.read_csv(PATH_TO_DATA_SOURCE, encoding='utf-8-sig')

    df.columns = ['id', 'service_name', 'level', 'text', 'reference_number', 'request_id', 'process_id', 'creation_date']
    df.fillna("unknown", inplace=True)
    logging.info(f"Loaded {len(df)} feedback records.")
    logging.info("Completed: load_and_process_data().")
    return df


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Splits text into overlapping chunks using GPT tokenizer."""
    tokenizer = tiktoken.encoding_for_model(MODEL_NAME_FOR_TOKENIZER)
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks


# ============================================================
# EMBEDDING + PINECONE UPSERT
# ============================================================

def create_pinecone_index():
    """Creates Pinecone index if not already exists."""
    logging.info("Starting: create_pinecone_index()")
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        logging.info(f"Created Pinecone index: {INDEX_NAME}")
    else:
        logging.info(f"Index '{INDEX_NAME}' already exists.")
    logging.info("Completed: create_pinecone_index().")



def truncate_pinecone_index(index_name=INDEX_NAME):
    """
    Truncate Pinecone index only if it has vectors.
    """
    logging.info(f"Starting: truncate_pinecone_index('{index_name}')")

    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    total_vectors = stats.total_vector_count
    logging.info(f"Index '{index_name}' contains {total_vectors} vectors.")

    if total_vectors > 0:
        logging.info(f"Deleting all vectors from '{index_name}'...")
        index.delete(delete_all=True)
        logging.info(f"All vectors deleted from '{index_name}'.")
    else:
        logging.info(f"Index '{index_name}' is already empty. No action taken.")
    logging.info(f"Completed: truncate_pinecone_index('{index_name}')")




def generate_and_store_chunk(df_feedbacks: pd.DataFrame) -> pd.DataFrame:
    """Generates embeddings for feedbacks and stores them in Pinecone."""
    logging.info("Starting: generate_and_store_chunk()")
    model = SentenceTransformer(model_name_or_path=EMBED_MODEL_NAME)
    index = pc.Index(INDEX_NAME)
    to_upsert = []

    for idx, row in df_feedbacks.iterrows():

        # print(f"Processing feedback ID: {row['id']}")
        # print(f"Text: {row['text'][:50]}...")
        chunks = chunk_text(str(row["text"]))
        for chunk in chunks:

            # print(f"  Chunk: {chunk[:50]}...")
            vector = model.encode(chunk).tolist()
            chunk_id = str(uuid.uuid4())

            # Update DataFrame
            df_feedbacks.at[idx, "chunk_id"] = chunk_id
            df_feedbacks.at[idx, "chunk_text"] = chunk

            meta = {
                "service_name": row["service_name"],
                "level": row["level"],
                "reference_number": row["reference_number"],
                "request_id": row["request_id"],
                "process_id": row["process_id"],
                "text": row["text"],
                "creation_date": row["creation_date"]
            }
            to_upsert.append((chunk_id, vector, meta))

            # print(f"Prepared chunk ID: {chunk_id} for upsert.") 
            # print(f"Vector length: {len(vector)}")
            # print(f"Vector length: {vector[:5]}...")
            # print(f"Metadata: {meta}")
            # print("-----")
            # for idx, item in enumerate(to_upsert[-3:]):
            #     # print(f"To upsert [{idx}]: ID={item[0]}, Vector length={len(item[1])}, Metadata keys={list(item[2].keys())}")
            #     print(f"To upsert [{idx}]: ID={item[0]}, Vector={item[1]}, Metadata keys={list(item[2].values())}")


    # Upsert to Pinecone in batches
    BATCH_SIZE = 100
    for i in range(0, len(to_upsert), BATCH_SIZE):
        batch = to_upsert[i:i + BATCH_SIZE]
        # print(f"Upserting batch {i // BATCH_SIZE + 1} with {len(batch)} records...")
        index.upsert(vectors=batch)
        logging.info(f"Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} records)")

    logging.info("Completed: generate_and_store_chunk().")
    return df_feedbacks


# ============================================================
# TOPIC GENERATION
# ============================================================

def generate_topics(df_feedbacks: pd.DataFrame):
    """Clusters feedbacks into topics and generates summary per topic."""
    logging.info("Starting: generate_topics()")
    model = SentenceTransformer(model_name_or_path=EMBED_MODEL_NAME)
    embeddings = model.encode(df_feedbacks["text"].tolist(), show_progress_bar=False)

    num_topics = 4
    kmeans = KMeans(n_clusters=num_topics, random_state=42)
    df_feedbacks["topic_id"] = kmeans.fit_predict(embeddings)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_feedbacks["text"])
    terms = vectorizer.get_feature_names_out()

    topic_keywords = {}
    for topic in range(num_topics):
        topic_indices = np.where(df_feedbacks["topic_id"] == topic)
        topic_texts = X[topic_indices]
        if topic_texts.shape[0] == 0:
            continue
        mean_tfidf = np.asarray(topic_texts.mean(axis=0)).ravel()
        top_indices = mean_tfidf.argsort()[-5:][::-1]
        topic_keywords[topic] = ", ".join([terms[i] for i in top_indices])

    topics = []
    for topic in range(num_topics):
        subset = df_feedbacks[df_feedbacks["topic_id"] == topic]
        keywords = topic_keywords.get(topic, "")
        avg_level = subset["level"].mean()
        topics.append({
            "topic_id": topic,
            "keywords": keywords,
            "num_feedbacks": len(subset),
            "avg_level": round(avg_level, 2),
            "summary": f"נושא כללי הקשור ל-{keywords.split(',')[0] if keywords else 'משוב'}"
        })

    df_topics = pd.DataFrame(topics)
    logging.info("Completed: generate_topics().")
    return df_feedbacks, df_topics


# ============================================================
# INSERT INTO DATABASE
# ============================================================

def insert_feedbacks_and_topics(df_feedbacks: pd.DataFrame, df_topics: pd.DataFrame):
    """Inserts feedbacks and topics into SQLite database."""
    logging.info("Starting: insert_feedbacks_and_topics()")
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    for _, row in df_feedbacks.iterrows():
        cur.execute('''
            INSERT INTO feedbacks_tbl (id, service_name, level, text, reference_number, request_id, process_id, creation_date, topic_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row['id'], row['service_name'], row['level'], row['text'], row['reference_number'],
            row['request_id'], row['process_id'], row['creation_date'], row['topic_id']
        ))

    for _, row in df_topics.iterrows():
        cur.execute('''
            INSERT INTO topics_tbl (topic_id, keywords, num_feedbacks, avg_level, summary)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            row['topic_id'], row['keywords'], row['num_feedbacks'], row['avg_level'], row['summary']
        ))

    conn.commit()
    conn.close()
    logging.info("Completed: insert_feedbacks_and_topics()")


# ============================================================
# MAIN PIPELINE
# ============================================================

if __name__ == "__main__":
    logging.info("=== PIPELINE START ===")
    truncate_pinecone_index()
    create_pinecone_index()
    delete_db_and_tables()
    create_db_and_tables()
    df_feedbacks = load_and_process_data()
    df_feedbacks = generate_and_store_chunk(df_feedbacks)
    df_feedbacks, df_topics = generate_topics(df_feedbacks)
    insert_feedbacks_and_topics(df_feedbacks, df_topics)
    logging.info("=== PIPELINE END ===")

    print("\n Topics Preview:")
    print(df_topics.head())
    print("\n Feedbacks Preview:")
    print(df_feedbacks.head())
