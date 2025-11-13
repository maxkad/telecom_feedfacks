import os, sys
import logging
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from services.vector_db_retriever import PineconeSemanticSearch
from services.rationl_db_retriever import SQLiteRationalDBSearch


class HybridSearch:
    # -------------------------------------------------------
    # Initialization
    # -------------------------------------------------------
    def __init__(
                    self, 
                    db_path, 
                    pinecone_index_name,
                    pinecone_api_key,
                    embed_model_name,
                    openai_api_key
                ):

        # Environment variables
        self.db_path = db_path
        self.pinecone_index_name = pinecone_index_name

        self.pinecone_api_key = pinecone_api_key
        self.embed_model_name = embed_model_name
        self.openai_api_key = openai_api_key
        self.conn = sqlite3.connect(self.db_path)
        self.client = OpenAI(
             api_key = self.openai_api_key
        )

        # Initialize Pinecone client (new API)
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        # Load model once
        self.model = SentenceTransformer(self.embed_model_name)

        self.search_engine = SQLiteRationalDBSearch()

        # Ensure index exists
        if self.index_name not in [idx.name for idx in self.pc.list_indexes()]:
            logging.info(f"Creating Pinecone index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # matches MiniLM embedding dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        self.index = self.pc.Index(self.index_name)

        logging.info(f"HybridSearch initialized for DB: {self.db_path}")



    def query_hybrid(self, user_query: str) -> dict:
        """
        Combine structured filter + semantic meaning.
        Example: filter 'Level < 3' then find main topics semantically.
        """
        # 1. Get relevant subset from SQLite (filter by Level < 3)
        # filtered_df = pd.read_sql_query(f"SELECT * FROM {DB_FEEDBACKS_TBL} WHERE Level < 3", self.conn)
        # self.conn.close()

        df = self.search_engine.query_rational_db(user_query)

        # 2. Semantic summarization (example using embeddings mean
        embeddings = self.model.encode(df["text"].tolist())

        # 3. Compute cluster centers or dominant topic
        mean_vector = np.mean(embeddings, axis=0)
        res = self.index.query(vector=mean_vector.tolist(), top_k=5, include_metadata=True)
        return {
            "structured_results": df.to_dict(orient="records"),
            "semantic_topics": [match["metadata"] for match in res["matches"]]
        }
