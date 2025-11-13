import logging
import re
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any


class PineconeSemanticSearch:
    """
    A modular class for managing semantic embeddings in Pinecone.
    Supports inserting new documents and querying similar texts.
    """

    # -------------------------------------------------------
    # Initialization
    # -------------------------------------------------------
    def __init__(
                    self, 
                    pinecone_index_name,
                    pinecone_api_key,
                    embed_model_name
                ):

        # Environment variables
        self.pinecone_api_key = pinecone_api_key
        self.embed_model_name = embed_model_name
        self.pinecone_index_name = pinecone_index_name

        # Initialize Pinecone client (new API)
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(self.pinecone_index_name)

        # Load model once
        self.model = SentenceTransformer(self.embed_model_name)
        logging.info(f"PineconeSemanticSearch initialized for index: {self.pinecone_index_name}")


    # -------------------------------------------------------
    # Static Method: Normalize text
    # -------------------------------------------------------
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize user text before embedding."""
        return str(text).strip().lower()


    # -------------------------------------------------------
    # Query Pinecone
    # -------------------------------------------------------
    def query_pinecone(self, user_query: str, filter_condition: int = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search on Pinecone."""
        if not user_query or not user_query.strip():
            logging.warning("Empty query.")
            return []

        logging.info(f"Querying: '{user_query}' (top_k={top_k})")

        query_vector = self.model.encode(self.normalize_text(user_query)).tolist()

        try:

            if filter_condition is not None and filter_condition >= 1 and filter_condition <= 5:

                if ("מ־" in user_query or "מעל" in user_query
                        or "גבוה מ" in user_query
                        or "מעל ל" in user_query):

                    # Example: filter by 'level' metadata field
                    response = self.index.query(
                        vector=query_vector,
                        top_k=top_k,
                        include_metadata=True,
                        filter={"level": {"$gt": filter_condition}}
                        )
                elif "שווים ל" in user_query or "שווה ל" in user_query:
                    # Example: filter by 'level' metadata field
                    response = self.index.query(
                        vector=query_vector,
                        top_k=top_k,
                        include_metadata=True,
                        filter={"level": {"$eq": filter_condition}}
                        )

                elif "בין" in user_query:
                    # Example: filter by 'level' metadata field
                    response = self.index.query(
                        vector=query_vector,
                        top_k=top_k,
                        include_metadata=True,
                        filter={"level": {"$gte": filter_condition - 1, "$lte": filter_condition + 1}}  # 👈 Filter condition
                        )

                elif ("עד" in user_query or "עד ל" in user_query 
                        or "נמוך מ" in user_query 
                        or "פחות מ" in user_query 
                        or "מתחת" in user_query or "מתחת ל" in user_query):
                # Example: filter by 'level' metadata field
                    response = self.index.query(
                        vector=query_vector,
                        top_k=top_k,
                        include_metadata=True,
                        filter={"level": {"$eq": filter_condition}}
                        )
                
            elif (
                "נמוך" in user_query 
                or "רמה נמוכה" in user_query
                or "ציון נמוך" in user_query
                or "נמוכה" in user_query
                or "לא מרוצים" in user_query
                or "לא מרוצה" in user_query
                or "בעיות שירות" in user_query
                or "שירות לקוי" in user_query
                or "שירות גרוע" in user_query
                or "לא טוב" in user_query
                or "לא טובה" in user_query
                or "תלונות" in user_query
                or "תלונה" in user_query):

                response = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True,
                    filter={"level": {"$lt": 3}}
                    )
            elif (
                "גבוה" in user_query
                or "רמה גבוהה" in user_query
                or "מרוצים" in user_query
                or "מרוצה" in user_query
                or "שירות טוב" in user_query
                or "שירות מצוין" in user_query
                or "שירות מעולה" in user_query
                or "שירות נהדר" in user_query
                or "שירות מצטיין" in user_query
                or "חוויית שירות טובה" in user_query
                or "חווית שירות טובה" in user_query
                or "מצוין" in user_query
                or "מעולה" in user_query
                or "נהדר" in user_query):

                response = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True,
                    filter={"level": {"$gt": 3}}  
                    )
            else:
                response = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)

            matches = response.get("matches", [])
            logging.info(f"Found {len(matches)} matches.")
            return matches
        
        except Exception as e:
            logging.error(f"Query failed: {e}")
            return []


    def extract_number(self, user_query: str):
        """
        Extract the first number (digit) appearing in the query.
        Works with Hebrew queries like 'מ־3', 'מעל 4', etc.
        Returns int or None.
        """
        
        # Normalize various Hebrew dash characters and spacing
        user_query = user_query.replace("־", "-").replace("–", "-").replace("מ-", " ").replace("מ־", " ")

        # Regular expression: find one or more digits
        match = re.search(r"\d+", user_query)
        if match:
            return int(match.group(0))
        return None


    # -------------------------------------------------------
    # Print Results Helper
    # -------------------------------------------------------
    @staticmethod
    def print_results(matches: List[Dict[str, Any]]):
        """Pretty print Pinecone search results."""
        if not matches:
            print("No matches found.")
            return

        print("Top Semantic Matches:")
        for match in matches:
            meta = match.get("metadata", {})
            print(f"• ID: {match.get('id')}")
            print(f"  Score: {match.get('score', 0):.4f}")
            print(f"  Text: {meta.get('text', '')[:120]}...\n")



    @staticmethod
    def get_context(matches: List[Dict[str, Any]]):
        """Pretty print Pinecone search results."""
        if not matches:
            print("No matches found.")
            return
        else:
            context_texts = []
            for match in matches:
                meta = match.get("metadata", {})
                text = meta.get("text") or meta.get("chunk_text") or "No text available"
                context_texts.append(f"• {text}")
            context = "\n".join(context_texts)
        return context
