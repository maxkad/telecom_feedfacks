import os, sys
import json
from openai import OpenAI

# Add project root (the parent of /rag/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from services.vector_db_retriever import PineconeSemanticSearch
from services.rationl_db_retriever import SQLiteRationalDBSearch
from services.hybrid_retriever import HybridSearch
from services.openai_get_response import OpenAIGetQueryResponse

from dotenv import load_dotenv


SYSTEM_PROMPT = """
You are a classifier that determines whether a user's query should be answered using:
1. RATIONAL_DB — for structured, numeric, date, aggregations, filters, or analytics questions.
2. VECTOR_DB — for semantic or meaning-based, text meaning, similarity questions.
3. HYBRID — for queries that require both structured filtering and semantic analysis.

Examples:
1. "מה הנושא המרכזי של המשובים שקיבלו ציון נמוך מ־3?" → RATIONAL_DB
2. "מה המשתמשים חושבים על השירות?" → VECTOR_DB
3. "מה הנושאים המרכזיים של המשובים על השירותים עם ציון נמוך?" → HYBRID
4. "כמה משובים התקבלו היום?" → RATIONAL_DB
5. "תמצא משובים דומים למשוב הזה: השירות היה איטי מאוד" → VECTOR_DB

When you classify, provide your output in the following JSON format:
{
  "intent": "RATIONAL_DB" | "VECTOR_DB" | "HYBRID",
  "confidence": {
    "DRATIONAL_DB": float,
    "VECTOR_DB": float,
    "HYBRID": float
  },
  "reasoning": "brief explanation"
}
Probabilities should sum to 1. Respond only with JSON.
"""

class QueryRouter:
    def __init__(self):
        load_dotenv()

        # Environment variables
        self.db_path = os.getenv("DB_FILE")
        self.db_feedbacks_tbl = os.getenv("DB_FEEDBACKS_TBL")
        self.db_topics_tbl = os.getenv("DB_TOPICS_TBL")

        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.embed_model_name = os.getenv("EMBED_MODEL_NAME")
        self.model_name_for_tokenizer = os.getenv("MODEL_NAME_FOR_TOKENIZER")

        self.model = os.getenv("OPENAI_GPT_4O_MINI_DEPLOYMENT_NAME")
        self.openai_api_key = os.getenv("OPENAI_GPT_4O_API_KEY")
        self.client = OpenAI(
            api_key = os.getenv(self.openai_api_key)
        )

        self.vector_db_retriever = PineconeSemanticSearch(
            self.pinecone_index_name,
            self.pinecone_api_key,
            self.embed_model_name
        )

        self.rational_db_retriever = SQLiteRationalDBSearch(
            self.db_path,
            self.db_feedbacks_tbl,
            self.db_topics_tbl,
            self.openai_api_key,
            self.model
        )
        self.hybrid_retriever = HybridSearch(
            self.db_path, 
            self.pinecone_index_name,
            self.pinecone_api_key,
            self.embed_model_name,
            self.openai_api_key
        )


    def classify_intent(self, user_query: str) -> str:
        
        print("Starting intent classification...")
        user_prompt = f"User query: {user_query}"

        response = self.client.chat.completions.create(
            model=self.model_name_for_tokenizer,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        content = response.choices[0].message.content.strip()

        try:
            result = json.loads(content)
            # pick the max probability
            top_intent = max(result["confidence"], key=result["confidence"].get)
            result["selected_intent"] = top_intent
            print(f"Classified intent: {top_intent} with confidence {result['confidence'][top_intent]:.2f}")
            return result
        except Exception:
            return {"intent": "UNKNOWN", "confidence": {}, "reasoning": "Invalid JSON output"}



    def handle_user_query(self, user_query):
        classification = self.classify_intent(user_query)
        intent = classification["selected_intent"]
        print(f"Intent Detected: {intent} (Confidence: {classification['confidence'][intent]:.2f})")
        results = None

        if intent == "RATIONAL_DB":
            df = self.rational_db_retriever.query_rational_db(user_query)
            # Handle large textual results
            if len(df) > 100 and "text" in df.columns:
                print("Large text result detected — summarizing common subjects...")
                summary = self.rational_db_retrieve.summarize_texts_llm(df['text'].tolist())
                results = {"records": len(df), "summary": summary}

        elif intent == "VECTOR_DB":
            results = self.vector_db_retriever.query_pinecone(user_query, top_k=2)
        elif intent == "HYBRID":
            results = self.hybrid_retriever.query_hybrid(user_query)
        else:
            results = {"error": "Unable to classify intent"}

        context = PineconeSemanticSearch.get_context(results)
        return OpenAIGetQueryResponse.retrieve_answer(user_query, context)
        