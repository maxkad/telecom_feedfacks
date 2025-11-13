import logging
from openai import OpenAI
import sqlite3
import pandas as pd


class SQLiteRationalDBSearch:

    # -------------------------------------------------------
    # Initialization
    # -------------------------------------------------------
    def __init__(
                    self,
                    db_path,
                    db_feedbacks_tbl,
                    db_topics_tbl,
                    openai_api_key,
                    model   
                ):

        # Environment variables
        self.db_path = db_path
        self.db_feedbacks_tbl = db_feedbacks_tbl
        self.db_topics_tbl = db_topics_tbl
        self.conn = sqlite3.connect(self.db_path)

        self.openai_api_key = openai_api_key
        self.client = OpenAI(
             api_key = self.openai_api_key
        )
        self.model = model
        logging.info(f"SQLiteRationalDBSearch initialized for DB: {self.db_path}")


    def query_rational_db(self, user_query: str) -> pd.DataFrame:
        """
        Handle structured queries.
        Example: 'What is the average rating per service?'
        """

        # Example pattern matching
        if "ממוצע" in user_query:
            sql = f"SELECT service_name, AVG(level) as avg_score FROM {self.db_feedbacks_tbl} GROUP BY service_name;"
        elif "גבוה" in user_query:
            sql = f"SELECT * FROM {self.db_feedbacks_tbl}s WHERE level >= 4;"
        elif "נמוך" in user_query or "נמוכה" in user_query:
            sql = f"SELECT * FROM {self.db_feedbacks_tbl} WHERE level < 3;"
        elif "נושא" in user_query or "topic" in user_query:
            sql = f"""
            SELECT t.topic_id, t.keywords, t.avg_level, COUNT(f.id) AS feedback_count
            FROM {self.db_feedbacks_tbl} f
            JOIN {self.db_topics_tbl} t ON f.topic_id = t.topic_id
            GROUP BY t.topic_id, t.keywords, t.avg_level
            ORDER BY t.avg_level ASC;
            """ 
        else:
            sql = f"SELECT * FROM {self.db_feedbacks_tbl} LIMIT 5"

        df = pd.read_sql_query(sql, self.conn)
        self.conn.close()
        return df
    

    def summarize_texts_llm(self, texts):
        """
        Summarize and extract key themes from a list of text feedbacks.
        """

        joined_text = "\n".join(texts[:200])  # Limit for context
        prompt = f"""
            You are an analyst that finds common themes in customer feedback.
            Your goal is to understand customer feedback and queries, and provide clear, concise, and professional answers in Hebrew.
            Summarize the main subjects appearing in these texts.
            Group them into 3-5 topics with short names and descriptions.

            Feedbacks:
            {joined_text}
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return response.choices[0].message.content.strip()

