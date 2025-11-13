import os
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv() 

class OpenAIGetQueryResponse:

    client = OpenAI(
        api_key= os.getenv("OPENAI_GPT_4O_API_KEY")
    )

  
    @staticmethod
    def retrieve_answer(user_query: str, context: str):

        chat_response = None

        SYSTEM_PROMPT = """ 
        You are a helpful, intelligent, and empathetic customer service assistant for a telecom company. 
        Your goal is to understand customer feedback and queries, and provide clear, concise, and professional answers in Hebrew.
        Use the provided context to help formulate your response, but only rely on relevant information.
        If the context doesn’t contain the answer, use your general knowledge to assist the customer.
        ==== CONTEXT ====
        {context}
    
        ==== CUSTOMER QUESTION ====
        {user_query}
   
        Now write a friendly and accurate response in Hebrew that addresses the customer's question,
        using the information from the context when appropriate.
        Keep your tone clear, polite, and customer-focused.
        """
        
        try:
            response = OpenAIGetQueryResponse.client.chat.completions.create(
                        model=os.getenv("OPENAI_GPT_4O_MINI_DEPLOYMENT_NAME"),
                        temperature=0.7,
                        top_p=0.95,
                        max_tokens=500,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None,
                        stream=False,
                        messages=[
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": SYSTEM_PROMPT.format(context=context, user_query=user_query)
                                }
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"{user_query}"
                                }
                            ]
                        }
                    ]
                )
            chat_response = response.choices[0].message.content
            if chat_response is None:
                chat_response = None
        except Exception as err:
                    print(f"ChatGPT call failed: {err}")
        
        return chat_response

