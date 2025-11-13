import streamlit as st
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from routers.query_router import QueryRouter

query_router = QueryRouter()

st.title("Telecom Feedback Assistant")
st.write("Ask your question below:")

user_query = st.text_input("Your query:")
if st.button("Submit"):
    if user_query:
        response = query_router.handle_user_query(user_query)
        st.success(response)
