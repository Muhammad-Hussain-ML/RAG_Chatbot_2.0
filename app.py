import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from qdrant_client.http.models import SearchParams
from qdrant_client import QdrantClient
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import os

# Initialize session state for chat history and unique_id tracking
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "last_unique_id" not in st.session_state:
    st.session_state.last_unique_id = None


def query_embedding(query, api_key):
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    return embeddings_model.embed_query(query)

def search_related_text(query_embedding, unique_id, collection_name, top_k=3):
    search_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        search_params=SearchParams(hnsw_ef=128),
        limit=top_k
    )
    return [
        result.payload["text"]
        for result in search_results.points
        if result.payload.get("unique_id") == unique_id
    ]

def generate_response(llm, related_texts, user_query):
    memory = st.session_state.chat_history  # Use session memory
    conversation_history = st.session_state.chat_history.chat_memory.messages

    # conversation_history = memory.chat_memory.messages
    formatted_history = "\n".join([
        f"User: {message.content}" if isinstance(message, HumanMessage) else f"Assistant: {message.content}"
        for message in conversation_history
    ])
    
    st.success(f"{formatted_history}\n")
    if related_texts:
        formatted_text = "\n".join(related_texts)
        prompt = f"""
        You are an interactive assistant who answers questions in a friendly and conversational tone, just like a real person from the company. 
        Your responses should sound natural, concise, warm, and engaging—like you're having a chat with the user. 
        Imagine you're speaking directly to the user, just like how a colleague from the company would interact with them. 
        If there’s no answer, politely let the user know.
        
        Here’s the relevant information that you should keep in mind:
        {formatted_text}

        Here's the conversation history so far:
        {formatted_history}

        Now, answer the user's question in a way that makes them feel like they're talking to a real person from the company. Feel free to offer additional insights or ask follow-up questions if needed. The user's query is:
        {user_query}
        """
    else:
        prompt = f"""
        You are an interactive assistant who answers questions in a friendly and conversational tone, just like a real person from the company. 
        If there’s no relevant information to answer the user’s question, kindly inform them that you don’t have the necessary details and encourage them to ask something else.

        Unfortunately, we don't have relevant information available for your query at the moment. Please feel free to ask something else!

        Here's the conversation history so far:
        {formatted_history}

        Now, answer the user's question in a way that makes them feel like they're talking to a real person from the company. Feel free to offer additional insights or ask follow-up questions if needed. The user's query is:
        {user_query}
        """

    response = llm.invoke(prompt)
    return response.content.strip()

def list_unique_ids_in_collection(qdrant_client, collection_name, limit=100):
    unique_ids = set()
    next_page_offset = None

    while True:
        points, next_page_offset = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=limit,
            offset=next_page_offset,
        )

        for point in points:
            if "unique_id" in point.payload:
                unique_ids.add(point.payload["unique_id"])

        if next_page_offset is None:
            break

    return list(unique_ids)

def pipeline(api_key, qdrant_client, collection_name, user_query, unique_id, top_k=2):
    query_embeddings = query_embedding(user_query, api_key)
    related_texts = search_related_text(query_embeddings, unique_id, collection_name, top_k=top_k)
    # st.write(related_texts)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.6,
        google_api_key=api_key
    )
    response = generate_response(llm, related_texts, user_query)

    st.session_state.chat_history.chat_memory.add_user_message(user_query)
    st.session_state.chat_history.chat_memory.add_ai_message(response)

    return response

# Streamlit UI
st.title("AI Query Pipeline")
st.write("Looking for specific information? Type your question and select the Hospital ID (Name) to get results instantly!")

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
collection_name = "new_documents_practice"
api_key = os.getenv("GOOGLE_API_KEY")

with st.spinner("Fetching Hospital Names..."):
    try:
        hospitals = list_unique_ids_in_collection(qdrant_client, collection_name)
        if not hospitals:
            hospitals = ["No hospitals found."]
    except Exception as e:
        st.error(f"Error fetching Hospital IDs: {e}")
        hospitals = ["Error fetching Hospital ID"]

user_query = st.text_input("Enter your Query:")
unique_id = st.selectbox("Select Hospital ID/Name:", options=hospitals)

# here change
if unique_id != st.session_state.last_unique_id:
    st.session_state.chat_history = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.last_unique_id = unique_id 
    
if st.button("Run Query"):
    if api_key and qdrant_client and collection_name and user_query:
        try:
            with st.spinner("Processing your query..."):
                response = pipeline(api_key, qdrant_client, collection_name, user_query, unique_id)
            st.write("Assistant:", response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please provide all the required inputs.")
