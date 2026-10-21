import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

os.environ.setdefault(
    "USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    return Chroma.from_documents(chunks, get_embeddings())


# Page Config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! Paste a website URL in the sidebar and ask me anything about it."),
    ]

# Sidebar for URL input
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")
    st.caption("Paste the full URL, then wait for the site to load.")

if not website_url:
    st.info("Please enter a website URL to start chatting.")
else:
    # Load vector store if URL changes
    if st.session_state.get("loaded_url") != website_url:
        with st.spinner("Loading website and building index..."):
            st.session_state.vector_store = get_vectorstore_from_url(website_url)
            st.session_state.loaded_url = website_url
            st.session_state.chat_history = [
                AIMessage(content="Website loaded successfully! Ask me anything."),
            ]

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.5,
    )

    # User Input
    user_query = st.chat_input("Type your message here...")
    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.spinner("Thinking..."):
            # Direct similarity search (Bilkul PDF code ki tarah)
            docs = st.session_state.vector_store.similarity_search(user_query, k=4)
            context = "\n\n".join(doc.page_content for doc in docs)

            prompt = f"""Answer the user's question based only on the context below from the website.
If the answer is not in the context, say I'm sorry, but I couldn't find the answer to that question on the provided website.

Context:
{context}

Question: {user_query}
"""
            response = llm.invoke(prompt)
            answer = response.text if hasattr(response, 'text') else str(response.content)

        st.session_state.chat_history.append(AIMessage(content=answer))

    # Render Chat History
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)





