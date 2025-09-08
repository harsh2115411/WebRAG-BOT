# sec_28_chatbot_streamlit_ui2.py

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from bs4 import SoupStrainer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ------------------ CACHED DOCUMENT PROCESSING ------------------
@st.cache_resource(show_spinner=False)
def load_and_process_url(url: str):
    only_content = SoupStrainer("p")
    loader = WebBaseLoader(url, bs_kwargs={"parse_only": only_content})
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
    document_chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(document_chunks, embeddings)

    return db

# ------------------ SETUP CHATBOT ------------------
def setup_chatbot(url: str):
    db = load_and_process_url(url)
    memory = ConversationBufferMemory(return_messages=True)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=st.secrets["GROQ_API_KEY"]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful chatbot. Use the context: {context} and answer with your best ability."),
        MessagesPlaceholder(variable_name="memory"),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()
    ret = create_retrieval_chain(retriever, document_chain)

    return ret, memory

# ------------------ STREAMLIT APP ------------------
def main():
    st.set_page_config(page_title="Chatbot", layout="centered", page_icon="💬")
    
    # Header
    st.markdown("<h1 style='text-align:center; color:#4B8BBE;'> WebRAG-Bot -> Chat with WebPages</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#555;'>Enter a URL below and chat with the content!</p>", unsafe_allow_html=True)
    st.markdown("---")  # separator

    # URL input
    url = st.text_input("Enter a URL to load content:", "")

    # Styling chat bubbles
    st.markdown(
    """
    <style>
    .chat-container {
        max-width: 800px;
        margin: auto;
        padding: 10px;
    }
    .user-msg {
        background-color: #0066CC;  /* Streamlit blue */
        color: #FFFFFF;  /* White text for contrast */
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 8px;
        margin-left: 30%;  /* Align to right like messaging apps */
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-weight: 500;
    }
    .bot-msg {
        background-color: #F0F2F6;  /* Light gray matching Streamlit theme */
        color: #262730;  /* Dark text for contrast */
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 8px;
        margin-right: 30%;  /* Align to left */
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 3px solid #FF4B4B;  /* Streamlit red accent */
    }
    .user-msg::before {
        content: "👤 You: ";
        font-weight: bold;
        opacity: 0.8;
    }
    .bot-msg::before {
        content: "🤖 Bot: ";
        font-weight: bold;
        color: #FF4B4B;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        padding: 10px;
        border: 2px solid #F0F2F6;
        transition: border-color 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #0066CC;
        box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.2);
    }
    .spacer {
        margin-bottom: 20px;
    }
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .bot-msg {
            background-color: #2D3748;
            color: #E2E8F0;
            border-left: 3px solid #FF4B4B;
        }
        .user-msg {
            background-color: #0066CC;
            color: #FFFFFF;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    if url:
        if "chatbot" not in st.session_state:
            with st.spinner("Processing URL, this may take a few seconds..."):
                ret, memory = setup_chatbot(url)
                st.session_state.chatbot = ret
                st.session_state.memory = memory
                st.session_state.history = []

        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

        # Chat UI
        user_input = st.chat_input("Ask Question...")
        if user_input:
            st.session_state.history.append(("user", user_input))

            answer = st.session_state.chatbot.invoke({
                "input": user_input,
                "memory": st.session_state.memory.chat_memory.messages
            })
            bot_response = answer.get("answer", "Sorry, I couldn’t find an answer.")
            st.session_state.history.append(("bot", bot_response))

            # Update memory
            st.session_state.memory.chat_memory.add_user_message(user_input)
            st.session_state.memory.chat_memory.add_ai_message(bot_response)

        # Display chat history
        for role, message in st.session_state.history:
            if role == "user":
                st.markdown(f"<div class='user-msg'>{message}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-msg'>{message}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
