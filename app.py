import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from io import BytesIO
import tempfile
import os

# Load environment variables
dotenv_path = load_dotenv()

# Streamlit page setup
st.set_page_config(page_title="RAG with Gemini", layout="wide")
st.title("📚 RAG Application built on Gemini Model")

# Initialize session state variables
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []  # track uploaded filenames

# Sidebar: document upload
st.sidebar.header("1. Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF/TXT files", type=["pdf", "txt"], accept_multiple_files=True
)

# Sidebar: chunk size and k
chunk_size = st.sidebar.number_input(
    "Chunk size (chars)", min_value=500, max_value=5000, value=1000, step=100
)
k = st.sidebar.slider("Retriever k", min_value=1, max_value=20, value=10)

# Detect change in uploaded files and reset vectorstore if needed
current_file_names = [f.name for f in uploaded_files] if uploaded_files else []
if current_file_names != st.session_state.uploaded_file_names:
    st.session_state.uploaded_file_names = current_file_names
    st.session_state.vectorstore = None
    st.session_state.chat_history = []

# Function to load and embed documents
@st.cache_resource(show_spinner=False)
def build_vectorstore(files, chunk_size):
    docs = []
    for uploaded in files:
        suffix = ".pdf" if uploaded.type == "application/pdf" else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded.getvalue())
            tmp_path = tmp_file.name
        try:
            if uploaded.type == "application/pdf":
                loader = PyPDFLoader(tmp_path)
                docs.extend(loader.load())
            else:
                text = open(tmp_path, "r", encoding="utf-8").read()
                docs.append(type("Doc", (), {"page_content": text}))
        finally:
            os.remove(tmp_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        timeout=120,
        batch_size=20
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="rag_collection"
    )
    return vectorstore

# Build vectorstore when needed
if uploaded_files and st.session_state.vectorstore is None:
    with st.spinner("Embedding documents and building vector store…"):
        st.session_state.vectorstore = build_vectorstore(uploaded_files, chunk_size)
    st.sidebar.success("Documents embedded! Now go to 'Ask a question'.")

# Chat interface
if st.session_state.vectorstore:
    st.header("2. Ask a question")
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-pro-exp-02-05",
        temperature=0,
        max_tokens=None,
        timeout=120
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following retrieved context to answer the question. "
        "If you don't know, say so. Keep answers under 15 sentences.\n\n"
        "{context}"
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Display chat history
    for role, msg in st.session_state.chat_history:
        st.chat_message(role).write(msg)

    # Input for new question
    query = st.chat_input("Say something:")
    if query:
        st.session_state.chat_history.append(("user", query))
        st.chat_message("user").write(query)
        with st.spinner("Generating answer…"):
            qa_chain = create_stuff_documents_chain(llm, prompt_template)
            rag_chain = create_retrieval_chain(retriever, qa_chain)
            response = rag_chain.invoke({"input": query})
        answer = response["answer"]
        st.session_state.chat_history.append(("assistant", answer))
        st.chat_message("assistant").write(answer)

# Footer
st.write("---")
st.write("Built with Streamlit, LangChain & Gemini")