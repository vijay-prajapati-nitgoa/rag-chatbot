import streamlit as st
import os
import tempfile
import time

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("🤖 Chat with Your PDFs (ObjectBox + Llama3)")

# ------------------ API KEY ------------------
groq_api_key = os.getenv("GROQ_API_KEY")

# ------------------ LLM ------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# ------------------ PROMPT ------------------
prompt = ChatPromptTemplate.from_template(
"""
You are a helpful AI assistant.

Answer ONLY from the provided context.
If answer is not in context, say "I don't know".

<context>
{context}
</context>

Question: {input}

Give a clear and structured answer.
"""
)

# ------------------ FILE UPLOAD ------------------
uploaded_files = st.file_uploader(
    "📂 Upload your PDF files",
    type="pdf",
    accept_multiple_files=True
)

# ------------------ VECTOR STORE ------------------
def vector_embedding():
    if "vectors" not in st.session_state:

        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
            return

        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        docs = []

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                loader = PyPDFLoader(tmp.name)
                docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        final_documents = text_splitter.split_documents(docs)

        st.session_state.vectors = ObjectBox.from_documents(
            final_documents,
            st.session_state.embeddings,
            embedding_dimensions=384
        )

        st.success("✅ Documents embedded successfully!")

# ------------------ BUTTON ------------------
if st.button("📥 Process Documents"):
    vector_embedding()

# ------------------ CHAT HISTORY ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------ USER INPUT ------------------
user_input = st.chat_input("💬 Ask something about your documents...")

if user_input:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please process documents first!")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # ------------------ RAG CHAIN ------------------
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_input})
    end = time.process_time()

    answer = response['answer']

    # ------------------ OUTPUT ------------------
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(f"⏱ Response time: {end - start:.2f} sec")

        # Show sources
        with st.expander("📄 Source Documents"):
            for doc in response["context"]:
                st.write(doc.page_content[:500])
                st.write("------")

    st.session_state.messages.append({"role": "assistant", "content": answer})
