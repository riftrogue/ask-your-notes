import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from docx import Document as DocxReader
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


st.set_page_config(page_title="Ask Your Notes", layout="centered")
st.title("🧠 Ask Your Notes")
uploaded_file = st.file_uploader("Upload a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"])

#Embeddings uploaded ifles
class MyEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, text):
        return self.model.encode([text])[0]

# Main workflow
if uploaded_file:
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load file based on extension
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "txt":
            loader = TextLoader(tmp_path)
            docs = loader.load()
        elif ext == "pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        elif ext == "docx":
            docx_file = DocxReader(tmp_path)
            full_text = "\n".join(para.text.strip() for para in docx_file.paragraphs if para.text.strip())
            docs = [Document(page_content=full_text, metadata={"source": uploaded_file.name})]

        # Delete temp file
        os.unlink(tmp_path)

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        st.info(f"📚 Document split into {len(chunks)} chunks.")

        #embeddings wrapper
        local_model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_wrapper = MyEmbeddings(local_model)

        with st.spinner("🔢 Generating embeddings..."):
            texts = [doc.page_content for doc in chunks]
            _ = local_model.encode(texts, show_progress_bar=True)

        #FAISS
        vectorstore = FAISS.from_documents(chunks, embedding_wrapper)
        vectorstore.save_local("notes_index")
        st.success(f"✅ Stored {len(chunks)} vectors in FAISS and saved to disk.")

        #Query Section
        st.markdown("---")
        st.subheader("💬 Ask a question from your notes")

        query = st.text_input("Type your question here:")

        if query:
            try:
                # Load saved FAISS index
                vectorstore = FAISS.load_local("notes_index", embedding_wrapper, allow_dangerous_deserialization=True)

                # Retrieve top relevant chunks
                with st.spinner("🔍 Searching notes..."):
                    top_docs = vectorstore.similarity_search(query, k=4)
                    context = "\n\n".join([doc.page_content for doc in top_docs])

               
                # LLM
                llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name="llama3-70b-8192"
                )

                # Define prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful academic assistant. Use the notes to answer student questions clearly."),
                    ("user", "Context:\n{context}\n\nQuestion: {query}")
                ])

                chain = prompt | llm

                # Get answer from LLM
                with st.spinner("🧠 Thinking..."):
                    response = chain.invoke({"context": context, "query": query})
                    st.success("✅ Answer:")
                    st.markdown(response.content)

            except Exception as e:
                st.error(f"❌ Error while processing query: {e}")

    except Exception as e:
        st.error(f"❌ Failed to load or process file: {e}")
