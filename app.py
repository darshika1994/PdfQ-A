import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

## load the GROQ And OpenAI API KEY 
groq_api_key=os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Chat with documents 📚", page_icon="📚")
st.sidebar.image("logo-PDF-Analyzer-website.png", caption="Smart PDF Explorer", use_container_width=True)

# Add custom CSS
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #4CAF50;
        font-family: 'Arial', sans-serif;
    }
    .custom-box {
        background-color: #f9f9f9;
        border: 2px solid #4CAF50;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("<h1 class='main-title'>Interactive PDF Explorer</h1>", unsafe_allow_html=True)

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.loader=PyPDFDirectoryLoader("./Research_Papers") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings


# Add this to initialize the session state at the start of your script
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False  # Initialize session state flag

# Update your button logic
if st.button("Please first Prepare Documents for Q&A"):
    with st.spinner("Processing documents..."):
        try:
            vector_embedding()  # Your document processing logic
            st.session_state.docs_processed = True
        except Exception as e:
            st.session_state.docs_processed = False
            st.error(f"An error occurred: {e}")

# Display confirmation message
if st.session_state.docs_processed:
    st.write("Documents processed!✅")

prompt1=st.text_input("Ask a question about the selected document")

import time

def load_pdf_files(folder_path):
    """Load PDF files from a given folder."""
    return [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

# Sidebar for PDF selection
folder_path = "./Research_Papers"  # Folder containing PDFs
st.sidebar.title("PDF Document Viewer")
pdf_files = load_pdf_files(folder_path)

if pdf_files:
    selected_pdf = st.sidebar.selectbox("Please select a research paper", pdf_files)

    # You can add any functionality here when a PDF is selected, e.g., load the document
else:
    st.sidebar.write("No PDFs found in the folder!")

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")



