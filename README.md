# PDF Q&A Tool

This **PDF Q&A Tool** is an interactive web application that allows users to query information from PDF documents using natural language. It loads PDFs, processes them into embeddings, and then answers user queries based on the content of these documents.

## How It Works

1. **Document Loading**: The tool loads PDFs from a designated folder using the `PyPDFDirectoryLoader` class, which extracts the text from the files.
   
2. **Document Splitting**: The content of the PDFs is split into manageable chunks using the `RecursiveCharacterTextSplitter`. This helps in processing large documents more effectively.

3. **Embedding**: The content chunks are converted into numerical vectors (embeddings) using Google's Generative AI Embeddings model. These vectors represent the documents in a high-dimensional space, enabling efficient similarity search.

4. **Question Answering**: When a user asks a question, the system retrieves the most relevant document chunks using the FAISS vector store. It then processes the retrieved information and provides an answer to the user's query.

5. **UI Interaction**: The user interacts with the app via a simple interface built with Streamlit. Users can select a PDF from the sidebar, input their query, and receive an answer based on the content of the document.

## Model Used

The tool uses the **Groq** language model, specifically the **Llama3-8b-8192** model, for generating responses based on the provided document context.

## Features

- **PDF Loading**: Load PDF documents from a designated folder.
- **Interactive Querying**: Ask questions based on the content of the selected PDF.
- **Document Similarity Search**: View document sections relevant to the queried content.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/darshika1994/PdfQ-A.git
   ```
   
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
3. Set up environment variables:
  - GROQ_API_KEY: API key for Groq service.
  - GOOGLE_API_KEY: API key for Google Generative AI services.

4. To run the app, use:
   ```bash
   streamlit app.py
   ```
