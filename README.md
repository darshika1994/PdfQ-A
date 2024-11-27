# Document Q&A Bot

## Overview:
This web application allows users to upload a PDF document and ask questions related to its content. Using language models and vector embeddings, the bot processes the document and returns relevant answers based on the uploaded file.

## Features:
- **PDF Upload:** Users can upload any PDF document for analysis.
- **Question Answering:** Ask questions based on the uploaded document’s content.
- **Context-Aware Responses:** Provides accurate answers based on the specific content of the PDF.

## Requirements:
- Python 3.7+
- Streamlit
- Langchain
- Google Generative AI
- FAISS
- PyPDF2
- dotenv

You can install the required dependencies by running:
```bash
pip install -r requirements.txt
```

Run the app using:
```bash
streamlit app run.py
```
