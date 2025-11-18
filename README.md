## RAG PDF Chatbot
This project implements a Retrieval Augmented Generation (RAG) chatbot that can answer questions from PDF.
The chatbot reads all PDFs from the pdfs folder, extracts text using OCR, groups the text by equipment, creates embeddings, stores them in a FAISS index, and then answers user queries using an LLM through OpenRouter.

## Features
- Extracts text from PDFs using **PyMuPDF (fitz)** + **Tesseract OCR**
- Detects equipment IDs such as:
      `T70-C-0102`
      `P-1203 A/B`
- Groups related pages belonging to the same equipment
- Splits large documents into manageable chunks
- Creates semantic embeddings using OpenRouter
- Stores vectors using FAISS
- Answers user's natural language queries
- Retrieves only equipment-specific chunks for better accuracy

### Models Used
Embeddings:
text-embedding-3-small

LLM (free model):
openai/gpt-oss-20b:free


## Project Structure
RAG_CHATBOAT/
│
├── pdfs/
│   ├── PDF_1.pdf
│   ├── PDF_2.pdf
│   └── PDF_3.pdf
│
├── create_faiss_index.py
├── data_processor.py
├── pdf_chatbot.py (Main File)
├── config.py
├── README.md
├── requirements.txt
├── .env
└── Assignment.pdf


## Installation & Setup

### Create a virtual environment (using conda) and install dependencies

```bash
conda create -n rag-chat python=3.11
conda activate rag-chat

## Install dependencies
pip install -r requirements.txt

```

### Install Tesseract OCR

- Download from:
      https://github.com/UB-Mannheim/tesseract/wiki
- After installation, set the Tesseract path in scripts:
      - pytesseract.pytesseract.tesseract_cmd = r"Path of tesseract.exe"

### Create .env file and add OpenRouter API key

OPENROUTER_API_KEY=your_key_here

## Run the Chatbot

```bash
      python pdf_chatbot.py
```
The above script will:
- Create FAISS index if not exists (only at first time run)
      - Read all PDF pages (OCR)
      - Detect equipment IDs
      - Group pages into equipment-specific documents
      - Split large documents into chunks
      - Build FAISS index 
- Accept user questions
- Retrieve relevant chunks
- Generate correct answers using LLM

### sample conversation example for validation
what is the maximum operating temperature of T70-C-0102?
A: 280 oF
• Q: What is the operating temperature of P-1203 A/B?
A: 39 oC
• Q: What is the Material of Construction of Shell of T70-C-0102?
A: Base material is SA-516 GR 70N and the cladding is SS 316 L


## Design Summary (How It Works)
1. OCR Extraction
      - Each PDF page is converted into an image using PyMuPDF
      - Tesseract extracts text accurately from scanned pages

2. Equipment Detection
-     Regex identifies equipment IDs inside each page.

3. Page Grouping
      - All pages belonging to the same equipment are merged into a single long document.

4. Chunk Splitting
      - Large documents are split into 1500–2000 character chunks so embeddings remain meaningful.

5. FAISS Indexing
      - All chunks are embedded and stored in a FAISS vector index.

6. Retrieval + LLM
      - Detect equipment ID from user question
      - Retrieve only relevant chunks
      - Build a clean prompt
      - Send context + question to an OpenRouter LLM
      - Return a concise, source-backed answer
