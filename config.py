
import os
from dotenv import load_dotenv
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Lenovo\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY","")
OPENROUTER_BASE_URL =  os.getenv("OPENROUTER_BASE_URL","")
PDF_FOLDER = "pdfs"
FAISS_FOLDER = "faiss_store"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
OCR_DPI = 300
LLM_MODEL = "openai/gpt-oss-20b:free"
LLM_TEMPERATURE = 0.0