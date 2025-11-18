import re
import io
import fitz
import pytesseract
from PIL import Image
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import *
from langchain_community.vectorstores import FAISS

def find_equipment_in_pdf(text):
    if not text:
        return None
    text_flat = text.replace("\n", " ")
    patterns = [
        r"\bT\d{2}-[A-Z]-\d{4}\b",
        r"\bT\d{2}-C-\d{4}\b",
        r"\bP-\d{4}\s*[A-Z]?\/?[A-Z]?\b",
        r"\bP\d{4}\b",
    ]
    for val in patterns:
        match = re.search(val, text_flat, flags=re.IGNORECASE)
        # print("match-->",match)
        if match:
            return match.group(0).upper().strip()
    return None

def extract_pdf_text_ocr(pdf_path, dpi = OCR_DPI):
    documents = []
    pdf_name = os.path.basename(pdf_path)
    pdf = fitz.open(pdf_path)
    for page_index in range(len(pdf)):
        page = pdf.load_page(page_index)
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        page_text = pytesseract.image_to_string(image)
        equipment_id = find_equipment_in_pdf(page_text)
        documents.append(
            Document(
                page_content=page_text,
                metadata={
                    "source_pdf": pdf_name,
                    "page": page_index,
                    "equipment_id": equipment_id
                }
            )
        )
    pdf.close()
    return documents

def organize_pages_by_equipment(page_docs):
    groups = {}
    last_equipment = None
    for doc in page_docs:
        equipment = doc.metadata.get("equipment_id")
        pdfname = doc.metadata.get("source_pdf")
        if equipment:
            last_equipment = equipment
        else:
            equipment = last_equipment if last_equipment else f"FILE_{pdfname}"
        # print("last_equipment-->",last_equipment)
        if equipment not in groups:
            groups[equipment] = {"pages": [], "texts": [], "pdfs": set()}
        groups[equipment]["pages"].append(doc.metadata.get("page"))
        groups[equipment]["texts"].append(doc.page_content)
        groups[equipment]["pdfs"].add(pdfname)
    # print("groups-->",groups)
    merged = []
    for eq_id, data in groups.items():
        text_combined = "\n\n".join(data["texts"]).strip()
        merged.append(
            Document(
                page_content=text_combined,
                metadata={
                    "equipment_id": eq_id,
                    "pages": sorted(set(data["pages"])),
                    "source_pdfs": sorted(list(data["pdfs"]))
                }
            )
        )
    return merged

def split_large_documents(equipment_docs, chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    final_chunks = []
    for doc in equipment_docs:
        if len(doc.page_content) <= chunk_size * 2:
            final_chunks.append(doc)
        else:
            chunk = splitter.split_documents([doc])
            for val in chunk:
                # print("val.metadata-->",val.metadata)
                val.metadata.update(doc.metadata)
            final_chunks.extend(chunk)
    return final_chunks

def get_documents_for_equipment(index, embeddings, equipment_id, query, top_k = 5):
    stored_docs = list(index.docstore._dict.values())
    filtered = [doc for doc in stored_docs if doc.metadata.get("equipment_id") == equipment_id]
    # print("filtered-->",filtered)
    if not filtered:
        return []
    filtered_index = FAISS.from_documents(filtered, embeddings)
    result = filtered_index.similarity_search(query, k=top_k)
    return result
