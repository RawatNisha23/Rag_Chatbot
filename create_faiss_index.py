from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config import *
from data_processor import extract_pdf_text_ocr, organize_pages_by_equipment, split_large_documents



def create_faiss_index(pdf_folder=PDF_FOLDER, index_folder=FAISS_FOLDER):
    all_document = []
    for file in sorted(os.listdir(pdf_folder)):
        if not file.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_folder, file)
        page_docs = extract_pdf_text_ocr(path)
        all_document.extend(page_docs)
    equipment_docs = organize_pages_by_equipment(all_document)
    final_chunks = split_large_documents(equipment_docs)
    # print("OPENROUTER_API_KEY--->",OPENROUTER_API_KEY, OPENROUTER_BASE_URL)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENROUTER_API_KEY, openai_api_base=OPENROUTER_BASE_URL)
    index = FAISS.from_documents(final_chunks, embeddings)
    index.save_local(index_folder)
    print("FAISS index saved in", index_folder)

# #if __name__ == "__main__":
#  #   if not os.path.exists(FAISS_FOLDER):
#  #       create_faiss_index()
#  #   else:
#  #       print("FAISS folder already exists:", FAISS_FOLDER)

