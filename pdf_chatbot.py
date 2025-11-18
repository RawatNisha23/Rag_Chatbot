import os
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document, SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from config import *
import re
from create_faiss_index import create_faiss_index
from data_processor import find_equipment_in_pdf, get_documents_for_equipment


def build_prompt_from_retirved_doc(docs, question):
    promt_text = []
    promt_text.append("You are a technical assistant. Answer ONLY from the context below.")
    promt_text.append("If the answer is not in the context, reply 'I don't know'.")
    promt_text.append("\nCONTEXT:")
    for i, context in enumerate(docs, start=1):
        data = context.metadata
        src = data.get("source_pdfs", data.get("source_pdf", "unknown"))
        pages = data.get("pages", data.get("page", "?"))
        page_content = context.page_content.strip()[:4000]
        promt_text.append(f"\n--- CHUNK {i} (src: {src}, pages: {pages}) ---\n{page_content}")
    promt_text.append("\nQUESTION:")
    promt_text.append(question.strip())
    promt_text.append("\nGive a short answer and then list SOURCES with chunk numbers.")
    return "\n".join(promt_text)


def pdf_chatbot(index_folder=FAISS_FOLDER, top_k=6):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENROUTER_API_KEY, openai_api_base=OPENROUTER_BASE_URL)

    index = FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)

    llm = ChatOpenAI(model=LLM_MODEL,temperature=LLM_TEMPERATURE,openai_api_key=OPENROUTER_API_KEY,openai_api_base=OPENROUTER_BASE_URL)

    print("Index loaded... Ask questions (type 'exit' to quit).")

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("Thank You!")
            break

        equipment_tag = find_equipment_in_pdf(question)
        if equipment_tag:
            docs = get_documents_for_equipment(index, embeddings, equipment_tag, question, top_k=top_k)
            if not docs:
                docs = index.similarity_search(question, k=top_k)
        else:
            docs = index.similarity_search(question, k=top_k)

        if not docs:
            print("\nAnswer:\nI don't know (no relevant documents found).")
            continue

        prompt_text = build_prompt_from_retirved_doc(docs, question)

        messages = [
            SystemMessage(content="You are a technical assistant. Use only the provided context."),
            HumanMessage(content=prompt_text)
        ]

        response = llm.invoke(messages)
        answer_text = getattr(response, "content", str(response))

        print("\nAnswer:\n", answer_text)
        # for d in docs:
        #     print(" -", d.metadata)


if __name__ == "__main__":
    if not os.path.exists(FAISS_FOLDER):
        print("FAISS Index not found, Creating new FAISS index...")
        create_faiss_index()
    pdf_chatbot()
       