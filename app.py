import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from functools import lru_cache
import time
from typing import List, Tuple
import numpy as np
import joblib
import pdf2image
import pytesseract
from PIL import Image
import io

import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Cache directory setup
# CACHE_DIR = "cache_data"
# if not os.path.exists(CACHE_DIR):
#     os.makedirs(CACHE_DIR)

# Cache utility functions
# def save_cache_to_disk(cache_name: str, data):
#     cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
#     joblib.dump(data, cache_path)

# def load_cache_from_disk(cache_name: str):
#     cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
#     if os.path.exists(cache_path):
#         return joblib.load(cache_path)
#     return None

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def extract_text_from_image(image):
    """Extract text from image using OCR"""
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def get_pdf_text(pdf_docs):
    """Extract text from PDFs by always converting to images first"""
    text = ""
    for pdf in pdf_docs:
        try:
            # Convert PDF to images
            pdf_bytes = pdf.read()
            images = pdf2image.convert_from_bytes(pdf_bytes)
            
            # Process each image with OCR
            for image in images:
                page_text = extract_text_from_image(image)
                text += page_text + "\n"
            
            # Reset file pointer for potential future use
            pdf.seek(0)
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            continue
            
    return text

def batch_process_chunks(chunks: List[str], batch_size: int = 5) -> List[str]:
    processed_chunks = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        processed_chunks.extend(batch)
        time.sleep(1)
    return processed_chunks

def get_text_chunks(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return batch_process_chunks(chunks)

# @lru_cache(maxsize=1000)
# def cached_embedding(text_chunk: str) -> Tuple[float]:
#     try:
#         embeddings = get_embeddings()
#         time.sleep(1)
#         return tuple(embeddings.embed_query(text_chunk))
#     except Exception as e:
#         print(f"Error in embedding: {e}")
#         time.sleep(60)
#         return tuple(embeddings.embed_query(text_chunk))

# def batch_create_embeddings(text_chunks: List[str], batch_size: int = 3):
#     all_embeddings = []
#     embeddings = get_embeddings()

#     for i in range(0, len(text_chunks), batch_size):
#         batch = text_chunks[i:i + batch_size]
#         try:
#             batch_embeddings = [cached_embedding(chunk) for chunk in batch]
#             all_embeddings.extend(batch_embeddings)
#             time.sleep(2)
#         except Exception as e:
#             print(f"Error in batch embedding: {e}")
#             time.sleep(60)
#             batch_embeddings = [cached_embedding(chunk) for chunk in batch]
#             all_embeddings.extend(batch_embeddings)

#     save_cache_to_disk('embeddings', all_embeddings)
#     return all_embeddings

def get_vector_store(text_chunks: List[str]):
    try:
        embeddings = get_embeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_vector_store")
    except Exception as e:
        print(f"Error in vector store creation: {e}")
        time.sleep(60)
        embeddings = get_embeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_vector_store")

@lru_cache(maxsize=1)
def get_conversational_chain():
    prompt_template = """
    Answer the question using the provided context. Ensure the response is comprehensive and beautifully formatted in Markdown as follows:
    - **Headings:** Use ### for headings and only headings, not for normal text.
    - **Subheadings:** Use #### for subheadings and only subheadings, not for normal text.
    - **Normal Text:** Write normal text without any special characters.
    - **Details:** Use bullet points (- ) or numbered lists (1. ) for clarity.
    - **New Lines:** Use double new lines (\\n\\n) to separate paragraphs and sections.
    - **Emphasis:** Use ** for bold and _ for italics to highlight important information.

    
    If the question is generic and not specific  to the context, provide a general answer yourself.
    If the answer is not available in the provided context and also not generic, clearly state:
    "The exact answer to this question is not available in the documentation. Here are some relevant details:"
    Then explain the relevant details in a clear and concise manner.

    Context: {context}

    Question: {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

@lru_cache(maxsize=100)
def cached_similarity_search(question: str):
    try:
        embeddings = get_embeddings()
        new_db = FAISS.load_local("faiss_vector_store", embeddings, allow_dangerous_deserialization=True)
        return new_db.similarity_search(question)
    except Exception as e:
        print(f"Error in similarity search: {e}")
        time.sleep(60)
        return new_db.similarity_search(question)

def user_input(user_question: str):
    try:
        docs = cached_similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        st.write(response["output_text"])
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        time.sleep(60)

def main():
    st.set_page_config("PDF Chatbot📚")
    st.header("Chat with PDFs📚")
   
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_question += """
        . Provide a comprehensive and informative explanation.
        **Do not give information which is incorrect or incomplete.**
        **Format the response clearly and concisely.**
        **Ensure the answer is accurate and addresses all aspects of the question.**
        **Use headings or bullet points to improve readability.**
        **Include tables if they add clarity or structure to the answer.**
        **If information is limited, state 'Based on the available information...' and provide the most relevant details.**
    """
        user_input(user_question)
   
    # with st.sidebar:
    #     st.title("Menu:")
    #     pdf_docs = st.file_uploader(
    #         "Upload your PDF Files and Click on the Submit & Process Button",
    #         accept_multiple_files=True
    #     )
       
    #     if st.button("Submit & Process"):
    #         with st.spinner("Processing..."):
    #             try:
    #                 raw_text = get_pdf_text(pdf_docs)
    #                 text_chunks = get_text_chunks(raw_text)
    #                 get_vector_store(text_chunks)
    #                 st.success("Done")
    #             except Exception as e:
    #                 st.error(f"An error occurred during processing: {str(e)}")
    #                 time.sleep(60)

if __name__ == "__main__":
    main()