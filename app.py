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
    new_db = None
    try:
        embeddings = get_embeddings()
        new_db = FAISS.load_local("faiss_vector_store", embeddings, allow_dangerous_deserialization=True)
        return new_db.similarity_search(question)
    except Exception as e:
        print(f"Error in similarity search: {e}")
        time.sleep(60)
        return new_db.similarity_search(question) if new_db else []

def user_input(user_question: str):
    try:
        docs = cached_similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        return response["output_text"]
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def main():
    # Initialize history in session state if not present
    if "history" not in st.session_state:
        st.session_state["history"] = []

    st.set_page_config(page_title="Chat with AI")

    # Sidebar input for chat
    st.sidebar.title("Chat with AI")
    user_question = st.sidebar.text_input("Ask Questions Here:")
    if user_question:
        original_user_question = user_question
        user_question += """
        . Provide a comprehensive and informative explanation.
        **Do not give information which is incorrect or incomplete.**
        **Format the response clearly and concisely.**
        **Ensure the answer is accurate and addresses all aspects of the question.**
        **Use headings or bullet points to improve readability.**
        **Include tables if they add clarity or structure to the answer.**
        **If information is limited, state 'Based on the available information...' and provide the most relevant details.**
        """
        answer = user_input(user_question)
        if answer:
            # Append the Q&A to history
            st.session_state["history"].append((original_user_question, answer))

    # Display the conversation history
    for idx, (q, a) in enumerate(st.session_state["history"], start=1):
        q_html = f'<div style="text-align: right;">{q}</div>'
        st.markdown("#### " + q_html, unsafe_allow_html=True)
        st.markdown(a, unsafe_allow_html=True)

if __name__ == "__main__":
    main()