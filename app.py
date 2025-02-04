import streamlit as st
from PyPDF2 import PdfReader # Keep import, though not used, but commented out original functionality
from langchain.text_splitter import RecursiveCharacterTextSplitter # Keep import, though not used, but commented out original functionality
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Keep import, now used for embeddings
from langchain.vectorstores import FAISS # Keep import, now used for vector store
from langchain_google_genai import ChatGoogleGenerativeAI # Keep import, though not used directly in chain, model is used
from langchain.chains.question_answering import load_qa_chain # Keep import, though not used directly in chain
from langchain.prompts import PromptTemplate # Keep import, now used for prompt template
from dotenv import load_dotenv
from functools import lru_cache # Keep import, now used for caching similarity search
import time
from typing import List, Tuple # Keep import, though not used, but commented out original functionality
import numpy as np # Keep import, though not used, but commented out original functionality
import joblib # Keep import, though not used, but commented out original functionality
import pdf2image # Keep import, though not used, but commented out original functionality
import pytesseract # Keep import, though not used, but commented out original functionality
from PIL import Image # Keep import, though not used, but commented out original functionality
import io # Keep import, though not used, but commented out original functionality

import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_embeddings(): # Keep: Function to get embeddings - needed for similarity search
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def extract_text_from_image(image): # Keep import and function definition, but commented out functionality
    """Extract text from image using OCR"""
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def get_pdf_text(pdf_docs): # Keep import and function definition, but commented out functionality
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

def batch_process_chunks(chunks: List[str], batch_size: int = 5) -> List[str]: # Keep import and function definition, but commented out functionality
    processed_chunks = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        processed_chunks.extend(batch)
        time.sleep(1)
    return processed_chunks

def get_text_chunks(text: str) -> List[str]: # Keep import and function definition, but commented out functionality
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return batch_process_chunks(chunks)

def get_vector_store(text_chunks: List[str]): # Keep import and function definition, but commented out functionality
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

@lru_cache(maxsize=100) # Keep: Function to perform cached similarity search - needed for context retrieval
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


# --- Streaming Response Function (from cheagpt) ---
def stream_response(model, prompt, answer_placeholder): # Keep: Streaming function
    """Streams the response from the model and displays it in Streamlit with letter-by-letter effect, returns full response."""
    full_response = ""
    response_stream = model.generate_content(prompt, stream=True)
    for chunk in response_stream:
        text_chunk = chunk.text or ""
        for char in text_chunk:
            full_response += char
            answer_placeholder.markdown(full_response + "▌", unsafe_allow_html=False)
            time.sleep(0.001)
    answer_placeholder.markdown(full_response, unsafe_allow_html=False)
    return full_response

# --- Generative Model Function (from cheagpt, adapted) ---
@lru_cache(maxsize=1) # Keep: Generative model function
def get_generative_model():
    """Return a generative model."""
    model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-1219') # or try 'gemini-1.5-flash' or 'gemini-pro'
    return model

# --- Modified process_question Function (adapted from cheagpt, now with context) ---
def process_question(question, answer_placeholder): # Keep: Process question function - now uses context and streaming
    """Generate an answer with streaming, now using document context."""
    question_with_instructions = question + ". Provide a comprehensive and informative explanation. If there is any relevant link or reference, please provide that as well. **Do not give information which is incorrect or incomplete.** **Format the response clearly and concisely.** **Ensure the answer is accurate and addresses all aspects of the question.** **Avoid vague or generic responses.** **Strictly do not reference the source document explicitly (e.g., avoid phrases like 'the provided documentation', 'the document says' or 'text mentions' or similar meaning phrases) instead answer as if you are answering from you own knowledge, if any link or reference is given then give it enclosed in some text where when clicked, it opens the link, also show what it is for.** **Strictly Present the information naturally and independently, as if explaining from knowledge.** **Use headings or bullet points to improve readability.** **Include tables if they add clarity or structure to the answer.** **If information is limited, state 'Based on the available information...' and provide the most relevant details.**"

    docs = cached_similarity_search(question) # Keep: Retrieve relevant docs using similarity search
    context_text = "\n\n".join([doc.page_content for doc in docs]) # Keep: Format context

    prompt_template = """
    Answer the question using the provided context.  **Your response MUST be formatted as TEXT with Markdown elements, NOT as a Markdown code block.**

    **CRITICAL FORMATTING RULES - FOLLOW THESE EXACTLY:**

    **1.  NO CODE BLOCKS!  ABSOLUTELY DO NOT USE ANY CODE BLOCKS!**  Do not enclose *any* part of your response in triple backticks (```) or single backticks (`). Your output must be rendered directly as Markdown, not as code.

    2.  Format your answer beautifully in Markdown as follows for TEXT ELEMENTS ONLY:

        - **Headings:** Use ### for headings and only headings, not for normal text.
        - **Subheadings:** Use #### for subheadings and only subheadings, not for normal text.
        - **Normal Text:** Write normal text without any special characters (except for Markdown formatting).
        - **Details:** Use bullet points (- ) or numbered lists (1. ) for clarity.
        - **New Lines:** Use double new lines (\\n\\n) to separate paragraphs and sections.
        - **Emphasis:** Use ** for bold and _ for italics to highlight important information.

    3.  If the answer is not available in the provided context, clearly state:
        "The exact answer to this question is not available in the documentation. Here are some relevant details:"
        Then explain the relevant details in a clear and concise manner using the Markdown formatting rules above.

    Context: {context}

    Question: {question}

    Answer:
    """

    prompt = prompt_template.format(context=context_text, question=question_with_instructions) # Keep: Include context in prompt

    model = get_generative_model()
    full_response = stream_response(model, prompt, answer_placeholder) # Keep: Stream response
    return full_response


def main():
    # Initialize history in session state if not present
    if "history" not in st.session_state:
        st.session_state["history"] = []

    st.set_page_config(page_title="Chat with PDFs") # Keep: Page title

    # with st.sidebar: # Commented out: PDF upload sidebar
    #     st.title('Upload PDFs Here')
    #     pdf_docs = st.file_uploader(
    #         'Upload your PDF Files and Click on Process', accept_multiple_files=True)

    #     if st.button("Process"):
    #         with st.spinner("Processing..."):
    #             raw_text = get_pdf_text(pdf_docs)
    #             text_chunks = get_text_chunks(raw_text)
    #             get_vector_store(text_chunks)
    #         st.success("Done")

    # Sidebar input for chat (exactly as before)
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

        # Display existing history *before* adding new question and answer
        for idx, (q, a) in enumerate(st.session_state["history"], start=1):
            q_html = f'<div style="text-align: right;">{q}</div>'
            st.markdown("#### " + q_html, unsafe_allow_html=True)
            if a:
                st.markdown(a, unsafe_allow_html=True)

        q_html_new = f'<div style="text-align: right;">{original_user_question}</div>'
        st.markdown("#### " + q_html_new, unsafe_allow_html=True)

        answer_placeholder = st.empty()

        with st.spinner(text="Thinking..."):
            answer = process_question(user_question, answer_placeholder)

            # Append the Q&A to history AFTER getting the full answer
            st.session_state["history"].append((original_user_question, answer))


    # ------------------- COMMENTED OUT ORIGINAL FUNCTIONS FROM SCANNED PDF CHATBOT -------------------
# @lru_cache(maxsize=1)
# def get_conversational_chain():
#     prompt_template = """
#     Answer the question using the provided context. Ensure the response is comprehensive and beautifully formatted in Markdown as follows:
#     - **Headings:** Use ### for headings and only headings, not for normal text.
#     - **Subheadings:** Use #### for subheadings and only subheadings, not for normal text.
#     - **Normal Text:** Write normal text without any special characters.
#     - **Details:** Use bullet points (- ) or numbered lists (1. ) for clarity.
#     - **New Lines:** Use double new lines (\\n\\n) to separate paragraphs and sections.
#     - **Emphasis:** Use ** for bold and _ for italics to highlight important information.


#     If the question is generic and not specific  to the context, provide a general answer yourself.
#     If the answer is not available in the provided context and also not generic, clearly state:
#     "The exact answer to this question is not available in the documentation. Here are some relevant details:"
#     Then explain the relevant details in a clear and concise manner.

#     Context: {context}

#     Question: {question}

#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain


# def user_input(user_question: str):
#     try:
#         docs = cached_similarity_search(user_question)
#         chain = get_conversational_chain()
#         response = chain(
#             {"input_documents": docs, "question": user_question},
#             return_only_outputs=True
#         )
#         return response["output_text"]
#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}")
#         return None


    # # Chat input area
    # col1, col2 = st.columns([1, 3]) # Adjust ratio as needed, answers on left, questions wider right
    # with col2:
    #     user_question = st.text_input("Ask questions about your PDF Documents:")
    #     if user_question:
    #         original_user_question = user_question
    #         with st.spinner("Thinking..."):
    #             process_question(user_question) # Modified: Removed resources argument
    #             # Append the Q&A to history after processing (streaming is handled inside process_question)
    #             st.session_state["history"].append((original_user_question, st.session_state.get("last_answer", ""))) # placeholder for answer


    # # Display the conversation history - Answers on Left, Questions on Right
    # for idx, (q, a) in enumerate(st.session_state["history"], start=1):
    #     if q and a: # Only display if both question and answer are available
    #         answer_html = f'<div style="text-align: left;">{a}</div>'
    #         question_html = f'<div style="text-align: right;">{q}</div>'
    #         with col1:
    #             st.markdown(answer_html, unsafe_allow_html=True)
    #         with col2:
    #             st.markdown("#### " + question_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()