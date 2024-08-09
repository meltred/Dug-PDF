import google.generativeai as genai
import pdfplumber
import dotenv
import os
import time
import streamlit as st
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from GenerateEmb import GenerateEmbedding
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from io import BytesIO
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter

dotenv.load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=key)

@st.cache_data
def process_pdf(uploaded_file):
    pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
    
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    texts = text_splitter.split_text(text)
    
    return texts

def create_vector_index(texts):
    # This function will create the vector index when needed
    return GenerateEmbedding(texts)

def main():
    st.title(":blue[💬Document Chatbot]")
    
    with st.sidebar:
        st.title(":blue[A Retrieval Augmented System on the 'Leave No Context Behind' Paper]")
    
    uploaded_file = st.file_uploader('Choose a PDF file', type='pdf')
    
    if uploaded_file is not None:
        with st.spinner("Uploading PDF..."):
            texts = process_pdf(uploaded_file)
        st.success("PDF uploaded successfully!")
        
        # Create the vector index here, not inside the cached function
        with st.spinner("Processing.."):
            vector_index = create_vector_index(texts)
        st.success("Now I Know Everything!")
        
        model = ChatGoogleGenerativeAI(
            streaming=True, 
            model="gemini-pro", 
            google_api_key=key,
            callbacks=[StreamingStdOutCallbackHandler()], 
            temperature=0.2, 
            convert_system_message_to_human=True
        )
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(model, prompt)
        rag_chain = create_retrieval_chain(vector_index, question_answer_chain)
        
        query = st.text_area("Enter your query:", placeholder="Enter your query here...", height=100)
        
        if st.button("Submit Your Query"):
            if query:
                with st.spinner("Generating answer..."):
                    answer_container = st.empty()
                    full_answer = ""
                    for chunk in rag_chain.stream({"input": query}):
                        if answer_chunk := chunk.get("answer"):
                            full_answer += answer_chunk
                            answer_container.write(full_answer)
            else:
                st.warning("Please enter a question.")
    else:
        st.warning("Please upload a PDF file.")

if __name__ == "__main__":
    main()
