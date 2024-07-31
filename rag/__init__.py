import google.generativeai as genai
import pdfplumber
import dotenv
import os
import time
import streamlit as st
dotenv.load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=key)
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from TextSplitter import Splitter
from GenerateEmb import GenerateEmbedding


uploaded_file = st.file_uploader('chooose a pdf file', type='pdf')
if uploaded_file is not None:

        model = ChatGoogleGenerativeAI(streaming=True, model = "gemini-pro", google_api_key = key,callbacks=[StreamingStdOutCallbackHandler()], temperature=0.2, convert_system_message_to_human=True)
        temp_file = "./temp.pdf"
        with open(temp_file, "wb") as file:
           file.write(uploaded_file.getvalue())
           file_name = uploaded_file.name

        pdf_loader =PyPDFLoader(temp_file)
        pages = pdf_loader.load_and_split()
        # print(pages[0].page_content)
        texts = Splitter(pages)
        vector_index = GenerateEmbedding(texts)
        print("again")

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
def chain(text):
    qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=vector_index,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
    result = qa_chain({"query": text})
    return result["result"]


with st.sidebar:
        st.title(":blue[A Retrieval Augmented System on the 'Leave No Context Behind' Paper]")
st.title(":blue[💬Document Chatbot]")
query = st.text_area("Enter your query:", placeholder="Enter your query here...", height=100)

if st.button("Submit Your Query"):
    if query:
        response = chain(query)
        st.write(response)
    else:
        st.warning("Please enter a question.")
