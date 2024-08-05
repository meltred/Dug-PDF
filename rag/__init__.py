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
#
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


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



system_prompt = ("You are an assistant for question-answering tasks. "
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
    ])# Run chain



with st.sidebar:
        st.title(":blue[A Retrieval Augmented System on the 'Leave No Context Behind' Paper]")
st.title(":blue[💬Document Chatbot]")
query = st.text_area("Enter your query:", placeholder="Enter your query here...", height=100)

if st.button("Submit Your Query"):
    if query:
        if uploaded_file is not None:
            question_answer_chain = create_stuff_documents_chain(model, prompt)
            rag_chain = create_retrieval_chain(vector_index, question_answer_chain)
            for chunk in rag_chain.stream({"input": query}):
                 if answer_chunk := chunk.get("answer"):
                    st.write(f"{answer_chunk}|", end="")
        else:
            st.warning("Please upload a pdf file.")
    else:
        st.warning("Please enter a question.")
