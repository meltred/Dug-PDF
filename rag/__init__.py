import google.generativeai as genai
import dotenv
import os

print(dotenv.load_dotenv())
key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=key)
import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model = "gemini-pro", google_api_key = key, temperature=0.2, convert_system_message_to_human=True)

pdf_loader =PyPDFLoader("attention.pdf")
pages = pdf_loader.load_and_split()
# print(pages[1].page_content)

from langchain_google_genai import GoogleGenerativeAIEmbeddings

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
context = "\n\n".join(str(p.page_content) for p in pages)
texts = text_splitter.split_text(context)


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=key)


vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})

     
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question = "Describe the Multi-head attention layer in detail?"
result = qa_chain({"query": question})
print(result["result"])

