import google.generativeai as genai
import dotenv
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from io import BytesIO
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from streamlit_pdf_viewer import pdf_viewer

dotenv.load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=key)

@st.cache_data
def process_pdf(uploaded_file):
    pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
    
    texts = []
    for page_num, page in enumerate(pdf_reader.pages, start=1):
        text = page.extract_text()
        texts.append({"page": page_num, "content": text})
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = []
    for item in texts:
        split_texts = text_splitter.split_text(item["content"])
        chunks.extend([{"page": item["page"], "content": chunk} for chunk in split_texts])
    
    return chunks

def create_vector_index(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [{"page": chunk["page"]} for chunk in chunks]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

def main():
    st.title(":blue[💬Document Chatbot]")
    
    uploaded_file = st.file_uploader('Choose a PDF file', type='pdf')
    if uploaded_file:
        with st.spinner("Uploading PDF..."):
            chunks = process_pdf(uploaded_file)
        st.success("PDF uploaded successfully!")
        
        pdf_viewer_key = "pdf_viewer_" + str(hash(uploaded_file.name))
        
        with st.spinner("Processing.."):
            vector_store = create_vector_index(chunks)
        st.success("Now I Know Everything!")
        
        model = ChatGoogleGenerativeAI(
            streaming=True, 
            model="gemini-2.0-flash", 
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
        
        retriever = vector_store.as_retriever()
        question_answer_chain = create_stuff_documents_chain(model, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        query = st.text_area("Enter your query:", placeholder="Enter your query here...", height=100)
        
        if st.button("Submit Your Query"):
            if query:
                with st.spinner("Generating answer..."):
                    answer_container = st.empty()
                    full_answer = ""
                    context_pages = set()
                    for chunk in rag_chain.stream({"input": query}):
                        if answer_chunk := chunk.get("answer"):
                            full_answer += answer_chunk
                            answer_container.write(full_answer)
                        if context := chunk.get("context"):
                            for doc in context:
                                context_pages.add(doc.metadata.get("page"))
                    
                    st.write(f"Debug: Context from pages: {context_pages}")
                    
                    if context_pages:
                        page_to_show = min(context_pages)  # Show the earliest page used as context
                        st.sidebar.write(f"Navigating to page {page_to_show}")
                        with st.sidebar:
                            pdf_viewer(key=pdf_viewer_key, input=uploaded_file.getvalue(),
                                       width=1000, height=1000, scroll_to_page=page_to_show)
                    else:
                        st.sidebar.write("Couldn't determine the context page.")
                        with st.sidebar:
                            pdf_viewer(key=pdf_viewer_key, input=uploaded_file.getvalue(),
                                       width=1000, height=1000)
            else:
                st.warning("Please enter a question.")
    else:
        st.warning("Please upload a PDF file.")

if __name__ == "__main__":
    main()
